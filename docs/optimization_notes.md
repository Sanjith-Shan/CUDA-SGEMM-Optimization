# SGEMM optimization notes

The performance story of the ladder, one section per rung, tied to the measured
jumps in [`../results/sgemm_results.csv`](../results/sgemm_results.csv). All figures
are mean GFLOPS at the headline size **4096×4096×4096** on an **NVIDIA A100-SXM4-80GB**
(CUDA 12.8) unless stated otherwise.

For roofline framing, the relevant A100 ceilings are roughly **19.5 TFLOPS** of
non-tensor-core FP32 and **~2.0 TB/s** of HBM2e bandwidth. The crossover arithmetic
intensity (where a kernel stops being memory bound and starts being compute bound) is
therefore about `19.5e12 / 2.0e12 ≈ 9.7` FLOP/byte. The whole ladder is the story of
pushing arithmetic intensity from far below that line to near it.

C = A·B with A (M×K), B (K×N), C (M×N), all row-major float32, alpha=1, beta=0.

---

## Rung 1 — `sgemm_naive` · 292 GFLOPS · 1.7% of cuBLAS

One thread computes one C element with a plain K-loop reading A and B straight from
global memory. The thread-to-data mapping is the problem: `threadIdx.x` indexes the
**row**, so the 32 threads of a warp walk *down a column* of C. Their reads of B share
the same column (a broadcast) but their reads of A and their writes to C are strided by
K and N respectively — every transaction is **uncoalesced**, scattering what should be
one 128-byte burst into many.

Arithmetic intensity is ~0.25 FLOP/byte (2 FLOPs per two 4-byte loads), so this kernel
is hopelessly **memory bound**, and worse, it cannot even reach HBM bandwidth because
the accesses are uncoalesced. 292 GFLOPS is the floor everything else is measured
against.

## Rung 2 — `sgemm_coalesced` · 2,574 GFLOPS · 14.5% · **8.8× over naive**

Exactly the same arithmetic, one change: threads are reindexed so `threadIdx.x` maps to
the **column**. Now consecutive threads in a warp touch consecutive addresses of B and
of C, so each warp's loads/stores collapse into single coalesced 128-byte transactions,
and the A access becomes a clean per-warp broadcast.

The 8.8× jump is the single largest multiplier on the ladder and comes purely from
*memory access pattern* — no change to the computation or the data layout. Arithmetic
intensity is still ~0.25 FLOP/byte, so the kernel is still memory bound; it simply now
uses the memory system efficiently. (It exceeds the naive AI-roofline estimate because
the L2 cache absorbs the heavily-reused B columns — the first sign that reuse, not raw
bandwidth, is the lever for everything that follows.)

## Rung 3 — `sgemm_smem_tiled` · 5,406 GFLOPS · 30.5% · **2.1× over coalesced**

The first rung that changes *data movement* rather than access pattern. Each block loads
a 32×32 tile of A and of B into shared memory, `__syncthreads()`, and computes a partial
product entirely out of shared memory before advancing along K. A global element is now
read **once per tile and reused 32 times** instead of being refetched from global memory
on every use.

This raises arithmetic intensity from ~0.25 to ~`TILE/4 = 8` FLOP/byte — close to the
~9.7 crossover, so the kernel is no longer purely memory bound. The 2.1× gain is the
payoff of cutting global traffic by ~32×. It does **not** reach the new roofline ceiling,
though: each thread still performs only one FMA per two shared-memory loads, so
*shared-memory throughput* is now the binding constraint. That is precisely what the next
two rungs attack.

## Rung 4 — `sgemm_1d_blocktile` · 10,301 GFLOPS · 58.1% · **1.9× over smem**

Each thread now computes a **column of TM=8 outputs** held in a register accumulator
array, with block tile BM=64, BN=64, BK=8. The key effect: one value loaded from `Bs`
into a register is reused across all 8 results, so the ratio of FMAs to shared-memory
loads improves from ~1:2 to ~8:9. Work per thread goes up, the number of redundant
shared-memory loads goes down, and instruction-level parallelism across the 8 independent
accumulators hides latency.

This is the move from "tiled but shared-memory-bound" to "register-blocked." The 1.9× gain
reflects shared-memory traffic dropping by roughly TM×, pushing arithmetic intensity well
past the roofline crossover — the kernel is now **compute/register bound**, which is where
you want to be.

## Rung 5 — `sgemm_2d_blocktile` · 15,529 GFLOPS · **87.6% of cuBLAS** · 1.5× over 1D

Each thread computes a **TM×TN = 8×8 tile** of C entirely in registers, with block tile
BM=128, BN=128, BK=16, 256 threads per block. Per step along BK, a thread loads TM values
of A and TN values of B into registers (`regM`, `regN`) and issues TM×TN = 64 FMAs — so
each shared-memory load now feeds **8 multiplies** in the inner product, the 2D
generalization of rung 4. The loads from global to shared are strided so the 128×128 block
tile can be staged by only 256 threads.

This is the highest arithmetic-intensity rung and the one that reaches a strong fraction of
cuBLAS: **87.6% of pedantic FP32** at 4096. The remaining ~12% is what cuBLAS buys with
techniques beyond this ladder — double-buffered (software-pipelined) shared-memory loads to
hide latency, `float4` vectorized loads, warp-level tiling matched to the register file, and
per-architecture autotuning.

### Tuning note (honest)

`BK` for this rung was set to **16** rather than the more common 8. With BK=8 the 2D kernel
landed *just below* the 1D kernel at 2048 (a ~3% inversion); BK=16 increases the work staged
per `__syncthreads()` and the reuse of each loaded A/B strip, which removed the inversion and
also improved 4096 (≈15.2k→15.5k GFLOPS). Both numbers are from actual runs; see the Phase 1
commit for the before/after.

### Why the small sizes invert

At 256 and 512 the 1D and 2D rungs fall *below* `smem_tiled`. With a 128×128 block tile, a
256×256 problem is only `2×2 = 4` thread blocks — against the A100's **108 SMs**, that leaves
the GPU almost entirely idle, so occupancy, not arithmetic intensity, decides the result. The
big-tile kernels are built to win on large matrices, and the headline 4096 number is exactly
that regime. This is expected tail behavior, reported rather than hidden.

---

## Summary table (4096, mean GFLOPS)

| Rung | GFLOPS | vs prev | % cuBLAS | Binding constraint |
|------|-------:|--------:|---------:|--------------------|
| naive        |    292 | —    | 1.7%  | uncoalesced global memory |
| coalesced    |  2,574 | 8.8× | 14.5% | global bandwidth |
| smem_tiled   |  5,406 | 2.1× | 30.5% | shared-memory throughput |
| 1d_blocktile | 10,301 | 1.9× | 58.1% | register/compute |
| 2d_blocktile | 15,529 | 1.5× | 87.6% | register/compute (near cuBLAS) |
| cuBLAS FP32  | 17,720 | —    | 100%  | reference |

Profiler-measured occupancy and memory-throughput numbers (Nsight Compute) are added in the
Phase 5 section below once collected, to back this reasoning with hardware counters rather
than first-principles argument alone.
