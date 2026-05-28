#pragma once

// ============================================================================
// cuda-kernel-lab : SGEMM optimization ladder
//
// All kernels compute the row-major product  C = alpha * (A * B) + beta * C
//   A is M x K, B is K x N, C is M x N, all row-major, all float32.
//
// The ladder progresses naive -> coalesced -> shared-memory tiled ->
// 1D block-tiled -> 2D block-tiled. Each rung adds exactly one idea and is
// kept in the tree so the progression is visible. Tile and block dimensions
// are named constants below; there are no magic numbers in the kernel bodies.
//
// These are textbook implementations of the well-known SGEMM optimization
// sequence; the value here is the measured ladder, not novelty.
// ============================================================================

#include <cstdint>

// ---- Tunable launch constants (no magic numbers in kernel bodies) ----------
namespace cfg {
// Rungs 1-2: one thread per output, 32x32 = 1024 threads per block.
constexpr uint32_t NAIVE_BLOCK = 32;
constexpr uint32_t COALESCE_BLOCK = 32;

// Rung 3: square shared-memory tile.
constexpr uint32_t SMEM_TILE = 32;

// Rung 4: 1D block tile. Block computes a BM x BN output tile, each thread a
// column of TM results. Threads = BM*BN/TM.
constexpr uint32_t BT1_BM = 64;
constexpr uint32_t BT1_BN = 64;
constexpr uint32_t BT1_BK = 8;
constexpr uint32_t BT1_TM = 8;

// Rung 5: 2D block tile. Each thread computes a TM x TN register tile.
// Threads = BM*BN/(TM*TN). Tuned for the A100 (large L2, plenty of SMEM).
constexpr uint32_t BT2_BM = 128;
constexpr uint32_t BT2_BN = 128;
constexpr uint32_t BT2_BK = 16;
constexpr uint32_t BT2_TM = 8;
constexpr uint32_t BT2_TN = 8;
}  // namespace cfg

// ============================================================================
// Rung 1: sgemm_naive
//   Optimization added: none. Baseline.
//   One thread computes one C element with a plain K-loop over global memory.
//   threadIdx.x maps to the row, so the 32 threads of a warp walk down a
//   column of C and access A with stride K and C with stride N: every global
//   load is uncoalesced. This is the floor everything else is measured against.
// ============================================================================
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < (uint32_t)M && col < (uint32_t)N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

// ============================================================================
// Rung 2: sgemm_coalesced
//   Optimization added: global-memory coalescing.
//   Same arithmetic as naive, but threads are reindexed so consecutive
//   threadIdx.x map to consecutive columns. Now the 32 threads of a warp read
//   consecutive addresses of B and write consecutive addresses of C (one
//   128-byte transaction instead of 32 scattered ones), and the A access is a
//   broadcast of one value. Expect a large jump over naive.
// ============================================================================
template <const uint32_t BLOCKSIZE>
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float *A, const float *B, float beta,
                                float *C) {
  const uint32_t row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint32_t col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
  if (row < (uint32_t)M && col < (uint32_t)N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

// ============================================================================
// Rung 3: sgemm_smem_tiled
//   Optimization added: shared-memory tiling for data reuse.
//   The block loads a TILE x TILE block of A and of B into shared memory,
//   syncs, then every thread computes its partial dot product out of shared
//   memory before advancing along K. Each global element is now read once per
//   tile and reused TILE times, cutting global traffic by ~TILE. Expect a
//   further jump because the kernel stops refetching from global memory.
//   Assumes M, N, K are multiples of TILE (true for the benchmark sweep).
// ============================================================================
template <const uint32_t TILE>
__global__ void sgemm_smem_tiled(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
  __shared__ float As[TILE * TILE];
  __shared__ float Bs[TILE * TILE];

  const uint32_t cRow = blockIdx.x;
  const uint32_t cCol = blockIdx.y;
  const uint32_t threadRow = threadIdx.x / TILE;
  const uint32_t threadCol = threadIdx.x % TILE;

  // Move pointers to the top-left of this block's working tiles.
  A += cRow * TILE * K;
  B += cCol * TILE;
  C += cRow * TILE * N + cCol * TILE;

  float acc = 0.0f;
  for (int bk = 0; bk < K; bk += TILE) {
    // Coalesced load of one tile of A and one tile of B into shared memory.
    As[threadRow * TILE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * TILE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();

    A += TILE;
    B += TILE * N;

    for (uint32_t k = 0; k < TILE; ++k)
      acc += As[threadRow * TILE + k] * Bs[k * TILE + threadCol];
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * acc + beta * C[threadRow * N + threadCol];
}

// ============================================================================
// Rung 4: sgemm_1d_blocktile
//   Optimization added: 1D register blocking (more work per thread).
//   Each thread now computes TM output rows in a register accumulator array
//   instead of a single element. One value of B loaded from shared memory is
//   reused across all TM results, so the ratio of FMAs to shared-memory loads
//   rises (higher arithmetic intensity). Threads per block = BM*BN/TM.
//   Assumes M, N, K are multiples of the tile dims (true for the sweep).
// ============================================================================
template <const uint32_t BM, const uint32_t BN, const uint32_t BK,
          const uint32_t TM>
__global__ void sgemm_1d_blocktile(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const uint32_t cRow = blockIdx.y;
  const uint32_t cCol = blockIdx.x;

  // Output coordinate this thread owns (a column of TM elements).
  const uint32_t threadRow = threadIdx.x / BN;
  const uint32_t threadCol = threadIdx.x % BN;

  // Indices used to cooperatively load the A and B tiles. BM*BK == BK*BN ==
  // blockDim.x for these constants, so each thread loads exactly one element.
  const uint32_t innerRowA = threadIdx.x / BK;
  const uint32_t innerColA = threadIdx.x % BK;
  const uint32_t innerRowB = threadIdx.x / BN;
  const uint32_t innerColB = threadIdx.x % BN;

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  float threadResults[TM] = {0.0f};

  for (int bk = 0; bk < K; bk += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint32_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
      const float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint32_t resIdx = 0; resIdx < TM; ++resIdx)
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
    }
    __syncthreads();
  }

  for (uint32_t resIdx = 0; resIdx < TM; ++resIdx) {
    const uint32_t r = threadRow * TM + resIdx;
    C[r * N + threadCol] =
        alpha * threadResults[resIdx] + beta * C[r * N + threadCol];
  }
}

// ============================================================================
// Rung 5: sgemm_2d_blocktile
//   Optimization added: 2D register blocking.
//   Each thread computes a TM x TN tile of C held entirely in registers. For
//   each step along BK it loads TM values of A and TN values of B into
//   registers (regM, regN) and does TM*TN FMAs, so each shared-memory load
//   feeds TM (or TN) multiplies. This is the highest arithmetic intensity rung
//   and typically reaches a strong fraction of cuBLAS.
//   Threads per block = BM*BN/(TM*TN). Loads are strided so the block can be
//   larger than the thread count. Assumes M, N, K multiples of the tile dims.
// ============================================================================
template <const uint32_t BM, const uint32_t BN, const uint32_t BK,
          const uint32_t TM, const uint32_t TN>
__global__ void sgemm_2d_blocktile(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const uint32_t cRow = blockIdx.y;
  const uint32_t cCol = blockIdx.x;

  constexpr uint32_t numThreads = (BM * BN) / (TM * TN);

  // Each thread owns a TM x TN tile within the BM x BN block tile.
  const uint32_t threadCol = threadIdx.x % (BN / TN);
  const uint32_t threadRow = threadIdx.x / (BN / TN);

  // Strided cooperative-load indices: the tiles are larger than numThreads, so
  // each thread loads several elements, advancing by strideA / strideB.
  const uint32_t innerRowA = threadIdx.x / BK;
  const uint32_t innerColA = threadIdx.x % BK;
  constexpr uint32_t strideA = numThreads / BK;
  const uint32_t innerRowB = threadIdx.x / BN;
  const uint32_t innerColB = threadIdx.x % BN;
  constexpr uint32_t strideB = numThreads / BN;

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int bk = 0; bk < K; bk += BK) {
    for (uint32_t loadOffset = 0; loadOffset < BM; loadOffset += strideA)
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    for (uint32_t loadOffset = 0; loadOffset < BK; loadOffset += strideB)
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint32_t dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint32_t i = 0; i < TM; ++i)
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      for (uint32_t i = 0; i < TN; ++i)
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      for (uint32_t resM = 0; resM < TM; ++resM)
        for (uint32_t resN = 0; resN < TN; ++resN)
          threadResults[resM * TN + resN] += regM[resM] * regN[resN];
    }
    __syncthreads();
  }

  for (uint32_t resM = 0; resM < TM; ++resM)
    for (uint32_t resN = 0; resN < TN; ++resN) {
      const uint32_t r = threadRow * TM + resM;
      const uint32_t c = threadCol * TN + resN;
      C[r * N + c] =
          alpha * threadResults[resM * TN + resN] + beta * C[r * N + c];
    }
}
