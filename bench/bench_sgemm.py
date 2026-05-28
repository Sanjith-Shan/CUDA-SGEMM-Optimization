#!/usr/bin/env python3
"""Run every SGEMM kernel across the size sweep and write results/sgemm_results.csv.

Invokes the compiled ./sgemm binary (which validates against a pedantic-FP32
cuBLAS reference and times with CUDA events), parses its JSON output, computes
each kernel's percentage of pedantic-FP32 cuBLAS, and writes a CSV whose header
block names the GPU, driver, and CUDA version so the numbers are attributable.

Nothing here is fabricated: every row comes from an actual subprocess run.
"""
import csv
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "sgemm")
OUT = os.path.join(ROOT, "results", "sgemm_results.csv")

# Order matters: this is the ladder, low to high, with the baselines last.
KERNELS = [
    "naive",
    "coalesced",
    "smem_tiled",
    "1d_blocktile",
    "2d_blocktile",
    "cublas",
    "cublas_tf32",
]
SIZES = [256, 512, 1024, 2048, 4096]
BASIS = "cublas"  # percent-of-cuBLAS basis: pedantic FP32


def device_info():
    out = subprocess.check_output([BIN, "--device-info"], text=True).strip()
    return json.loads(out)


def run(kernel, size):
    proc = subprocess.run([BIN, kernel, str(size)], capture_output=True, text=True)
    # The binary exits non-zero only when a *gated* kernel fails correctness.
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not line:
        sys.stderr.write(f"no output for {kernel} {size}: {proc.stderr}\n")
        sys.exit(1)
    rec = json.loads(line)
    if proc.returncode != 0 and rec.get("gated", True):
        sys.stderr.write(
            f"FAIL: {kernel} {size} failed correctness "
            f"(abs={rec['max_abs_err']:.2e} rel={rec['max_rel_err']:.2e})\n"
        )
    return rec


def main():
    if not os.path.exists(BIN):
        sys.stderr.write(f"binary not found: {BIN}\nBuild first (make).\n")
        sys.exit(1)

    info = device_info()
    print(f"GPU: {info['gpu']}  cc {info['cc']}  "
          f"driver {info['driver']}  CUDA {info['cuda_runtime']}")

    # Collect everything first so we can compute percent-of-cuBLAS per size.
    rows = []           # flat list of records
    basis_gflops = {}   # size -> cuBLAS pedantic mean GFLOPS
    for kernel in KERNELS:
        for size in SIZES:
            rec = run(kernel, size)
            rows.append(rec)
            if kernel == BASIS:
                basis_gflops[size] = rec["gflops_mean"]
            print(f"  {kernel:<13} {size:>5}  "
                  f"{rec['gflops_mean']:>10.1f} GFLOPS  "
                  f"abs_err={rec['max_abs_err']:.2e}  "
                  f"{'OK' if rec['correct'] else 'CHECK'}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT, "w", newline="") as f:
        f.write(f"# cuda-kernel-lab SGEMM results\n")
        f.write(f"# generated: {now}\n")
        f.write(f"# gpu: {info['gpu']}\n")
        f.write(f"# compute_capability: {info['cc']}\n")
        f.write(f"# driver: {info['driver']}\n")
        f.write(f"# cuda_runtime: {info['cuda_runtime']}\n")
        f.write(f"# percent_basis: cublas pedantic FP32 (CUBLAS_PEDANTIC_MATH)\n")
        f.write(f"# timing: {10} warmup + {50} timed CUDA-event launches\n")
        w = csv.writer(f)
        w.writerow([
            "kernel", "size", "mean_ms", "best_ms",
            "gflops_mean", "gflops_best",
            "max_abs_err", "max_rel_err", "pct_cublas_mean", "correct",
        ])
        for r in rows:
            pct = 100.0 * r["gflops_mean"] / basis_gflops[r["M"]]
            w.writerow([
                r["kernel"], r["M"],
                f"{r['mean_ms']:.6f}", f"{r['best_ms']:.6f}",
                f"{r['gflops_mean']:.3f}", f"{r['gflops_best']:.3f}",
                f"{r['max_abs_err']:.3e}", f"{r['max_rel_err']:.3e}",
                f"{pct:.2f}", "true" if r["correct"] else "false",
            ])

    print(f"\nWrote {OUT}")
    headline = next(r for r in rows if r["kernel"] == "2d_blocktile" and r["M"] == max(SIZES))
    pct = 100.0 * headline["gflops_mean"] / basis_gflops[max(SIZES)]
    print(f"Headline: 2d_blocktile @ {max(SIZES)} = "
          f"{headline['gflops_mean']:.1f} GFLOPS = {pct:.1f}% of pedantic cuBLAS")


if __name__ == "__main__":
    main()
