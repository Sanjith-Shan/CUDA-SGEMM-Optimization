#!/usr/bin/env python3
"""Correctness verification for the SGEMM ladder.

Two independent layers:

1. The ./sgemm binary validates every kernel against a pedantic-FP32 cuBLAS
   reference on-device and reports measured max abs / max rel error. This script
   collects those numbers across the full size sweep and applies the project
   pass condition (max abs error < 1e-2 AND max rel error < 1e-2 for float32).

2. A genuinely independent NumPy cross-check via the binary's --check mode:
   NumPy generates A and B, the kernel computes C on the GPU, and the result is
   compared against a float64 NumPy matmul of the same inputs. This is fully
   independent of cuBLAS, guarding against a shared bug in the reference path.

Exits non-zero if any gated kernel fails, so CI / the build plan can gate on it.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "sgemm")

KERNELS = ["naive", "coalesced", "smem_tiled", "1d_blocktile", "2d_blocktile"]
SIZES = [256, 512, 1024, 2048, 4096]
ABS_TOL = 1e-2
REL_TOL = 1e-2


def run(kernel, size):
    proc = subprocess.run([BIN, kernel, str(size)], capture_output=True, text=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def numpy_crosscheck(kernel, size):
    """Independent NumPy reference via --check: NumPy makes inputs, compares C."""
    rng = np.random.default_rng(0)
    M = N = K = size
    A = rng.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    B = rng.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32)
    ref = (A.astype(np.float64) @ B.astype(np.float64))
    with tempfile.TemporaryDirectory() as d:
        aP, bP, cP = (os.path.join(d, f) for f in ("a.f32", "b.f32", "c.f32"))
        A.tofile(aP)
        B.tofile(bP)
        subprocess.run([BIN, "--check", kernel, str(M), str(N), str(K),
                        aP, bP, cP], check=True)
        C = np.fromfile(cP, dtype=np.float32).reshape(M, N).astype(np.float64)
    # Normwise relative error (infinity-norm), matching sgemm.cu.
    abs_e = float(np.max(np.abs(C - ref)))
    rel_e = abs_e / (float(np.max(np.abs(ref))) + 1e-30)
    return abs_e, rel_e


def main():
    if not os.path.exists(BIN):
        sys.stderr.write(f"binary not found: {BIN}\nBuild first (make).\n")
        sys.exit(1)

    print(f"Pass condition: max_abs_err < {ABS_TOL} AND max_rel_err < {REL_TOL} "
          f"(reference: pedantic-FP32 cuBLAS)\n")
    print(f"{'kernel':<14}{'size':>6}{'max_abs_err':>14}{'max_rel_err':>14}  result")
    print("-" * 60)

    failures = 0
    for kernel in KERNELS:
        for size in SIZES:
            rec = run(kernel, size)
            ok = rec["max_abs_err"] < ABS_TOL and rec["max_rel_err"] < REL_TOL
            if not ok:
                failures += 1
            print(f"{kernel:<14}{size:>6}{rec['max_abs_err']:>14.3e}"
                  f"{rec['max_rel_err']:>14.3e}  {'PASS' if ok else 'FAIL'}")

    print("-" * 60)
    print("Independent NumPy cross-check (float64 reference, via --check):")
    for kernel in ["smem_tiled", "2d_blocktile"]:
        abs_e, rel_e = numpy_crosscheck(kernel, 512)
        ok = abs_e < ABS_TOL and rel_e < REL_TOL
        if not ok:
            failures += 1
        print(f"  {kernel:<14} @512  abs={abs_e:.3e} rel={rel_e:.3e}  "
              f"{'PASS' if ok else 'FAIL'}")

    if failures:
        print(f"\n{failures} kernel/size combination(s) FAILED correctness.")
        sys.exit(1)
    print("\nAll kernels PASS across the full size sweep.")


if __name__ == "__main__":
    main()
