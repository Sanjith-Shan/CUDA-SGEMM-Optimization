#!/usr/bin/env python3
"""Benchmark and validate the fused softmax / layernorm kernels against torch.

For each op and shape: torch generates a fixed-seed input on the GPU, computes
the reference and times the torch op (CUDA events, warmup + timed), then the
hand-written kernel runs on the same input via file handoff (it times itself
with CUDA events and writes its output). The kernel output is compared to the
torch reference (max abs + normwise relative error), and both bandwidths and the
speedup are reported. Results are written to results/fused_results.csv.

Nothing is fabricated: every row is an actual run.
"""
import csv
import datetime
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "fused_results.csv")

ABS_TOL = 1e-2
REL_TOL = 1e-2
WARMUP = 10
TIMED = 50
# (rows, cols): transformer-ish shapes (tokens x feature/vocab dim).
SHAPES = [(4096, 1024), (8192, 2048), (16384, 4096)]


def torch_time(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(TIMED):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return float(np.mean(times)), float(np.min(times))


def run_op(op, binary, rows, cols):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    x = torch.empty((rows, cols), device=dev, dtype=torch.float32)
    x.uniform_(-3.0, 3.0, generator=gen)

    if op == "softmax":
        ref = torch.softmax(x, dim=-1)
        tmean, tbest = torch_time(lambda: torch.softmax(x, dim=-1))
    elif op == "layernorm":
        ref = torch.nn.functional.layer_norm(x, (cols,))
        tmean, tbest = torch_time(
            lambda: torch.nn.functional.layer_norm(x, (cols,)))
    else:
        raise ValueError(op)

    with tempfile.TemporaryDirectory() as d:
        inP, outP = os.path.join(d, "in.f32"), os.path.join(d, "out.f32")
        x.cpu().numpy().astype(np.float32).tofile(inP)
        proc = subprocess.run([binary, str(rows), str(cols), inP, outP],
                              capture_output=True, text=True, check=True)
        rec = json.loads(proc.stdout.strip().splitlines()[-1])
        got = torch.from_numpy(
            np.fromfile(outP, dtype=np.float32).reshape(rows, cols))

    ref_c = ref.cpu()
    abs_e = float((got - ref_c).abs().max())
    rel_e = abs_e / (float(ref_c.abs().max()) + 1e-30)
    rec["max_abs_err"] = abs_e
    rec["max_rel_err"] = rel_e
    rec["torch_mean_ms"] = tmean
    rec["torch_best_ms"] = tbest
    rec["speedup_mean"] = tmean / rec["mean_ms"]
    rec["correct"] = abs_e < ABS_TOL and rel_e < REL_TOL
    return rec


def main():
    sm = os.path.join(ROOT, "softmax")
    ln = os.path.join(ROOT, "layernorm")
    if not (os.path.exists(sm) and os.path.exists(ln)):
        sys.stderr.write("build the fused binaries first (make fused)\n")
        sys.exit(1)

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    rows_out = []
    failures = 0
    for op, binary in [("softmax", sm), ("layernorm", ln)]:
        for (r, c) in SHAPES:
            rec = run_op(op, binary, r, c)
            rows_out.append(rec)
            if not rec["correct"]:
                failures += 1
            print(f"  {op:<10} {r}x{c:<6}  kernel {rec['gbps_mean']:>7.1f} GB/s  "
                  f"torch {2.0*r*c*4/(rec['torch_mean_ms']*1e-3)/1e9:>7.1f} GB/s  "
                  f"speedup {rec['speedup_mean']:.2f}x  "
                  f"abs_err {rec['max_abs_err']:.2e}  "
                  f"{'OK' if rec['correct'] else 'FAIL'}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT, "w", newline="") as f:
        f.write("# cuda-kernel-lab fused-kernel results\n")
        f.write(f"# generated: {now}\n")
        f.write(f"# gpu: {torch.cuda.get_device_name(0)}\n")
        f.write(f"# torch: {torch.__version__}\n")
        f.write(f"# reference: torch.softmax / torch.nn.functional.layer_norm\n")
        f.write(f"# timing: {WARMUP} warmup + {TIMED} timed CUDA-event launches\n")
        w = csv.writer(f)
        w.writerow(["op", "rows", "cols", "kernel_mean_ms", "kernel_gbps_mean",
                    "torch_mean_ms", "speedup_mean",
                    "max_abs_err", "max_rel_err", "correct"])
        for r in rows_out:
            w.writerow([r["op"], r["rows"], r["cols"],
                        f"{r['mean_ms']:.6f}", f"{r['gbps_mean']:.2f}",
                        f"{r['torch_mean_ms']:.6f}", f"{r['speedup_mean']:.3f}",
                        f"{r['max_abs_err']:.3e}", f"{r['max_rel_err']:.3e}",
                        "true" if r["correct"] else "false"])
    print(f"\nWrote {OUT}")
    if failures:
        sys.stderr.write(f"{failures} shape(s) FAILED correctness\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
