#!/usr/bin/env python3
"""Turn results/sgemm_results.csv into results/sgemm_chart.png.

Reads the CSV written by bench_sgemm.py (skipping its '#' header block), and
draws GFLOPS vs matrix size, one line per kernel, with cuBLAS pedantic FP32 as
the reference. The GPU name from the header is used as the chart subtitle so
the figure is self-describing.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "results", "sgemm_results.csv")
PNG = os.path.join(ROOT, "results", "sgemm_chart.png")

# Plot the ladder + pedantic cuBLAS. TF32 is omitted from the main chart because
# it is ~6x off-scale (tensor cores, lower precision); it stays in the CSV.
ORDER = ["naive", "coalesced", "smem_tiled", "1d_blocktile", "2d_blocktile", "cublas"]
LABELS = {
    "naive": "naive",
    "coalesced": "coalesced",
    "smem_tiled": "smem tiled",
    "1d_blocktile": "1D blocktile",
    "2d_blocktile": "2D blocktile",
    "cublas": "cuBLAS (pedantic FP32)",
}


def main():
    meta = {}
    rows = []
    with open(CSV) as f:
        for line in f:
            if line.startswith("#"):
                if ":" in line:
                    k, v = line[1:].split(":", 1)
                    meta[k.strip()] = v.strip()
                continue
            rows = list(csv.DictReader([line] + f.readlines()))
            break

    data = {}  # kernel -> (sizes, gflops)
    for r in rows:
        data.setdefault(r["kernel"], ([], []))
        data[r["kernel"]][0].append(int(r["size"]))
        data[r["kernel"]][1].append(float(r["gflops_mean"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    for kernel in ORDER:
        if kernel not in data:
            continue
        sizes, gflops = data[kernel]
        style = dict(marker="o", linewidth=2)
        if kernel == "cublas":
            style = dict(marker="s", linewidth=2.5, linestyle="--", color="black")
        ax.plot(sizes, gflops, label=LABELS[kernel], **style)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(r["size"]) for r in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Square matrix size (M = N = K)")
    ax.set_ylabel("GFLOPS (mean of 50 timed launches)")
    ax.set_title(f"SGEMM optimization ladder vs cuBLAS\n{meta.get('gpu', '')}",
                 fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PNG, dpi=130)
    print(f"Wrote {PNG}")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401  (referenced above)
    main()
