import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
METRICS = (
    (
        "time_seconds", "Median execution time (s)",
        "Execution time", "time.png",
    ),
    ("gflops", "GFLOP/s", "Computational performance", "gflops.png"),
    (
        "estimated_gbs", "Estimated effective bandwidth (GB/s)",
        "Estimated bytes / median time", "bandwidth.png",
    ),
)
plt.switch_backend("Agg")


def read_summary(path):
    groups = defaultdict(list)
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            operation, dtype = row["operation"], row["dtype"]
            if operation not in ("matvec", "matmul"):
                raise ValueError("Unknown operation in summary.csv")
            if dtype not in ("float32", "float64"):
                raise ValueError("Unknown precision in summary.csv")
            row["N"] = int(row["N"])
            key = (operation, dtype, row["N"])
            if row["N"] < 1 or key in seen:
                raise ValueError("Invalid size or duplicate summary row")
            seen.add(key)
            for metric, _, _, _ in METRICS:
                row[metric] = float(row[metric])
                if not math.isfinite(row[metric]) or row[metric] <= 0:
                    raise ValueError("Plot values must be positive and finite")
            groups[(operation, dtype)].append(row)
    sizes = sorted({key[2] for key in seen})
    expected = {
        (operation, dtype, n)
        for operation in ("matvec", "matmul")
        for dtype in ("float32", "float64")
        for n in sizes
    }
    if not sizes or seen != expected:
        raise ValueError("Need both operations and precisions at every size")
    return groups, sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "summary.csv")
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("Missing summary.csv; obtain the merged analysis first")
    groups, sizes = read_summary(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, title, filename in METRICS:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        for (operation, dtype), rows in sorted(groups.items()):
            ordered = sorted(rows, key=lambda row: row["N"])
            ax.plot(
                [row["N"] for row in ordered],
                [row[metric] for row in ordered],
                marker="o", label=f"{operation}, {dtype}",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.set_xlabel("Matrix dimension N")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        if metric == "estimated_gbs":
            fig.text(
                0.5, 0.02, "Model-based estimate, not measured RAM bandwidth.",
                ha="center", fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.06, 1, 1))
        else:
            fig.tight_layout()
        destination = args.output / filename
        fig.savefig(destination, dpi=180)
        plt.close(fig)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
