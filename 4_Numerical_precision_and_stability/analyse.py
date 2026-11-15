import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
FIELDS = [
    "operation", "dtype", "N", "repetitions", "time_seconds",
    "flops", "estimated_bytes", "arithmetic_intensity",
    "gflops", "estimated_gbs",
]


def estimate(operation, dtype, n):
    if n < 1 or dtype not in ("float32", "float64"):
        raise ValueError("Invalid size or dtype")
    size = 4 if dtype == "float32" else 8
    if operation == "matvec":
        flops = n * (2 * n - 1)
        estimated_bytes = size * (n * n + 2 * n)
    elif operation == "matmul":
        flops = n * n * (2 * n - 1)
        estimated_bytes = size * 3 * n * n
    else:
        raise ValueError("Unknown operation")
    return flops, estimated_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "results.csv")
    parser.add_argument("--output", type=Path, default=HERE / "summary.csv")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    groups = defaultdict(dict)
    with args.input.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["operation"], row["dtype"], int(row["N"]))
            estimate(*key)
            repetition = int(row["repetition"])
            seconds = float(row["time_seconds"])
            if repetition < 1 or repetition in groups[key]:
                raise ValueError("Invalid or duplicate repetition")
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Each timing must be positive and finite")
            groups[key][repetition] = seconds
    sizes = {key[2] for key in groups}
    expected = {
        (operation, dtype, n)
        for operation in ("matvec", "matmul")
        for dtype in ("float32", "float64")
        for n in sizes
    }
    if not groups or set(groups) != expected:
        raise ValueError("Need both operations and both precisions at each N")
    counts = {len(values) for values in groups.values()}
    if len(counts) != 1 or min(counts) < 2:
        raise ValueError("Use the same repeated measurements for every case")
    records = []
    for key, values in sorted(groups.items()):
        if set(values) != set(range(1, len(values) + 1)):
            raise ValueError("Some repetition numbers are missing")
        operation, dtype, n = key
        seconds = median(values.values())
        flops, estimated_bytes = estimate(*key)
        records.append({
            "operation": operation, "dtype": dtype, "N": n,
            "repetitions": len(values), "time_seconds": seconds,
            "flops": flops, "estimated_bytes": estimated_bytes,
            "arithmetic_intensity": flops / estimated_bytes,
            "gflops": flops / seconds / 1e9,
            "estimated_gbs": estimated_bytes / seconds / 1e9,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} cases to {args.output}")


if __name__ == "__main__":
    main()
