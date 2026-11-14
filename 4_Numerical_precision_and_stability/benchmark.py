import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
FIELDS = ["operation", "dtype", "N", "repetition", "time_seconds"]


def matvec(a, x, out):
    return np.matmul(a, x, out=out)


def matmul(a, b, out):
    return np.matmul(a, b, out=out)


def measure(operation, dtype, n, repeats):
    a = np.full((n, n), 1.0, dtype=dtype)
    if operation == "matvec":
        other = np.full(n, 3.0, dtype=dtype)
        out = np.zeros(n, dtype=dtype)
        function = matvec
    else:
        other = np.full((n, n), 2.0, dtype=dtype)
        out = np.zeros((n, n), dtype=dtype)
        function = matmul
    function(a, other, out)
    records = []
    for repetition in range(1, repeats + 1):
        start = perf_counter()
        function(a, other, out)
        seconds = perf_counter() - start
        if seconds <= 0:
            raise RuntimeError("Invalid timing; use larger problem sizes")
        records.append({
            "operation": operation,
            "dtype": dtype,
            "N": n,
            "repetition": repetition,
            "time_seconds": seconds,
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=[256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=HERE / "results.csv")
    args = parser.parse_args()
    if min(args.sizes) < 1 or len(set(args.sizes)) != len(args.sizes):
        parser.error("Sizes must be distinct positive integers")
    if args.repeats < 2:
        parser.error("Use at least two repetitions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for n in sorted(args.sizes):
            for operation in ("matvec", "matmul"):
                for dtype in ("float32", "float64"):
                    print(f"{operation}, {dtype}, N={n}", flush=True)
                    records = measure(operation, dtype, n, args.repeats)
                    writer.writerows(records)
                    stream.flush()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
