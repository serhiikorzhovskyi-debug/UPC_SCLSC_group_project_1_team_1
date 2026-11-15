# Homework 4: Exploring the Roofline Model

We compare NumPy matrix-vector and matrix-matrix multiplication using
`float32` and `float64`. The benchmark and analysis are ready. Plotting
and the final report will be added by Student 3.

## How to run

Run these commands from the repository root:

```powershell
python -m pip install numpy
python 4_Numerical_precision_and_stability/benchmark.py --sizes 256 512 1024 2048 4096 --repeats 5
python 4_Numerical_precision_and_stability/analyse.py
```

`benchmark.py` saves the timings in `results.csv`. Running it again
replaces that file, and the new timings may be different.

`analyse.py` reads the timings and saves `summary.csv`. It uses the median
time for each operation, precision and size. The recorded data has 100
measurements, giving 20 summary rows. To analyse the saved data, run only
the last command.

## How we measured

We use constant inputs: `A = 1.0`, `B = 2.0` and `x = 3.0`. Each case has
one untimed warm-up call and five timed repetitions. Input and output
arrays are created before timing and reused. We time the multiplication
call, not array setup or CSV writing.

## Calculations

For square matrices of size `N`, let `s` be 4 bytes for `float32` and
8 bytes for `float64`.

| Operation | FLOPs | Estimated bytes |
|---|---|---|
| Matrix-vector | `N * (2*N - 1)` | `s * (N*N + 2*N)` |
| Matrix-matrix | `N*N * (2*N - 1)` | `s * 3*N*N` |

We assume each input element is read once and each output element is
written once, with ideal data reuse. Actual memory traffic can be different,
so the calculated bandwidth is an estimate, not measured RAM bandwidth.

```text
arithmetic_intensity = flops / estimated_bytes
GFLOP/s = flops / median_time_seconds / 1e9
estimated GB/s = estimated_bytes / median_time_seconds / 1e9
```

In `summary.csv`, `time_seconds` is the median time. Student 3 will use
this file to make the plots and complete the report with the team's
results and discussion.
