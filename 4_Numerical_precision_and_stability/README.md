# Homework 4: Exploring the Roofline Model

We compare NumPy matrix-vector and matrix-matrix multiplication using
`float32` and `float64`. This folder includes the benchmark, recorded timings,
analysis and plotting code.

## How to run

Run these commands from the repository root:

```powershell
python -m pip install numpy matplotlib
python 4_Numerical_precision_and_stability/analyse.py
python 4_Numerical_precision_and_stability/plot_results.py
```

These commands use the saved timings in `results.csv`. The analysis writes
`summary.csv`, and plotting creates `figures/time.png`, `figures/gflops.png`
and `figures/bandwidth.png`.

To collect new timings instead, run this before analysis and plotting:

```powershell
python 4_Numerical_precision_and_stability/benchmark.py --sizes 256 512 1024 2048 4096 --repeats 5
```

This replaces `results.csv`. New timings will depend on the computer and
conditions. Keep the recorded data when reproducing the submitted plots.

## How we measured

We used sizes 256, 512, 1024, 2048 and 4096. The inputs were constant:
`A = 1.0`, `B = 2.0` and `x = 3.0`. Each operation and precision had one
untimed warm-up call and five timed repetitions at each size.

Input and output arrays were created before timing and reused. We timed
the multiplication call, not array setup or CSV writing. This produced
100 measurements. The analysis uses their median, giving 20 summary rows.
In `summary.csv`, `time_seconds` is the median time in seconds.

## Calculations

For square matrices of size `N`, `s` is 4 bytes for `float32` and 8 for
`float64`.

| Operation | FLOPs | Estimated bytes |
|---|---|---|
| Matrix-vector | `N * (2*N - 1)` | `s * (N*N + 2*N)` |
| Matrix-matrix | `N*N * (2*N - 1)` | `s * 3*N*N` |

```text
arithmetic_intensity = flops / estimated_bytes
GFLOP/s = flops / median_time_seconds / 1e9
estimated GB/s = estimated_bytes / median_time_seconds / 1e9
```

The byte estimate assumes each input is read once and each output is
written once, with ideal data reuse. Actual traffic can differ, especially
when data is cached. The bandwidth plot is not a direct measurement of
RAM bandwidth. Both plot axes use logarithmic scales.

The team report uses these results and figures to discuss scaling,
precision, data reuse and the Roofline model. It is submitted separately
through Atenea.
