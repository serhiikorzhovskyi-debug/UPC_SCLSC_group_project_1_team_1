# Homework: Exploring the Roofline Model

This folder contains the NumPy matrix-vector and matrix-matrix experiments
for `float32` and `float64`. Run the following commands from the repository root.

## Install the libraries used by these scripts

```powershell
python -m pip install numpy matplotlib
```

## Run the experiments and reproduce the plots

```powershell
python 4_Numerical_precision_and_stability/benchmark.py --sizes 256 512 1024 2048 4096 --repeats 5
python 4_Numerical_precision_and_stability/analyse.py
python 4_Numerical_precision_and_stability/plot_results.py
```

The sizes and repetition count above are the settings used by these example
commands, not fixed values specified by the assignment. If the team changes
these settings for its submitted measurements, update the commands here too.
Use a range with sufficiently large sizes that your computer can handle.

The benchmark writes `results.csv`. Running it again replaces that file.
Analysis writes `summary.csv`; plotting writes `figures/time.png`,
`figures/gflops.png`, and `figures/bandwidth.png`. All paths are relative to
this homework folder. Analysis and plotting can be run without repeating the
benchmark when the recorded results are already present.

## Implementation and measurement method

For each size, the code runs both operations with both precisions. `A` is
filled with 1.0, `B` with 2.0, and `x` with 3.0. NumPy performs the dense
products; the code does not replace them with formulas for constant inputs.

Input and output arrays are allocated and initialized before timing. The
output is passed to `np.matmul` with `out`. One untimed call is followed by
five individually timed calls. The timed region contains the operation call;
CSV writing and array setup are outside it. The same arrays are reused for
the repetitions. The analysis uses their median execution time. This setup
does not control allocations internal to NumPy or its underlying libraries.

## Calculations

Let `s` be 4 bytes for float32 or 8 bytes for float64. The arithmetic count
uses N multiplications and N-1 additions per dot product:

| Operation | FLOPs | Estimated transferred bytes |
|---|---|---|
| Matrix-vector | `N * (2*N - 1)` | `s * (N*N + 2*N)` |
| Matrix-matrix | `N*N * (2*N - 1)` | `s * 3*N*N` |

The byte model assumes each input element is read once and each output
element is written once, with ideal reuse. It is an estimate, not measured
traffic at a particular memory level. Repeated calls can reuse cached data.

`arithmetic_intensity = flops / estimated_bytes`

`gflops = flops / median_time_seconds / 1e9`

`estimated_gbs = estimated_bytes / median_time_seconds / 1e9`

Actual timings depend on the computer and its software. See the report for
the settings and conditions of the submitted experiment and the discussion
of size, precision, data reuse and the Roofline model.