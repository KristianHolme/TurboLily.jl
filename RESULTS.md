# Benchmarks

## WaterLily-Benchmarks cases on an A10

These are the cases of [WaterLily-Benchmarks](https://github.com/WaterLily-jl/WaterLily-Benchmarks)
(`src/cases.jl` at f6034df), on the GPU only, at every size that has a checkpoint in the suite.
TurboLily runs against the latest registered releases, WaterLily 1.10.0 and
BiotSavartBCs 1.1.0. The machine has 4× NVIDIA A10 (22.5 GB, 600 GB/s). Float32, Julia 1.13.2,
CUDA.jl 6.4.2.

The method is the suite's. Each case runs in its own process and starts from a developed-flow
checkpoint. It warms up for at least 50 steps and 2 s, then does 5 runs of 25 steps, each run
reset to the checkpoint. A step is timed with a device synchronisation, and the time is the
minimum over the runs of the mean step. Both codes load the same checkpoint. The reference and
TurboLily runs of a case ran back to back on the same GPU, with two GPUs in use.

The rows with x-y planes of 40,000 cells or more (tgv p=8, sphere p=5, cylinder p=5) use the larger
block shapes of README section 3.5.

| case | log2p | cells | reference [ms] | TurboLily [ms] | speedup | ref cost [ns/cell] | Turbo cost [ns/cell] |
|---|---:|---:|---:|---:|---:|---:|---:|
| tgv | 6 | 262k | 3.15 | 0.62 | 5.1× | 12.0 | 2.4 |
| tgv | 7 | 2.1M | 11.00 | 2.64 | 4.2× | 5.2 | 1.3 |
| tgv | 8 | 16.8M | 75.85 | 17.37 | 4.4× | 4.5 | 1.0 |
| tgv-periodic | 6 | 262k | 4.27 | 0.56 | 7.6× | 16.3 | 2.2 |
| tgv-periodic | 7 | 2.1M | 12.38 | 2.43 | 5.1× | 5.9 | 1.2 |
| sphere | 3 | 295k | 3.00 | 0.73 | 4.1× | 10.2 | 2.5 |
| sphere | 4 | 2.4M | 11.76 | 2.88 | 4.1× | 5.0 | 1.2 |
| sphere | 5 | 18.9M | 85.15 | 19.94 | 4.3× | 4.5 | 1.1 |
| sphere-biotsavart | 3 | 295k | 4.96 | 1.39 | 3.6× | 16.8 | 4.7 |
| sphere-biotsavart | 4 | 2.4M | 16.80 | 4.34 | 3.9× | 7.1 | 1.8 |
| cylinder | 4 | 442k | 6.76 | 1.35 | 5.0× | 15.3 | 3.1 |
| cylinder | 5 | 3.5M | 22.22 | 6.33 | 3.5× | 6.3 | 1.8 |
| cylinder-biotsavart | 4 | 442k | 13.15 | — | — | 29.7 | — |
| cylinder-biotsavart | 5 | 3.5M | 27.57 | — | — | 7.8 | — |
| donut | 5 | 66k | 3.12 | 0.56 | 5.6× | 47.6 | 8.5 |
| donut | 6 | 524k | 5.43 | 1.29 | 4.2× | 10.4 | 2.5 |
| jelly-biotsavart | 5 | 131k | 6.53 | 2.03 | 3.2× | 49.8 | 15.5 |
| jelly-biotsavart | 6 | 1.0M | 14.35 | 4.99 | 2.9× | 13.7 | 4.8 |
| jelly-biotsavart | 7 | 8.4M | 68.29 | 24.38 | 2.8× | 8.1 | 2.9 |

- **Speedup:** 2.8–5.1× from 2M cells up, highest for the periodic TGV and lowest for jelly.
- **Small grids:** below 1M cells both codes are limited by per-step overhead, so the cost per
  cell rises.
- **cylinder-biotsavart:** TurboLily has no time because `TurboLily.BiotSimulation` does not
  support periodic directions.
- **Validation:** the test suite passes against these releases.
