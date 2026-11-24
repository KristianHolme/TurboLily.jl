# Benchmarks

## Current benchmarks (`9490874`)

After the four-GPU suite (`fe0e2c6`) below, commits `3d22ba7` and `9490874` (split-precision
V(1_RBGS, 2_RBGS) multigrid, dirty-box `measure!`/`update!`, and Biot-Savart fusions) added a
further 14–17 % speedup across the suite.

Same method as below: WaterLily-Benchmarks (`src/cases.jl` at f6034df), Reference WaterLily 1.11.1
or BiotSavartBCs 1.1.0 for `-biotsavart`, Float32, Julia 1.13, CUDA.jl 6.5.1, KernelAbstractions
0.10.8 (2026-11-24).

Over all 17 case sizes:

| GPU | speedup range | geometric mean |
|---|---:|---:|
| A100 80 GB PCIe | 2.5–7.5× | 3.83× |
| RTX 3090 | 3.1–7.4× | 4.30× |

### A100 80 GB PCIe (`9490874`)

| case | log2p | cells | reference [ms] | TurboLily [ms] | speedup | ref cost [ns/cell] | Turbo cost [ns/cell] |
|---|---:|---:|---:|---:|---:|---:|---:|
| tgv | 6 | 262k | 2.12 | 0.47 | 4.5× | 8.1 | 1.8 |
| tgv | 7 | 2.1M | 5.15 | 1.43 | 3.6× | 2.5 | 0.7 |
| tgv | 8 | 16.8M | 28.82 | 8.12 | 3.5× | 1.7 | 0.5 |
| tgv-periodic | 6 | 262k | 3.58 | 0.48 | 7.5× | 13.7 | 1.8 |
| tgv-periodic | 7 | 2.1M | 6.70 | 1.35 | 5.0× | 3.2 | 0.6 |
| sphere | 3 | 295k | 2.36 | 0.57 | 4.1× | 8.0 | 1.9 |
| sphere | 4 | 2.4M | 5.57 | 1.61 | 3.5× | 2.4 | 0.7 |
| sphere | 5 | 18.9M | 32.02 | 8.93 | 3.6× | 1.7 | 0.5 |
| sphere-biotsavart | 3 | 295k | 4.14 | 1.33 | 3.1× | 14.0 | 4.5 |
| sphere-biotsavart | 4 | 2.4M | 9.10 | 2.81 | 3.2× | 3.9 | 1.2 |
| cylinder | 4 | 442k | 4.85 | 1.02 | 4.8× | 11.0 | 2.3 |
| cylinder | 5 | 3.5M | 10.26 | 2.99 | 3.4× | 2.9 | 0.8 |
| donut | 5 | 66k | 2.50 | 0.48 | 5.2× | 38.1 | 7.3 |
| donut | 6 | 524k | 3.04 | 0.73 | 4.1× | 5.8 | 1.4 |
| jelly-biotsavart | 5 | 131k | 6.01 | 2.03 | 3.0× | 45.9 | 15.5 |
| jelly-biotsavart | 6 | 1.0M | 8.95 | 3.52 | 2.5× | 8.5 | 3.4 |
| jelly-biotsavart | 7 | 8.4M | 30.24 | 10.69 | 2.8× | 3.6 | 1.3 |

### RTX 3090 (`9490874`)

| case | log2p | cells | reference [ms] | TurboLily [ms] | speedup | ref cost [ns/cell] | Turbo cost [ns/cell] |
|---|---:|---:|---:|---:|---:|---:|---:|
| tgv | 6 | 262k | 2.08 | 0.47 | 4.4× | 8.0 | 1.8 |
| tgv | 7 | 2.1M | 6.14 | 1.50 | 4.1× | 2.9 | 0.7 |
| tgv | 8 | 16.8M | 39.68 | 8.91 | 4.5× | 2.4 | 0.5 |
| tgv-periodic | 6 | 262k | 3.50 | 0.47 | 7.4× | 13.3 | 1.8 |
| tgv-periodic | 7 | 2.1M | 7.72 | 1.47 | 5.3× | 3.7 | 0.7 |
| sphere | 3 | 295k | 2.29 | 0.56 | 4.1× | 7.8 | 1.9 |
| sphere | 4 | 2.4M | 6.82 | 1.59 | 4.3× | 2.9 | 0.7 |
| sphere | 5 | 18.9M | 45.04 | 9.93 | 4.5× | 2.4 | 0.5 |
| sphere-biotsavart | 3 | 295k | 3.74 | 1.20 | 3.1× | 12.7 | 4.1 |
| sphere-biotsavart | 4 | 2.4M | 10.46 | 2.58 | 4.1× | 4.4 | 1.1 |
| cylinder | 4 | 442k | 4.97 | 0.96 | 5.2× | 11.2 | 2.2 |
| cylinder | 5 | 3.5M | 12.56 | 3.07 | 4.1× | 3.5 | 0.9 |
| donut | 5 | 66k | 2.48 | 0.44 | 5.6× | 37.9 | 6.7 |
| donut | 6 | 524k | 3.27 | 0.70 | 4.6× | 6.2 | 1.3 |
| jelly-biotsavart | 5 | 131k | 5.57 | 1.67 | 3.3× | 42.5 | 12.8 |
| jelly-biotsavart | 6 | 1.0M | 10.12 | 3.21 | 3.2× | 9.7 | 3.1 |
| jelly-biotsavart | 7 | 8.4M | 38.79 | 12.06 | 3.2× | 4.6 | 1.4 |

## Four-GPU suite (`fe0e2c6`)

GPU cases of [WaterLily-Benchmarks](https://github.com/WaterLily-jl/WaterLily-Benchmarks)
(`src/cases.jl` at f6034df) at every size with a checkpoint, with the suite's method: each case in
its own process from a developed-flow checkpoint, a warm-up of at least 50 steps and 2 s, then 5
runs of 25 steps, each reset to the checkpoint. The time is the minimum over the runs of the mean
step. Reference and TurboLily ran back to back on the same GPU.

Reference: WaterLily 1.11.1, or BiotSavartBCs 1.1.0 for `-biotsavart`. TurboLily fe0e2c6, Float32,
Julia 1.13, CUDA.jl 6.5.1, KernelAbstractions 0.10.8 (2026-11-19).
`cylinder-biotsavart` has a periodic direction, which `TurboLily.BiotSimulation` does not support.

Over all 17 case sizes:

| GPU | speedup range | geometric mean |
|---|---:|---:|
| A100 80 GB | 2.4–5.7× | 3.27× |
| GH200 | 2.5–6.0× | 3.35× |
| RTX 3090 | 2.6–7.1× | 3.77× |
| P100 16 GB | 2.4–4.5× | 3.23× |

### A100 80 GB

| case | log2p | cells | reference [ms] | TurboLily [ms] | speedup | ref cost [ns/cell] | Turbo cost [ns/cell] |
|---|---:|---:|---:|---:|---:|---:|---:|
| tgv | 6 | 262k | 2.12 | 0.54 | 3.9× | 8.1 | 2.1 |
| tgv | 7 | 2.1M | 5.16 | 1.74 | 3.0× | 2.5 | 0.8 |
| tgv | 8 | 16.8M | 28.46 | 9.52 | 3.0× | 1.7 | 0.6 |
| tgv-periodic | 6 | 262k | 3.50 | 0.62 | 5.7× | 13.4 | 2.3 |
| tgv-periodic | 7 | 2.1M | 6.62 | 1.51 | 4.4× | 3.2 | 0.7 |
| sphere | 3 | 295k | 2.35 | 0.64 | 3.7× | 8.0 | 2.2 |
| sphere | 4 | 2.4M | 5.58 | 1.91 | 2.9× | 2.4 | 0.8 |
| sphere | 5 | 18.9M | 31.57 | 10.66 | 3.0× | 1.7 | 0.6 |
| sphere-biotsavart | 3 | 295k | 4.16 | 1.44 | 2.9× | 14.1 | 4.9 |
| sphere-biotsavart | 4 | 2.4M | 8.96 | 3.15 | 2.8× | 3.8 | 1.3 |
| cylinder | 4 | 442k | 4.69 | 1.14 | 4.1× | 10.6 | 2.6 |
| cylinder | 5 | 3.5M | 10.20 | 3.83 | 2.7× | 2.9 | 1.1 |
| cylinder-biotsavart | 4 | 442k | 11.84 | — | — | 26.8 | — |
| cylinder-biotsavart | 5 | 3.5M | 15.93 | — | — | 4.5 | — |
| donut | 5 | 66k | 2.42 | 0.57 | 4.2× | 37.0 | 8.7 |
| donut | 6 | 524k | 3.04 | 0.97 | 3.1× | 5.8 | 1.9 |
| jelly-biotsavart | 5 | 131k | 6.03 | 2.11 | 2.9× | 46.0 | 16.1 |
| jelly-biotsavart | 6 | 1.0M | 8.97 | 3.74 | 2.4× | 8.6 | 3.6 |
| jelly-biotsavart | 7 | 8.4M | 30.12 | 12.51 | 2.4× | 3.6 | 1.5 |

### GH200

| case | log2p | cells | reference [ms] | TurboLily [ms] | speedup | ref cost [ns/cell] | Turbo cost [ns/cell] |
|---|---:|---:|---:|---:|---:|---:|---:|
| tgv | 6 | 262k | 1.42 | 0.39 | 3.7× | 5.4 | 1.5 |
| tgv | 7 | 2.1M | 3.56 | 1.08 | 3.3× | 1.7 | 0.5 |
| tgv | 8 | 16.8M | 18.56 | 5.67 | 3.3× | 1.1 | 0.3 |
| tgv-periodic | 6 | 262k | 2.27 | 0.38 | 6.0× | 8.6 | 1.4 |
| tgv-periodic | 7 | 2.1M | 4.49 | 0.92 | 4.9× | 2.1 | 0.4 |
| sphere | 3 | 295k | 1.58 | 0.46 | 3.5× | 5.4 | 1.5 |
| sphere | 4 | 2.4M | 3.76 | 1.20 | 3.1× | 1.6 | 0.5 |
| sphere | 5 | 18.9M | 20.39 | 6.39 | 3.2× | 1.1 | 0.3 |
| sphere-biotsavart | 3 | 295k | 2.95 | 1.06 | 2.8× | 10.0 | 3.6 |
| sphere-biotsavart | 4 | 2.4M | 6.03 | 2.06 | 2.9× | 2.6 | 0.9 |
| cylinder | 4 | 442k | 3.28 | 0.82 | 4.0× | 7.4 | 1.9 |
| cylinder | 5 | 3.5M | 7.14 | 2.42 | 2.9× | 2.0 | 0.7 |
| cylinder-biotsavart | 4 | 442k | 8.76 | — | — | 19.8 | — |
| cylinder-biotsavart | 5 | 3.5M | 10.98 | — | — | 3.1 | — |
| donut | 5 | 66k | 1.74 | 0.46 | 3.8× | 26.6 | 7.1 |
| donut | 6 | 524k | 2.14 | 0.71 | 3.0× | 4.1 | 1.3 |
| jelly-biotsavart | 5 | 131k | 4.40 | 1.52 | 2.9× | 33.6 | 11.6 |
| jelly-biotsavart | 6 | 1.0M | 6.54 | 2.48 | 2.6× | 6.2 | 2.4 |
| jelly-biotsavart | 7 | 8.4M | 19.17 | 7.66 | 2.5× | 2.3 | 0.9 |

### RTX 3090

| case | log2p | cells | reference [ms] | TurboLily [ms] | speedup | ref cost [ns/cell] | Turbo cost [ns/cell] |
|---|---:|---:|---:|---:|---:|---:|---:|
| tgv | 6 | 262k | 2.19 | 0.54 | 4.0× | 8.3 | 2.1 |
| tgv | 7 | 2.1M | 6.20 | 1.69 | 3.7× | 3.0 | 0.8 |
| tgv | 8 | 16.8M | 39.92 | 10.66 | 3.7× | 2.4 | 0.6 |
| tgv-periodic | 6 | 262k | 3.70 | 0.52 | 7.1× | 14.1 | 2.0 |
| tgv-periodic | 7 | 2.1M | 8.04 | 1.58 | 5.1× | 3.8 | 0.8 |
| sphere | 3 | 295k | 2.37 | 0.64 | 3.7× | 8.0 | 2.2 |
| sphere | 4 | 2.4M | 6.92 | 1.85 | 3.7× | 2.9 | 0.8 |
| sphere | 5 | 18.9M | 46.14 | 12.13 | 3.8× | 2.4 | 0.6 |
| sphere-biotsavart | 3 | 295k | 3.83 | 1.25 | 3.1× | 13.0 | 4.2 |
| sphere-biotsavart | 4 | 2.4M | 10.71 | 3.02 | 3.5× | 4.5 | 1.3 |
| cylinder | 4 | 442k | 4.88 | 1.12 | 4.4× | 11.0 | 2.5 |
| cylinder | 5 | 3.5M | 12.74 | 3.95 | 3.2× | 3.6 | 1.1 |
| cylinder-biotsavart | 4 | 442k | 10.68 | — | — | 24.1 | — |
| cylinder-biotsavart | 5 | 3.5M | 17.43 | — | — | 4.9 | — |
| donut | 5 | 66k | 2.62 | 0.56 | 4.7× | 39.9 | 8.6 |
| donut | 6 | 524k | 3.42 | 1.01 | 3.4× | 6.5 | 1.9 |
| jelly-biotsavart | 5 | 131k | 5.91 | 1.76 | 3.4× | 45.1 | 13.4 |
| jelly-biotsavart | 6 | 1.0M | 10.40 | 3.63 | 2.9× | 9.9 | 3.5 |
| jelly-biotsavart | 7 | 8.4M | 39.22 | 14.86 | 2.6× | 4.7 | 1.8 |

### P100 16 GB

| case | log2p | cells | reference [ms] | TurboLily [ms] | speedup | ref cost [ns/cell] | Turbo cost [ns/cell] |
|---|---:|---:|---:|---:|---:|---:|---:|
| tgv | 6 | 262k | 3.73 | 1.30 | 2.9× | 14.2 | 5.0 |
| tgv | 7 | 2.1M | 17.75 | 4.88 | 3.6× | 8.5 | 2.3 |
| tgv | 8 | 16.8M | 126.36 | 30.83 | 4.1× | 7.5 | 1.8 |
| tgv-periodic | 6 | 262k | 5.18 | 1.16 | 4.5× | 19.8 | 4.4 |
| tgv-periodic | 7 | 2.1M | 18.96 | 4.39 | 4.3× | 9.0 | 2.1 |
| sphere | 3 | 295k | 3.98 | 1.44 | 2.8× | 13.5 | 4.9 |
| sphere | 4 | 2.4M | 19.13 | 5.18 | 3.7× | 8.1 | 2.2 |
| sphere | 5 | 18.9M | 141.54 | 34.78 | 4.1× | 7.5 | 1.8 |
| sphere-biotsavart | 3 | 295k | 7.67 | 2.61 | 2.9× | 26.0 | 8.9 |
| sphere-biotsavart | 4 | 2.4M | 28.96 | 7.88 | 3.7× | 12.3 | 3.3 |
| cylinder | 4 | 442k | 7.93 | 2.36 | 3.4× | 17.9 | 5.3 |
| cylinder | 5 | 3.5M | 34.36 | 10.52 | 3.3× | 9.7 | 3.0 |
| cylinder-biotsavart | 4 | 442k | 22.18 | — | — | 50.1 | — |
| cylinder-biotsavart | 5 | 3.5M | 48.20 | — | — | 13.6 | — |
| donut | 5 | 66k | 2.71 | 1.11 | 2.4× | 41.3 | 16.9 |
| donut | 6 | 524k | 6.45 | 2.43 | 2.6× | 12.3 | 4.6 |
| jelly-biotsavart | 5 | 131k | 9.16 | 3.54 | 2.6× | 69.9 | 27.0 |
| jelly-biotsavart | 6 | 1.0M | 22.27 | 8.99 | 2.5× | 21.2 | 8.6 |
| jelly-biotsavart | 7 | 8.4M | 111.60 | 41.06 | 2.7× | 13.3 | 4.9 |

## Large grids (GH200)

Sphere at Re=3700 from an impulsive start, 40 warm-up steps, then the minimum over 5 runs of 20
steps; same versions as above. Memory is that of the TurboLily Biot-Savart run; the others use
up to 3 % less.

| grid | cells | WaterLily [ms] | TurboLily [ms] | speedup | BiotSavartBCs [ms] | TurboLily [ms] | speedup | memory [GiB] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512×192×192 | 18.9M | 20.1 | 6.3 | 3.2× | 25.2 | 8.4 | 3.0× | 2 |
| 768×288×288 | 63.7M | 63.5 | 19.7 | 3.2× | 75.2 | 23.4 | 3.2× | 8 |
| 1024×384×384 | 151.0M | 150.5 | 45.6 | 3.3× | 169.0 | 52.6 | 3.2× | 18 |
| 1152×432×432 | 215.0M | 212.7 | 64.3 | 3.3× | 293.4 | 96.0 | 3.1× | 26 |
| 1280×480×480 | 294.9M | 290.8 | 87.6 | 3.3× | 330.3 | 100.0 | 3.3× | 36 |
| 1536×576×576 | 509.6M | 500.0 | 152.4 | 3.3× | 557.2 | 170.0 | 3.3× | 62 |
| 2048×576×576 | 679.5M | 676.9 | 205.1 | 3.3× | 750.0 | 225.9 | 3.3× | 83 |

## Earlier runs

### A10, WaterLily 1.11.0

NVIDIA A10 (22.5 GB, 600 GB/s), same software, a quiet host (load average 2–3).

| case | log2p | cells | reference [ms] | TurboLily [ms] | speedup | ref cost [ns/cell] | Turbo cost [ns/cell] |
|---|---:|---:|---:|---:|---:|---:|---:|
| tgv | 6 | 262k | 2.26 | 0.59 | 3.9× | 8.6 | 2.2 |
| tgv | 7 | 2.1M | 9.90 | 2.58 | 3.8× | 4.7 | 1.2 |
| tgv | 8 | 16.8M | 67.36 | 17.28 | 3.9× | 4.0 | 1.0 |
| tgv-periodic | 6 | 262k | 3.67 | 0.53 | 6.9× | 14.0 | 2.0 |
| tgv-periodic | 7 | 2.1M | 10.85 | 2.39 | 4.5× | 5.2 | 1.1 |
| sphere | 3 | 295k | 2.53 | 0.69 | 3.7× | 8.6 | 2.3 |
| sphere | 4 | 2.4M | 10.80 | 2.85 | 3.8× | 4.6 | 1.2 |
| sphere | 5 | 18.9M | 77.12 | 19.89 | 3.9× | 4.1 | 1.1 |
| sphere-biotsavart | 3 | 295k | 4.46 | 1.42 | 3.1× | 15.1 | 4.8 |
| sphere-biotsavart | 4 | 2.4M | 15.59 | 4.40 | 3.5× | 6.6 | 1.9 |
| cylinder | 4 | 442k | 5.62 | 1.32 | 4.2× | 12.7 | 3.0 |
| cylinder | 5 | 3.5M | 20.56 | 6.32 | 3.3× | 5.8 | 1.8 |
| cylinder-biotsavart | 4 | 442k | 12.94 | — | — | 29.3 | — |
| cylinder-biotsavart | 5 | 3.5M | 26.72 | — | — | 7.6 | — |
| donut | 5 | 66k | 2.25 | 0.52 | 4.3× | 34.4 | 7.9 |
| donut | 6 | 524k | 4.05 | 1.27 | 3.2× | 7.7 | 2.4 |
| jelly-biotsavart | 5 | 131k | 5.45 | 2.03 | 2.7× | 41.6 | 15.5 |
| jelly-biotsavart | 6 | 1.0M | 13.43 | 5.09 | 2.6× | 12.8 | 4.9 |
| jelly-biotsavart | 7 | 8.4M | 65.90 | 24.68 | 2.7× | 7.9 | 2.9 |


### A10, WaterLily 1.10.0

The previous full run: WaterLily 1.10.0 and BiotSavartBCs 1.1.0 on an A10, Float32, Julia
1.13.2, CUDA.jl 6.4.2.

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
