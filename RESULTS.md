# Benchmarks

The paths in parentheses, such as `bench/suite/`, name the benchmark scripts. These are kept
outside this repository.

## WaterLily-Benchmarks cases on an A10, October 2026 (`bench/suite/`)

These are the eight cases of [WaterLily-Benchmarks](https://github.com/WaterLily-jl/WaterLily-Benchmarks)
(`src/cases.jl` at f6034df), on the GPU only, from the smallest default size up to the largest
power of two that fits on the card. The references are WaterLily master eb1ace3 (1.9.1) and
BiotSavartBCs main c7b6ffd (1.0.0). The machine has 4× NVIDIA A10
(22.5 GB, 600 GB/s). Float32, Julia 1.13.1, CUDA.jl 6.4.

The method is the suite's. Each case runs in its own process and starts from a developed-flow
checkpoint. It warms up for at least 50 steps and 2 s, then does 5 runs of 25 steps, each run
reset to the checkpoint. A step is timed with a device synchronisation, and the time is the
minimum over the runs of the mean step. Both codes load the same checkpoint.

The suite has checkpoints up to tgv 8, tgv-periodic 7, sphere 5, sphere-biotsavart 4,
cylinder 5, donut 6 and jelly 7. `bench/suite/develop.jl` makes the larger ones. It upsamples
the next smaller checkpoint 2× (trilinear on the staggered grid) and develops it with
TurboLily for 5 convective times; the TGVs get 1 and jelly gets 2 swimming cycles. The cases
from 4M cells up ran four at a time, one per GPU. The smaller cases ran again one at a time
on an idle machine, and those runs are the ones shown. The two passes agree within ±5 %
(`bench/suite/results*`).

| case | log2p | cells | reference [ms] | TurboLily [ms] | speedup | ref cost [ns/cell] | Turbo cost [ns/cell] |
|---|---:|---:|---:|---:|---:|---:|---:|
| tgv | 6 | 262k | 2.99 | 0.81 | 3.7× | 11.4 | 3.1 |
| tgv | 7 | 2.1M | 11.61 | 3.13 | 3.7× | 5.5 | 1.5 |
| tgv | 8 | 16.8M | 81.82 | 21.04 | 3.9× | 4.9 | 1.3 |
| tgv | 9 | 134.2M | 670.31 | 170.97 | 3.9× | 5.0 | 1.3 |
| tgv-periodic | 6 | 262k | 4.30 | 0.81 | 5.3× | 16.4 | 3.1 |
| tgv-periodic | 7 | 2.1M | 13.09 | 2.87 | 4.6× | 6.2 | 1.4 |
| tgv-periodic | 8 | 16.8M | 83.77 | 19.93 | 4.2× | 5.0 | 1.2 |
| tgv-periodic | 9 | 134.2M | 699.19 | 165.03 | 4.2× | 5.2 | 1.2 |
| sphere | 3 | 295k | 3.27 | 0.97 | 3.4× | 11.1 | 3.3 |
| sphere | 4 | 2.4M | 12.49 | 3.51 | 3.6× | 5.3 | 1.5 |
| sphere | 5 | 18.9M | 90.35 | 24.87 | 3.6× | 4.8 | 1.3 |
| sphere | 6 | 151.0M | 772.28 | 187.80 | 4.1× | 5.1 | 1.2 |
| sphere-biotsavart | 3 | 295k | 5.56 | 1.81 | 3.1× | 18.8 | 6.1 |
| sphere-biotsavart | 4 | 2.4M | 20.95 | 5.05 | 4.1× | 8.9 | 2.1 |
| sphere-biotsavart | 5 | 18.9M | 124.20 | 29.89 | 4.2× | 6.6 | 1.6 |
| sphere-biotsavart | 6 | 151.0M | 1020.85 | 230.64 | 4.4× | 6.8 | 1.5 |
| cylinder | 4 | 442k | 6.86 | 1.71 | 4.0× | 15.5 | 3.9 |
| cylinder | 5 | 3.5M | 24.30 | 7.73 | 3.1× | 6.9 | 2.2 |
| cylinder | 6 | 28.3M | 161.34 | 45.80 | 3.5× | 5.7 | 1.6 |
| cylinder-biotsavart | 4 | 442k | 13.81 | — | — | 31.2 | — |
| cylinder-biotsavart | 5 | 3.5M | 32.44 | — | — | 9.2 | — |
| donut | 5 | 66k | 3.44 | 0.92 | 3.8× | 52.5 | 14.0 |
| donut | 6 | 524k | 5.08 | 1.64 | 3.1× | 9.7 | 3.1 |
| donut | 7 | 4.2M | 21.80 | 6.05 | 3.6× | 5.2 | 1.4 |
| donut | 8 | 33.6M | 158.88 | 44.20 | 3.6× | 4.7 | 1.3 |
| jelly-biotsavart | 5 | 131k | 7.04 | 2.81 | 2.5× | 53.7 | 21.5 |
| jelly-biotsavart | 6 | 1.0M | 17.09 | 6.12 | 2.8× | 16.3 | 5.8 |
| jelly-biotsavart | 7 | 8.4M | 86.11 | 27.64 | 3.1× | 10.3 | 3.3 |
| jelly-biotsavart | 8 | 67.1M | 546.61 | 171.26 | 3.2× | 8.1 | 2.6 |

- **Largest size:** the last row of each case is the largest power of two that fits in 22.5 GB:
  134M cells (TGV 512³) and 151M cells (sphere 1024×384×384). Both codes need about the same
  memory; TurboLily's largest is sphere-biotsavart p=6 at 21.4 GB against 19.8 GB. The next
  power of two is 8× larger.
- **Speedup:** from 16M cells up it is 3.5–4.4×, except jelly-biotsavart (3.1–3.2×). Both
  codes are then bandwidth-bound. TurboLily costs 1.2–1.6 ns per cell and step,
  WaterLily 4.7–5.7 ns and BiotSavartBCs 6.6–6.8 ns, all without jelly. Jelly costs more in
  both codes (2.6 against 8.1 ns at p=8). It needs about 3 V-cycles per step instead of 2, and
  its two symmetry planes add image targets to every Biot-Savart evaluation; those are the
  likely reasons, not profiled.
- **Small grids:** below 1M cells both codes are limited by per-step overhead, so the cost
  per cell rises, and the speedup ranges from 2.5× (jelly) to 5.3× (tgv-periodic).
- **cylinder-biotsavart:** TurboLily has no time because `TurboLily.BiotSimulation` does
  not support periodic directions yet. Its p=6 checkpoint would need BiotSavartBCs to develop
  it, so that size was not run.
- **Pressure iterations:** these are identical except in `cylinder`. Its periodic z direction
  converges in fewer V-cycles in TurboLily (2.7 against 3.7 per step at p=4), as in the
  periodic tests.
- **Checkpoint loading fix:** this benchmark found a bug in TurboLily's `measure!`. After a
  jump in time larger than the measure band, it left stale μ₀ where the body had been. The
  jump happens when a moving body is reset to a checkpoint. The run after the jump needed many
  more V-cycles (40 instead of 2.7 per step for cylinder p=4). The minimum over the runs mostly
  hid this. This is fixed, with a test, and all moving-body cases were rerun after the
  fix.

## Against WaterLily master + BiotSavartBCs main, October 2026, RTX 2080 Ti (`bench/`)

References: WaterLily master eb1ace3 (1.9.1; contains #346, which supersedes #337) and
BiotSavartBCs main c7b6ffd (1.0.0; contains #13's stopping criterion and the `fill_ω!` part
of #14, but not #14's FMM `interaction` rewrite). One RTX 2080 Ti, Float32, Julia 1.13.1,
CUDA.jl 6.4, ms per step. The pressure-iteration counts are identical in both codes. Logs are
in `bench/oct8/`. The sweeps and the standard cases ran at the same time on separate GPUs;
the sphere 256×128×128 rows are from a rerun on an idle machine (two runs, mean).

| case | grid | reference | TurboLily | speedup |
|---|---|---|---|---|
| WaterLily, sphere | 256×128×128 | 19.21 | 6.07 | 3.2× |
| WaterLily, moving sphere (`remeasure=true`) | 256×128×128 | 23.60 | 8.38 | 2.8× |
| WaterLily, 2D circle | 1024×512 | 3.74 | 1.36 | 2.8× |
| WaterLily, TGV | 256³ | 76.67 | 20.29 | 3.8× |
| Biot-Savart, sphere | 256×128×128 | 31.46 | 8.55 | 3.7× |
| Biot-Savart, moving sphere | 256×128×128 | 45.25 | 11.04 | 4.1× |
| Biot-Savart, 2D circle | 1024×512 | 5.03 | 2.23 | 2.3× |

Size sweep (`bench_stack/sweep.jl` run in `bench/`), sphere Re=3700, ms per step (peak GPU
memory):

| WaterLily grid | cells | WL | TurboLily | speedup |
|---|---|---|---|---|
| 64×32×32 | 0.07M | 3.30 | 0.91 | 3.6× |
| 128×64×64 | 0.52M | 5.32 | 1.29 | 4.1× |
| 192×96×96 | 1.8M | 8.97 | 2.71 | 3.3× |
| 256×128×128 | 4.2M | 20.68 | 5.73 | 3.6× |
| 384×192×192 | 14.2M | 63.67 | 17.45 | 3.6× |
| 512×256×256 | 33.6M | 148.6 | 41.2 | 3.6× |
| 576×288×288 | 47.8M | 223.1 | 58.2 | 3.8× |
| 640×320×320 | 65.5M (10.2 GB) | 306.9 | 79.4 | 3.9× |

| Biot-Savart grid | cells | BSBC | TurboLily | speedup |
|---|---|---|---|---|
| 80×32×32 | 0.08M | 5.10 | 1.75 | 2.9× |
| 120×48×48 | 0.28M | 5.83 | 1.69 | 3.4× |
| 160×64×64 | 0.66M | 8.67 | 2.45 | 3.5× |
| 240×96×96 | 2.2M | 19.88 | 4.71 | 4.2× |
| 320×128×128 | 5.2M | 36.91 | 9.57 | 3.9× |
| 400×160×160 | 10.2M | 66.78 | 17.32 | 3.9× |
| 480×192×192 | 17.7M | 109.4 | 29.8 | 3.7× |
| 560×224×224 | 28.1M | 186.7 | 48.9 | 3.8× |
| 640×256×256 | 41.9M (7.2 GB) | 281.1 | 75.4 | 3.7× |

Compared with the earlier runs below, WaterLily master is within about ±10 % of #337, and
BiotSavartBCs main matches the 1.0 release. It is 7-50 % slower than the #14 PR branch, most
on the moving body, because #14's FMM rewrite was not merged. TurboLily is unchanged within
run-to-run noise (about ±10 %). TurboLily needs the same
memory as the reference (+2-6 %).

Long runs, sphere Re=3700, 192×96×96, t = 0…20:

| | wall | steps | ms/step | mean C_d (t>10) |
|---|---|---|---|---|
| BiotSavartBCs main | 25.5 s | 1165 | 21.9 | 0.3895 |
| TurboLily.BiotSimulation | 5.4 s (4.7×) | 1166 | 4.61 | 0.3897 |
| WaterLily master | 13.7 s | 852 | 16.1 | 0.4140 |
| TurboLily.Simulation | 2.7 s (5.1×) | 852 | 3.12 | 0.4141 |

## Where the time goes (TurboLily, sphere 256×128×128)

- **Plain step (≈6 ms):** multigrid ≈ 55 % (one V-cycle with smoothing ≈ 1.4 ms), momentum
  predictor + corrector ≈ 1.1 ms, projection source and correction ≈ 1 ms, the rest is BCs and
  the CFL condition. Nearly every large kernel runs at 80–90 % of the 616 GB/s memory bandwidth.
- **Biot step (≈9 ms):** on top of the plain step, there are 2 full FMM evaluations
  (≈ 0.45 ms each), 3 incremental FMM updates (≈ 0.1 ms each), vorticity (≈ 0.3 ms per
  projection), and the boundary updates.

## Earlier runs

The sections below predate WaterLily 1.9.1 and the BiotSavartBCs `symmetry` and residual
changes. They are kept for the comparison against the PR branches.

### Against BiotSavartBCs 1.0 (`bench_release/`)

BiotSavartBCs 1.0 is `main` (ce98bf2): #13's stopping criterion and the `fill_ω!` part of #14,
but not #14's FMM `interaction` rewrite. WaterLily is #337 as in the stack below, because
WaterLily master did not run on GPUs at that time (fixed in 1.9.1, see the last section). One RTX 2080 Ti, ms per step, sphere Re=3700.

| Biot-Savart grid | cells | BSBC#14 PR | BSBC 1.0 | TurboLily | speedup vs #14 PR | vs 1.0 |
|---|---|---|---|---|---|---|
| 80×32×32 | 0.08M | 4.68 | 5.93 | 1.64 | 2.6× | 3.6× |
| 120×48×48 | 0.28M | 4.42 | 5.88 | 1.65 | 2.6× | 3.6× |
| 160×64×64 | 0.66M | 6.67 | 10.26 | 2.51 | 2.6× | 4.1× |
| 240×96×96 | 2.2M | 15.13 | 18.11 | 4.79 | 3.1× | 3.8× |
| 320×128×128 | 5.2M | 31.80 | 36.34 | 9.67 | 3.2× | 3.8× |
| 400×160×160 | 10.2M | 62.56 | 70.03 | 17.52 | 3.1× | 4.0× |
| 480×192×192 | 17.7M | 101.98 | 108.65 | 30.18 | 3.0× | 3.6× |
| 560×224×224 | 28.1M | 173.4 | 179.9 | 48.4 | 3.6× | 3.7× |
| 640×256×256 | 41.9M | 258.2 | 273.3 | 72.4 | 3.5× | 3.8× |

The #14-PR column and its speedup come from the `bench_stack` run; the 1.0 column and its
speedup from the `bench_release` run (TurboLily is within ±10 % between the two runs). The
three `bench_release` runs (sweep, standard case, long run) ran at the same time on three
GPUs; the smallest grids are partly host-bound and may be affected by the shared CPU.

Standard case, sphere 256×128×128: BSBC 1.0 30.52 ms/step, TurboLily 8.88 ms/step (3.4×).

Long run, sphere Re=3700, 192×96×96, t = 0…20: BSBC 1.0 26.1 s, TurboLily 5.3 s (4.9×);
mean C_d (t>10) 0.3893 and 0.3898.

### Against WaterLily#337 + BiotSavartBCs#13+#14 (`bench_stack/`)

Reference: WaterLily PR #337 (3f1de92, on 55b2f5d) and BiotSavartBCs PR #14 (5aeb5c5, which
contains #13). TurboLily follows #13's stopping criterion (`max|r| < 2e-3`, `Σ|r|/N < 2e-4`)
and has the same pressure-iteration counts as this stack. One RTX 2080 Ti, ms per step.

| case | grid | stack | TurboLily | speedup |
|---|---|---|---|---|
| WaterLily, sphere | 256×128×128 | 18.86 | 5.97 | 3.2× |
| WaterLily, moving sphere (`remeasure=true`) | 256×128×128 | 22.78 | 8.39 | 2.7× |
| WaterLily, 2D circle | 1024×512 | 3.32 | 1.26 | 2.6× |
| WaterLily, sphere | 512×256×256 | 143.0 | 41.4 | 3.5× |
| WaterLily, TGV | 256³ | 73.97 | 20.09 | 3.7× |
| Biot-Savart, sphere | 256×128×128 | 26.65 | 8.66 | 3.1× |
| Biot-Savart, moving sphere | 256×128×128 | 29.82 | 10.78 | 2.8× |
| Biot-Savart, 2D circle | 1024×512 | 4.63 | 2.25 | 2.1× |
| Biot-Savart, sphere | 160×64×64 | 8.18 | 2.56 | 3.2× |
| Biot-Savart, sphere | 320×128×128 | 32.34 | 9.59 | 3.4× |
| Biot-Savart, sphere | 480×192×192 | 98.86 | 30.80 | 3.2× |

Size sweep (`bench_stack/sweep.jl`), sphere Re=3700, ms per step (peak GPU memory):

| WaterLily grid | cells | stack | TurboLily | speedup |
|---|---|---|---|---|
| 64×32×32 | 0.07M | 4.38 | 0.92 | 4.8× |
| 128×64×64 | 0.52M | 4.64 | 1.32 | 3.5× |
| 192×96×96 | 1.8M | 8.76 | 2.69 | 3.3× |
| 256×128×128 | 4.2M | 18.57 | 5.68 | 3.3× |
| 384×192×192 | 14.2M | 58.32 | 18.66 | 3.1× |
| 512×256×256 | 33.6M | 142.9 | 41.0 | 3.5× |
| 576×288×288 | 47.8M | 219.2 | 58.7 | 3.7× |
| 640×320×320 | 65.5M (8.7 GB) | 299.3 | 79.7 | 3.8× |

| Biot-Savart grid | cells | stack | TurboLily | speedup |
|---|---|---|---|---|
| 80×32×32 | 0.08M | 4.68 | 1.81 | 2.6× |
| 120×48×48 | 0.28M | 4.42 | 1.71 | 2.6× |
| 160×64×64 | 0.66M | 6.67 | 2.53 | 2.6× |
| 240×96×96 | 2.2M | 15.13 | 4.81 | 3.1× |
| 320×128×128 | 5.2M | 31.80 | 9.80 | 3.2× |
| 400×160×160 | 10.2M | 62.56 | 20.28 | 3.1× |
| 480×192×192 | 17.7M | 101.98 | 34.52 | 3.0× |
| 560×224×224 | 28.1M | 173.4 | 48.6 | 3.6× |
| 640×256×256 | 41.9M (6.8 GB) | 258.2 | 74.3 | 3.5× |

Below about 1M cells both codes are limited by launch and synchronisation overhead, not by
the GPU. The reference stays at a floor of about 4.4 ms/step, TurboLily at about 0.9 ms
(plain) and 1.7 ms (Biot-Savart). Above about 2M cells both run near memory bandwidth, so the
ratio levels off at 3.1-3.8×. TurboLily needs the same memory as the reference (+2-6 %).

Long runs, sphere Re=3700, 192×96×96, t = 0…20:

| | wall | steps | mean C_d (t>10) |
|---|---|---|---|
| BiotSavartBCs#14 | 21.2 s | 1165 | 0.3890 |
| TurboLily.BiotSimulation | 5.2 s (4.1×) | 1165 | 0.3892 |
| WaterLily#337 | 13.1 s | 852 | 0.4141 |
| TurboLily.Simulation | 2.5 s (5.2×) | 852 | 0.4141 |

### Against WaterLily master + BiotSavartBCs main, September 2026 (`bench/`)

These used BiotSavartBCs' old stopping criterion (`Σr² < 1e-4`), so the Biot-Savart iteration
counts differ from the stack above.

The hardware is one NVIDIA RTX 2080 Ti (11 GB, sm_75) with Julia 1.13.1, CUDA.jl 6.4 and Float32.
The references are WaterLily `main` (4da089f, with the GPU fix for `mom_project!`, see below)
and BiotSavartBCs `master` (6ffe945), run on the same GPU. Timings are wall-clock ms per time
step after warm-up. GPU boost clocks vary by about ±10 % between runs.

#### Sphere, Re=3700, D = N/2 (`bench/bench.jl`, `bench/bench2.jl`)

| case | grid | reference | TurboLily | speedup |
|---|---|---|---|---|
| WaterLily, static body | 256×128×128 | 19.28 | 6.07 | 3.2× |
| WaterLily, moving body (`remeasure=true`) | 256×128×128 | 22.95 | 8.15 | 2.8× |
| WaterLily, 2D circle | 1024×512 | 3.78 | 1.27–1.61 | 2.3–3.0× |
| Biot-Savart, static body | 256×128×128 | 53.90 | 10.84 | 5.0× |
| Biot-Savart, moving body (`remeasure=true`) | 256×128×128 | 66.82 | 14.42 | 4.6× |
| Biot-Savart, 2D circle | 1024×512 | 4.83 | 2.22 | 2.2× |

#### Size scaling, developed flow (`bench/bigbench.jl`)

| case | grid | cells | reference | TurboLily | speedup |
|---|---|---|---|---|---|
| Biot-Savart | 160×64×64 | 0.7M | 11.38 | 2.60 | 4.4× |
| Biot-Savart | 320×128×128 | 5.2M | 60.67 | 11.33 | 5.4× |
| Biot-Savart | 480×192×192 | 17.7M | 179.00 | 34.14 | 5.2× |
| WaterLily | 256×128×128 | 4.2M | 19.81 | 5.79 | 3.4× |
| WaterLily | 512×256×256 | 33.6M | 147.57 | 40.99 | 3.6× |

The pressure-iteration counts are the same in both codes.

Taylor-Green vortex, Re=1600, 256³, periodic in all directions: WaterLily 77.6 ms/step,
TurboLily 20.4 ms/step (3.8×).

#### Long runs (`bench/longrun.jl`): sphere, Re=3700, 192×96×96, t = 0…20

| | wall | steps | ms/step | mean C_d (t>10) | std C_d |
|---|---|---|---|---|---|
| BiotSavartBCs | 35.9 s | 1160 | 30.9 | 0.3867 | 0.0169 |
| TurboLily.BiotSimulation | 6.7 s | 1165 | 5.73 | 0.3865 | 0.0168 |
| WaterLily | 13.0 s | 852 | 15.2 | 0.4140 | 0.0072 |
| TurboLily.Simulation | 2.7 s | 852 | 3.21 | 0.4142 | 0.0070 |

### WaterLily master did not run on GPUs (September 2026)

WaterLily `master` at 4da089f failed in the first `sim_step!` of every GPU simulation:
`mom_project!` passed the `MultiLevelPoisson` itself into a `@loop` kernel (`δv(b,I)`), which
is not a bitstype. The runs in that section used a local 3-line fix. WaterLily 1.9.1 fixes it
upstream with `diagonal(p)` and `δv(D,I)`.
