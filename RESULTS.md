# Benchmarks

## Against WaterLily#337 + BiotSavartBCs#13+#14 (`bench_stack/`)

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

## Against WaterLily master + BiotSavartBCs main (earlier runs, `bench/`)

These used BiotSavartBCs' old stopping criterion (`Σr² < 1e-4`), so the Biot-Savart iteration
counts differ from the stack above.

The hardware is one NVIDIA RTX 2080 Ti (11 GB, sm_75) with Julia 1.13.1, CUDA.jl 6.4 and Float32.
The references are WaterLily `main` (4da089f, with the GPU fix for `mom_project!`, see below)
and BiotSavartBCs `master` (6ffe945), run on the same GPU. Timings are wall-clock ms per time
step after warm-up. GPU boost clocks vary by about ±10 % between runs.

## Sphere, Re=3700, D = N/2 (`bench/bench.jl`, `bench/bench2.jl`)

| case | grid | reference | TurboLily | speedup |
|---|---|---|---|---|
| WaterLily, static body | 256×128×128 | 19.28 | 6.07 | 3.2× |
| WaterLily, moving body (`remeasure=true`) | 256×128×128 | 22.95 | 8.15 | 2.8× |
| WaterLily, 2D circle | 1024×512 | 3.78 | 1.27–1.61 | 2.3–3.0× |
| Biot-Savart, static body | 256×128×128 | 53.90 | 10.84 | 5.0× |
| Biot-Savart, moving body (`remeasure=true`) | 256×128×128 | 66.82 | 14.42 | 4.6× |
| Biot-Savart, 2D circle | 1024×512 | 4.83 | 2.22 | 2.2× |

## Size scaling, developed flow (`bench/bigbench.jl`)

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

## Long runs (`bench/longrun.jl`): sphere, Re=3700, 192×96×96, t = 0…20

| | wall | steps | ms/step | mean C_d (t>10) | std C_d |
|---|---|---|---|---|---|
| BiotSavartBCs | 35.9 s | 1160 | 30.9 | 0.3867 | 0.0169 |
| TurboLily.BiotSimulation | 6.7 s | 1165 | 5.73 | 0.3865 | 0.0168 |
| WaterLily | 13.0 s | 852 | 15.2 | 0.4140 | 0.0072 |
| TurboLily.Simulation | 2.7 s | 852 | 3.21 | 0.4142 | 0.0070 |

## Where the time goes (TurboLily, sphere 256×128×128)

- **Plain step (≈6 ms):** multigrid ≈ 55 % (one V-cycle with smoothing ≈ 1.4 ms), momentum
  predictor + corrector ≈ 1.1 ms, projection source and correction ≈ 1 ms, the rest is BCs and
  the CFL condition. Nearly every large kernel runs at 80–90 % of the 616 GB/s memory bandwidth.
- **Biot step (≈11 ms):** on top of the plain step, there are 2 full FMM evaluations
  (≈ 0.45 ms each), 3 incremental FMM updates (≈ 0.1 ms each), vorticity (≈ 0.3 ms per
  projection), and the boundary updates.

## WaterLily master does not run on GPUs

WaterLily `master` at 4da089f (merge of PR #328, "deforming body source", 2026-09-30) fails
in the first `sim_step!` of every GPU simulation, with or without a body:

```
KernelError: passing non-bitstype argument
Argument 5 to your kernel function is of type MultiLevelPoisson{Float32, CuArray{…}, …}, which is not a bitstype
```

`mom_project!` now does `@inside b.z[I] = div(I,a.u)-δv(b,I)*div(I,a.V)`. `@loop` passes
every symbol in the expression to the kernel, including `b` (the `MultiLevelPoisson`). That
struct holds `CuArray`s and a `Vector`, and it has no Adapt rule, so CUDA cannot convert it.
On the CPU the KA backend accepts any argument, so the CPU tests pass. The example scripts
default to `mem=Array`, so they also run.

The fix: take the array outside the loop, `Dp = levels[1].D`, and use
`(1+Dp[I]/2D)` in the kernel. The checkout in `dev/WL` has this 3-line fix; it is the
WaterLily used in `bench/`. WaterLily#337 is based on 55b2f5d, before #328, so it is not
affected. BiotSavartBCs and TurboLily replace `mom_project!` and never hit this code.
