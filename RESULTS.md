# Benchmarks

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

## WaterLily GPU fix

WaterLily `main` at 4da089f (PR #328, "deforming body source") no longer runs on GPUs. The
call `δv(b,I)` inside `@inside` passes the `MultiLevelPoisson` struct into a kernel. The local
checkout in `dev/WL` has a 3-line fix that reads the diagonal array instead. TurboLily does not
depend on this fix.
