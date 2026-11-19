# TurboLily

CUDA rewrite of the [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl) time step and of the
[BiotSavartBCs.jl](https://github.com/WaterLily-jl/BiotSavartBCs.jl) pressure solver, behind
WaterLily's API. The constructors take the arguments of `WaterLily.Simulation` and
`BiotSavartBCs.BiotSimulation` and return a `WaterLily.Simulation`.

```julia
using WaterLily, TurboLily, StaticArrays

body = AutoBody((x,t)->√sum(abs2, x .- SA[64f0,64f0,64f0]) - 32f0)
sim  = TurboLily.Simulation((256,128,128), (1,0,0), 64; ν=64/3700, body)        # WaterLily
bsim = TurboLily.BiotSimulation((256,128,128), (1,0,0), 64; ν=64/3700, body)    # Biot-Savart BCs
sim_step!(bsim, 10; remeasure=false)
```

```julia
using Pkg; Pkg.add(url="https://github.com/KristianHolme/TurboLily.jl")
```

Requires WaterLily 1.11 and BiotSavartBCs 1.1. Tested with Julia 1.13, CUDA.jl 6.5 and
KernelAbstractions 0.10 on RTX 2080 Ti and A10. Benchmarked on A100, GH200, RTX 3090 and P100.

## Limitations

- `T=Float32` and `mem=CuArray` only. One GPU per simulation.
- Grid size: kernels use 32-bit indices, so D·cells (with ghosts) ≤ 2³¹−1: about 715M cells in 3D.
  2048×576×576 (680M cells, 82 GiB) runs on a GH200.
- `flow.u`, `flow.u⁰`, `flow.f` are swapped, not copied: read `sim.flow.u` afresh after a step.
- `BiotSimulation`: no `fmm=false`, no periodic directions, at least two Biot-Savart levels.
- No automatic differentiation (`ForwardDiff.Dual` element types are rejected).
- The pressure solver has its own layout: `pois.levels[1].r`, `.iD` and `residual!` do not exist.
- Tangential ghost values of `flow.u` can differ from WaterLily's (neither code reads them).

Other packages (bodies that hold arrays must be built with `mem=CuArray`):

| package | status |
|---|---|
| [ParametricBodies.jl](https://github.com/WaterLily-jl/ParametricBodies.jl) `ParametricBody`, `HashedBody` | agrees with WaterLily to rounding (2D, static and moving) |
| [WaterLilyMeshBodies.jl](https://github.com/WaterLily-jl/WaterLilyMeshBodies.jl) `MeshBody` | agrees with WaterLily on the GPU to rounding (3D) |
| [Pathlines.jl](https://github.com/WaterLily-jl/Pathlines.jl) | not tested |
| [WaterLilyNarrowBand.jl](https://github.com/WaterLily-jl/WaterLilyNarrowBand.jl) | not supported (`measure!` already updates only the cells near the body) |
| [InterfaceAdvection.jl](https://github.com/WaterLily-jl/InterfaceAdvection.jl), [LilyPad.jl](https://github.com/WaterLily-jl/LilyPad.jl) | not applicable: own flow or simulation type |

`symmetry=(-2,-3)` works as in BiotSavartBCs: the Biot-Savart sum includes the images across every
subset of the planes. Two planes cost 28 % more FMM time, 3 % of a step.

## Speed

GPU cases of [WaterLily-Benchmarks](https://github.com/WaterLily-jl/WaterLily-Benchmarks), largest
size per case, ms per step on one A100. Reference: WaterLily 1.11.1, or BiotSavartBCs 1.1.0 for
`-biotsavart`. All sizes and the method: [`RESULTS.md`](RESULTS.md).

| case | grid | cells | reference | TurboLily | speedup |
|---|---|---|---|---|---|
| tgv | 256³ | 16.8M | 28.5 | 9.5 | 3.0× |
| tgv-periodic | 128³ | 2.1M | 6.6 | 1.5 | 4.4× |
| sphere | 512×192×192 | 18.9M | 31.6 | 10.7 | 3.0× |
| sphere-biotsavart | 256×96×96 | 2.4M | 9.0 | 3.2 | 2.8× |
| cylinder (moving) | 288×192×64 | 3.5M | 10.2 | 3.8 | 2.7× |
| donut | 128×64×64 | 524k | 3.0 | 1.0 | 3.1× |
| jelly-biotsavart (moving) | 128×128×512 | 8.4M | 30.1 | 12.5 | 2.4× |

Over all 17 case sizes:

| GPU | speedup range | geometric mean |
|---|---:|---:|
| A100 80 GB | 2.4–5.7× | 3.27× |
| GH200 | 2.5–6.0× | 3.35× |
| RTX 3090 | 2.6–7.1× | 3.77× |
| P100 16 GB | 2.4–4.5× | 3.23× |

Sphere at Re=3700 on a GH200, WaterLily BCs:

| grid | cells | WaterLily | TurboLily | speedup |
|---|---|---|---|---|
| 512×192×192 | 19M | 20 | 6 | 3.2× |
| 1024×384×384 | 151M | 151 | 46 | 3.3× |
| 2048×576×576 | 680M | 677 | 205 | 3.3× |

## Validation

`test/runtests.jl` steps each configuration beside WaterLily or BiotSavartBCs (on the CPU) and
requires agreement to rounding and equal pressure-iteration counts: 2D and 3D, `exitBC`, periodic
directions, `uBC(i,x,t)`, `g`, `λ=cds`, moving and deforming bodies, `udf`, thin grids,
`setproperties`, `nonbiotfaces`, `symmetry`. `test/waterlily_suite.jl` runs WaterLily's own test
files on `TurboLily.Simulation`.

- Velocity is compared on the interior and the boundary-normal faces; pressure on cells with no
  closed face (cells cut by the body are ill-conditioned, and two reference runs differ there too).
- Runs are 5–10 steps: no check of grid or time-step convergence.
- A separate test recomputes the final Poisson residual `z − Ax` with WaterLily.
- With periodic directions the agreement is to the solver tolerance (4.2).

Sphere at Re=3700 (48 cells per diameter), 192×96×96, t = 0…100, statistics for t>20. "+1e-6" is the
same code with a perturbed initial velocity (WaterLily 1.11.1, BiotSavartBCs 1.1.0, RTX 2080 Ti):

| | wall time | steps | mean C_d | std C_d |
|---|---:|---:|---:|---:|
| WaterLily | 153 s | 11857 | 0.6304 | 0.0038 |
| WaterLily + 1e-6 | 156 s | 11760 | 0.6296 | 0.0037 |
| TurboLily.Simulation | 53 s | 11816 | 0.6301 | 0.0041 |
| TurboLily.Simulation + 1e-6 | 44 s | 11882 | 0.6309 | 0.0043 |
| BiotSavartBCs | 180 s | 9347 | 0.3567 | 0.0072 |
| BiotSavartBCs + 1e-6 | 172 s | 9490 | 0.3575 | 0.0052 |
| TurboLily.BiotSimulation | 53 s | 9507 | 0.3577 | 0.0044 |
| TurboLily.BiotSimulation + 1e-6 | 60 s | 9452 | 0.3591 | 0.0053 |

## What changed relative to the reference

The discretisation and solvers are unchanged: QUICK/CDS, BDIM, predictor-corrector, the multigrid
V-cycle with adaptive ω, the stopping criteria, the FMℓM sum. Changes are grouped as:

1. algorithmic: work that cannot change the result is skipped;
2. algorithms shared with BiotSavartBCs 1.1, with faster kernels;
3. computational: the same arithmetic, faster;
4. places where results are not identical.

"WL" is WaterLily 1.10.0, "BSBC" BiotSavartBCs 1.1.0, N the number of cells, a "pass" one sweep
over a full-grid array. Measurements: sphere at Re=3700 on one A10 with CUDA.jl 6.4, per step at
4.2M cells (33.6M cells in parentheses). WaterLily 1.11 changes none of this work.

Share of the time saved:

| | from 4.2M cells up | at 66k cells |
|---|---|---|
| kernel efficiency (groups 2, 3) | 77–93 % | 25 % (WaterLily BCs), 51 % (Biot-Savart BCs) |
| algorithmic changes (group 1) | 7–8 % (WaterLily BCs), 10–11 % (Biot-Savart BCs), 14–17 % (moving body) | 3 % (WaterLily BCs), 7–20 % (Biot-Savart BCs) |
| host gaps (launches, syncs, CUDA graphs) | 0–6 % | 72 % (WaterLily BCs), 42 % (Biot-Savart BCs) |

Largest single changes (each measured where it applies, so not additive):

| change | section | saved | applies to simulations with |
|---|---|---|---|
| fused momentum step | 3.1 | 8.6 ms (66 ms) per step | any BCs |
| FMM kernel | 3.4 | 4.9 ms (17 ms) per step | Biot-Savart BCs |
| multigrid kernels and compressed stencil | 3.2, 3.3 | 1.45 ms (14 ms) per V-cycle | any BCs |
| `measure!` only in the body band | 1.5 | 1.5 ms per step | a moving body (any BCs) |
| residual norms and flux fix from the boundary cells | 1.2 | 0.74 ms (5.0 ms) per step | Biot-Savart BCs |
| residual correction folded into the next kernel | 1.4 | 0.68 ms (4.1 ms) per step | any BCs |
| level-1 vorticity only where the FMM reads it | 1.1 | 0.66 ms (6.0 ms) per step | Biot-Savart BCs |
| pressure solved in physical units | 1.3 | 0.49 ms (4.0 ms) per step | any BCs |

### 1. Algorithmic changes

Each relies on a property of the discrete operators, listed so that a change to them shows what
it breaks.

**1.1 Level-1 vorticity only where the FMM reads it (Biot-Savart BCs).**
- BSBC: `fill_ω!` zeroes and writes the curl in every cell, then restricts it level by level.
- TurboLily: `k_curl_layer!` writes the level-1 curl only within 6 (3D) or 10 (2D) cells of a
  domain face, and `k_curl_restrict!` sums level 2 directly from the curl of `u`.
- Relies on: level-1 targets read sources at most 3 (3D) or 7 (2D) cells from the face. If `close`
  or the level-1 source range changes, `layer_width` must follow.
- Saves: 0.66 ms (6.0 ms).

**1.2 Residual norms and flux fix from the boundary cells (Biot-Savart BCs).**
- BSBC: after each Biot update, `sum(r)`, L₁ and L∞ over the whole grid (3 reductions, 2 syncs).
- TurboLily: a Biot update changes r only in the cells B next to the boundary, so
  Σ|r| = Σ|r|(multigrid) − Σ_B|r_old| + Σ_B|r_new| and max|r| = max(non-B, B). Σr is the net
  change of the boundary flux, summed by the boundary kernel.
- Relies on: Σ(Aϵ) = 0, which needs L = 0 on non-periodic domain faces.
- Saves: 0.74 ms (5.0 ms).

**1.3 Pressure in physical units (any BCs).**
- WL: `b.x .*= Δt`, solve, `b.x ./= Δt` (2 passes).
- TurboLily: solves `A p = z/Δt` with tolerances divided by Δt.
- Saves: 0.49 ms (4.0 ms).

**1.4 Residual correction folded into the next kernel (any BCs).**
- WL: r = z − Ax, `sum(r)`, r −= mean, then L₁ and L∞ (2 passes, 3 reductions, 3 syncs).
- TurboLily: the residual kernel evaluates `div(u)` itself (z is never stored) and writes block
  sums; the first Jacobi kernel applies the mean and returns the norms. The solved-row count is
  kept on the device.
- Saves: 0.68 ms (4.1 ms).

**1.5 `measure!` rewrites only the body band (any BCs).**
- WL: resets V, μ₀, μ₁ everywhere, evaluates the sdf everywhere, measures the band; `update!`
  recomputes the multigrid coefficients with several kernels per level.
- TurboLily: one kernel evaluates the sdf and measures the band. A byte per cell records whether
  the cell was in the band and on which side of the body, and only cells that left the band or
  changed side are reset. This holds for any motion between two measures. The same kernel builds
  the BDIM face lists (3.1); coefficients and flags are updated by fused kernels.
- Saves: 1.5 ms per step for a body remeasured every step.

### 2. Algorithms shared with BiotSavartBCs 1.1

Cumulative pressure, vorticity recomputed only in the body box, and FMM re-summed only for
targets that see the body box. TurboLily differs in that `u − Δt·L∇p` is evaluated on the fly,
level-1 vorticity exists only in the layer (1.1), and a target is also re-summed when one of its
symmetry images sees the box.

| per update, 4.2M (33.6M) cells | BSBC | TurboLily |
|---|---|---|
| vorticity | 0.09 ms (0.43) | 0.11 ms (0.44) |
| later FMM update | 0.74 ms (1.34) | 0.12 ms (0.32) |

### 3. Computational changes

**3.1 Momentum step.** 9.6 → 1.0 ms (74 → 7.6 ms), 43–58 % of the time saved.
- WL: 21 launches in 3D, fluxes stored to `σ` and read back, separate passes for `u⁰ .= u`,
  `fill!`, `accelerate!` and two for `BDIM!`.
- One kernel per phase: each thread computes the fluxes through its 2D faces for all components,
  the acceleration and `u⁰ + Δt·r`. A face flux is computed by both neighbours, not stored.
- The upwind stencil is selected before `λ`, with both far values loaded before the selection.
- The core kernel covers the full x range with a branch-free boundary flux; periodic flags and
  strides are type parameters (64 registers). The y and z boundary layers are one thin kernel.
- BDIM runs only on two face lists: μ₀ ≠ 1 with μ₁ = 0, and μ₁ ≠ 0 (8 lanes per face, combined by
  warp shuffles). Elsewhere it is the identity.
- `u⁰ ← u` and the corrector output are pointer swaps.
- The time is set by the traffic of the ±2-cell stencil, not by the QUICK limiter: at 512×192×192
  the kernel moves its nine passes at 84 % of copy bandwidth (1.68 ms on an A10).

**3.2 Compressed Poisson stencil.** Saves 1.9 ms (24 ms).
- WL reads L (D values), D and iD in every cell of every operator.
- A byte per cell flags cells where L equals the level's fluid value on all faces; there nothing
  is loaded (6–7 → 2–3 values per cell). A second bit flags cells whose neighbours are flagged
  too. Stencil values are loaded before branching on the flag.

**3.3 Multigrid kernels.** One V-cycle with smoothing and norms: 2.92 → 1.47 ms (24.5 → 10.1 ms).

| WL | TurboLily |
|---|---|
| `Jacobi!`: ϵ = r·iD, `perBC!`, increment | one kernel; neighbours' r·iD on the fly; wrap in the index |
| `restrict!` after Jacobi | in the Jacobi kernel, by shared-memory block sums |
| `prolongate!`, then `increment!` | one kernel reading the coarse level |
| `GaussSeidelRB!`: init, `perBC!`, 4 half sweeps | init in the first half sweep; sweeps 3 and 4 in one z wavefront |
| one launch per kernel per level | levels of ≤ 2048 cells in one single-block kernel |
| L₁, L∞ per V-cycle: 2 reductions, 2 syncs | in the last increment kernel; 1 host read |
| ω updated on the host | updated on the device |
| about 70 launches per V-cycle | one cached CUDA graph (30 % of the step at 66k cells) |

**3.4 Biot-Savart kernels.** The FMM kernel saves 4.9 ms (17 ms).
- Each source x-line is split around the near-field hole (no exclusion test, same summation
  order), the target component is a compile-time constant, and 1/r³ uses `rsqrt`.
- On coarse levels up to 16 threads share one target.
- The boundary kernel also updates the adjacent residual and sums the flux change, from
  precomputed adjacent-cell indices.

**3.5 General.**
- Reductions: per-block partial sums, reduced in Float64 in two levels (Float64 runs at 1/32
  rate on GeForce). One host copy per reduction, after a sync that blocks in the driver
  (CUDA.jl's default wait idles the GPU 50–200 µs).
- The corrector projection also computes the CFL; residual and projection kernels write the x
  faces of the domain BCs.
- Block shapes: with 32×4×2 blocks a ±2-cell stencil reads each z plane three times, and planes of
  65k cells or more fall out of a 6 MB L2. From 40,000 cells per x-y plane the momentum core uses
  32×1×8 and the streaming multigrid and projection kernels 32×8×1 (512×192×192: 21.1 → 20.0 ms
  on an A10, same results). A GH200 (50 MB L2) shows no difference.
- Blocks above 256 threads are checked against the kernel's register limit and halved to fit.
- Index arithmetic is Int32 (`sum` over Int32 widens). KernelAbstractions' Int64 index divisions
  cost WaterLily's streaming kernels a third of the bandwidth at 18.9M cells.
- Loops over dimensions are fully unrolled; face kernels run threads in the plane; the BDIM lists
  are built with warp-aggregated atomics.

### 4. Where results differ from the reference

**4.1 Rounding.** Block-wise reductions, `rsqrt`, the FMM summation order and the cumulative
pressure change the order of operations. 3D flows amplify this as they amplify a 1e-6 perturbation
of the reference (see Validation).
- Pressure in cells cut by the body can differ by up to 0.1, as between two reference runs;
  cells with all faces open agree to 1e-5.
- With Biot-Savart BCs the pressure can differ by a uniform constant of about 1e-3.
- Σr drifts in Float32 (1.2 assumes it stays zero): about 1e-9 in the flux fix.

**4.2 Periodic smoother.** WL fills the periodic ghosts once per `GaussSeidelRB!`, so sweeps 2–4
read stale values. TurboLily wraps the neighbour index in every half sweep, except on levels with
an odd periodic length (the two cells across the wrap have the same colour), which use WL's
method. Results then agree to the solver tolerance (against WaterLily 1.10):

| case | steps | max \|Δu\|, default tol | max \|Δu\|, tol=1e-6 | V-cycles per solve at tol=1e-6, WL / TurboLily |
|---|---|---|---|---|
| Taylor-Green vortex 32³, fully periodic | 10 | 1.9e-3 | 1.5e-5 | 5.3 / 4.3 |
| sphere 64×32×32, periodic in z, `exitBC` | 5 | 2.3e-4 | 3.8e-5 | 5.2 / 4.7 |

**4.3 `udf`.** As in WaterLily: the predictor callback sees `flow.u` with a zero interior, the
corrector callback the projected predictor velocity; advect with the third argument of
`udf(flow, u, t; kwargs...)`.

## Tests

The tests need a CUDA GPU; CI only checks that the dependencies resolve.

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

In the vendored WaterLily suite (`test/waterlily/`, changes commented in place), Float64 tests run
in Float32, `mem=Array` references are WaterLily itself, `test_alloc.jl` is skipped (SIMD backend),
and 4 tests are marked broken (3 ForwardDiff, 1 on `pois.levels[1].r`).

## License

MIT, see [`LICENSE`](LICENSE). Parts of TurboLily are derived from WaterLily.jl and
BiotSavartBCs.jl by Gabriel Weymouth, both MIT-licensed. TurboLily was written with
[Claude Code](https://claude.com/claude-code).
