# TurboLily

CUDA rewrite of the [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl) time step and of the
[BiotSavartBCs.jl](https://github.com/WaterLily-jl/BiotSavartBCs.jl) pressure solver. It keeps WaterLily's
API: `TurboLily.Simulation` and `TurboLily.BiotSimulation` take the same arguments as
`WaterLily.Simulation` and `BiotSavartBCs.BiotSimulation`, and they return an ordinary
`WaterLily.Simulation`. `sim_step!`, `measure!`, `sim_time`, `pressure_force`, plotting and I/O
keep working unchanged.

```julia
using WaterLily, TurboLily, StaticArrays

body = AutoBody((x,t)->√sum(abs2, x .- SA[64f0,64f0,64f0]) - 32f0)
sim  = TurboLily.Simulation((256,128,128), (1,0,0), 64; ν=64/3700, body)        # WaterLily
bsim = TurboLily.BiotSimulation((256,128,128), (1,0,0), 64; ν=64/3700, body)    # Biot-Savart BCs
sim_step!(bsim, 10; remeasure=false)
```

The arrays always live on the current CUDA device (`mem` is ignored) and `T` must be `Float32`.
`WaterLily.setproperties(sim.flow; uBC)` works as for a WaterLily flow.

TurboLily is not registered. Install it from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/KristianHolme/TurboLily.jl")
```

It requires WaterLily 1.9 (1.9.1 or later) or 1.10, and BiotSavartBCs 1.0 or later. The test
suite passes against WaterLily 1.9.1 with BiotSavartBCs 1.0.0, and against WaterLily 1.10.0
with BiotSavartBCs 1.0.1, with Julia 1.13.1 and CUDA.jl 6.2.2 and 6.4.

Symmetry planes take the same `symmetry` keyword as `BiotSavartBCs.BiotSimulation`:

```julia
# square plate on two symmetry planes, the lower y and z faces
sim = TurboLily.BiotSimulation((2N,N,N), (1,0,0), N/2; ν, body, symmetry=(-2,-3))
```

`-j` is the lower and `+j` the upper face normal to j, at most one per direction. These faces
get no Biot-Savart BCs, and the Biot-Savart sum includes the reflection of the domain across
every non-empty subset of the planes (for two planes: three images), with the normal velocity
sign flipped across each plane. Without `symmetry` the FMM kernel has no image loop. Two
planes cost about 28 % more FMM time, about 3 % of a time step.

## Speed

### WaterLily-Benchmarks cases

These are the GPU cases of
[WaterLily-Benchmarks](https://github.com/WaterLily-jl/WaterLily-Benchmarks), with the suite's
method: developed-flow checkpoints, and the minimum over 5 runs of 25 steps. The reference is
WaterLily 1.9.1, or BiotSavartBCs 1.0.0 for the `-biotsavart` cases, on the same GPU. One
NVIDIA A10 (22.5 GB), Float32, times in ms per step. Each row is the largest power-of-two size
of the case that fits on the card. All sizes are in [`RESULTS.md`](RESULTS.md).

| case | grid | cells | reference | TurboLily | speedup |
|---|---|---|---|---|---|
| tgv | 512³ | 134M | 670 | 171 | 3.9× |
| tgv-periodic | 512³ | 134M | 699 | 165 | 4.2× |
| sphere | 1024×384×384 | 151M | 772 | 188 | 4.1× |
| sphere-biotsavart | 1024×384×384 | 151M | 1021 | 231 | 4.4× |
| cylinder (moving) | 576×384×128 | 28M | 161 | 46 | 3.5× |
| donut | 512×256×256 | 34M | 159 | 44 | 3.6× |
| jelly-biotsavart (moving) | 256×256×1024 | 67M | 547 | 171 | 3.2× |

Over all sizes, from 66k cells up, the speedup is 2.5–5.3×. From 16M cells up it is 3.1–4.4×.
On large grids both codes are limited by memory bandwidth, and on small grids (below about 1M
cells) by per-step overhead. Both codes need about the same GPU memory. The
`cylinder-biotsavart` case is not supported (see 3.4).

### Sphere and Taylor-Green vortex (RTX 2080 Ti)

One RTX 2080 Ti, Float32, ms per step. The flow is a sphere at Re=3700 (`D = N/2`,
256×128×128, 4.2M cells) unless noted. The reference is WaterLily 1.9.1 and BiotSavartBCs
1.0.0 on the same GPU.

| case | reference | TurboLily | speedup |
|---|---|---|---|
| WaterLily, static body | 19.2 | 6.1 | 3.2× |
| WaterLily, moving body (`remeasure=true`) | 23.6 | 8.4 | 2.8× |
| WaterLily, 512×256×256 (33.6M cells) | 148.6 | 41.2 | 3.6× |
| WaterLily, TGV 256³ | 76.7 | 20.3 | 3.8× |
| Biot-Savart, static body | 31.5 | 8.6 | 3.7× |
| Biot-Savart, moving body | 45.3 | 11.0 | 4.1× |
| Biot-Savart, 480×192×192 (17.7M cells) | 109.4 | 29.8 | 3.7× |

## Validation

Every configuration is stepped side by side with the reference implementation, and the results
must agree to rounding: `test/runtests.jl` contains checks against WaterLily 1.9.1 and
BiotSavartBCs 1.0.0. These cover 2D and 3D, `exitBC`, periodic directions (including a fully
periodic Taylor-Green vortex and grids with odd periodic multigrid levels), `uBC(i,x,t)`, `g`,
moving and deforming bodies (WaterLily's pulsating-circle test bodies) with `remeasure=true`,
`measure!` after a jump in time, `udf` (body force, the Smagorinsky `sgs!`, and the state of
`flow.u` that the callback sees), thin grids such as 2×128, `setproperties(flow; uBC)`,
Biot-Savart with `nonbiotfaces` and `symmetry` planes (2D and 3D), and the BiotSavartBCs
potential-flow test. The pressure-iteration counts are identical, except with periodic
directions (see 3.2).

What the comparisons establish:

- The reference runs on the CPU, TurboLily on the GPU.
- The reference is patched: the tests replace `WaterLily.GaussSeidelRB!` on levels with an odd
  last dimension, where WaterLily 1.9.1 skips cells
  ([WaterLily#353](https://github.com/WaterLily-jl/WaterLily.jl/pull/353)).
- Velocity is compared on the interior and on the boundary faces normal to each component.
  Tangential ghost values are not compared: neither code reads them, and TurboLily leaves
  scratch values there.
- Pressure is compared on cells with no fully closed face (`μ₀>0` on every face). Cells with a
  closed face next to the body are ill-conditioned, and two reference runs differ there too.
- Most cases run 5–10 steps. The tests do not establish grid or time-step convergence or
  long-time statistics.
- Equal iteration counts are a regression check, not an accuracy check. A separate test
  recomputes the final Poisson residual `z - Ax` from scratch with WaterLily, independent of
  the residual that the solver updates incrementally.

The `3D sphere uBC(t) + g` case has a velocity tolerance of 3e-4 instead of 1e-4. Its second
step is Δt≈5.5, and the WaterLily 1.9.1 and 1.10.0 references differ from each other by
1.9e-4 on it.

The velocity differences are of the same size as those between two runs of the reference code
when one run gets a 1e-6 perturbation. The 3D flows amplify rounding differences in both codes
in the same way.

Long runs of a sphere at Re=3700, 192×96×96, t = 0…20:

| | wall time | steps | mean C_d (t>10) |
|---|---|---|---|
| BiotSavartBCs | 25.5 s | 1165 | 0.3895 |
| TurboLily.BiotSimulation | 5.4 s (4.7×) | 1166 | 0.3897 |
| WaterLily | 13.7 s | 852 | 0.4140 |
| TurboLily.Simulation | 2.7 s (5.1×) | 852 | 0.4141 |

## What changed relative to the reference

The discretisation and the solvers are WaterLily's and BiotSavartBCs': QUICK/CDS convection,
BDIM, the predictor-corrector step, the multigrid V-cycle (Jacobi pre-smoothing, red-black
Gauss-Seidel post-smoothing, adaptive ω), the stopping criteria ([BiotSavartBCs#13](https://github.com/WaterLily-jl/BiotSavartBCs.jl/pull/13)), and the
FMℓM Biot-Savart sum. The changes fall into three groups:

1. algorithmic changes, which skip work that cannot change the result;
2. computational changes, which do the same arithmetic faster;
3. places where the results are not identical to the reference.

Below, "WL" means WaterLily 1.9.1 and "BSBC" means BiotSavartBCs 1.0.0. N is the number of
cells. A "pass" is one sweep over a full-grid array.

### 1. Algorithmic changes (same result in exact arithmetic, less work)

Each of these relies on a property of the discrete operators. The property is listed so that a
future change to those operators knows what it could break.

#### 1.1 Level-1 vorticity only where the FMM reads it (Biot-Savart)

**BSBC.** Every Biot update calls `fill_ω!`:

1. zero the level-1 vorticity array (`flow.f`, 3N values in 3D);
2. compute the centred curl in every cell 3..N-2 and write it;
3. `restrict!` level 2 by reading the whole level-1 array back and summing each 2×2×2 block;
4. restrict level 3 and coarser the same way.

**What the level-1 values are used for.** Only two things:

- the sources of the level-2 sum;
- the near-field sources of the level-1 (finest) interactions.

A level-1 target on a domain face sums sources in `up(close(down(T))) ∩ inside(buff=2)`. With
`close` = ±2 cells (3D) or ±4 cells (2D), that set lies at most 3 cells (3D) or 7 cells (2D)
from the face, for example x = 3..5 for a target on the x = 1 face. The level-1 vorticity in
the rest of the domain is computed, written and read back, but only ever used through the
level-2 sum.

**TurboLily.** Two kernels:

- `k_curl_layer!` computes and writes the level-1 curl only in cells within 6 (3D) or 10 (2D)
  cells of a domain face. That is the near-field region plus a margin of 3.
- `k_curl_restrict!` computes level 2 directly: one thread per level-2 cell evaluates the curl
  of its 2×2×2 fine cells (zero outside 3..N-2, as BSBC) and stores the sum. The full level-1
  field is never written or read back.

Levels 3 and coarser are restricted from level 2 as in BSBC.

**Why equal.** The level-2 values are the same sums of the same curls, and the level-1
interactions read only cells that are computed.

**Relies on:** the layer width covering the FMM near field. If `close` or the level-1 source
range in `interaction` changes, `layer_width` must change with it.

**Saves:** writing 3N values, zeroing 3N values and reading 3N values back per vorticity
evaluation. At 4.2M cells the two kernels take about 0.07 ms and 0.16 ms.

#### 1.2 Vorticity recomputed only next to the body (Biot-Savart)

**BSBC.** After each round of V-cycles, `project_update!` applies the pressure increment to
the velocity in a full pass (`u -= μ₀∇x`), and the vorticity of the new velocity is recomputed
everywhere (1.1). There are 3–6 such Biot updates per time step.

**TurboLily.** Within a projection, `u` is left unchanged (see 1.4), and the vorticity is that
of `u − Δt·L∇p`, evaluated on the fly. At the start of the projection the full curl of `u` is
computed once (1.1). After that, only the "affected" cells are recomputed with the current p:
cells whose curl stencil reads a face where L ≠ 1, and their level-2 parents. Levels 3 and
coarser are rebuilt from level 2, which is small.

**Why equal.** The curl is linear, so curl(u − Δt·L∇p) = curl(u) − Δt·curl(L∇p). Where L = 1
on every face the stencil reads, curl(L∇p) is the centred curl of a discrete gradient. That is
identically zero: both cross terms reduce to
`(p[I+δj+δk] − p[I+δj−δk] − p[I−δj+δk] + p[I−δj−δk])/4`. So ω can only change where some L ≠ 1,
which is next to the body. L = μ₀ is exactly 1 away from the body, because μ₀(d,ϵ) = 1 for
d ≥ ϵ, and is 0 inside it.

A check with a random pressure on 160×64×64: recomputing 1,088 of 94,792 level-2 cells
reproduces the direct computation to 2e-6, which is Float32
rounding of `u − Δt∇p`. Skipping them gives errors of 7.7.

**Relies on:**

- the centred curl stencil and the staggered `L∇p` update being the ones above;
- μ₀ being exactly 1 away from the body;
- interior `u` not changing between Biot updates. The curl of cells 3..N-2 never reads the
  boundary-normal faces, which the Biot update does change.

The affected list is rebuilt whenever μ₀ changes (`update!` after `measure!`).

**Saves:** one full velocity pass and one full vorticity evaluation per Biot update.

#### 1.3 Incremental FMM (Biot-Savart)

**BSBC.** Every Biot update evaluates the full multilevel sum: every target on every level sums
its source box, roughly 300–400 sources per target per level in 3D. Then the coarse-to-fine
projection of the target values runs.

**TurboLily.** The first update of each projection does the same. After that, ω changes only
in the affected cells (1.2), so the sum is evaluated on Δω (new minus old):

1. the curl kernels write Δω at the affected cells;
2. levels 3 and coarser are restricted from Δω;
3. each target sums only over its source box intersected with the bounding box of the nonzero
   Δω on that level, so most targets exit immediately;
4. the coarse-to-fine projection runs on the Δ values, and the result is added to the stored
   boundary velocities.

**Why equal.** Both the interaction sums and the projection are linear in ω.

**Relies on:** the boundary velocities of the previous update staying stored, in the target
slots of `flow.f`, for the duration of the projection.

**Saves:** most of the FMM work in all but the first Biot update of each projection.

#### 1.4 Cumulative pressure, one velocity update at the end (Biot-Savart)

**BSBC.**

1. `a.p .*= Δt`, then `project_update!` applies the initial pressure to `u` (a full pass),
   copies it to an accumulator `b.p` and zeroes `a.p`.
2. Each round of V-cycles solves for an increment x starting from zero.
3. `project_update!` applies x to `u` (a full pass) and adds it to `b.p`.
4. At the end, `a.p .= b.p ./ Δt`.

**TurboLily.** The multigrid continues on the total pressure p, with the residual kept
consistent. `u` is not touched during the projection, and the single update
`u −= Δt·L∇p` is applied once at the end.

**Why equal.** The residual after applying an increment is `div(u − Δt·L∇x)/Δt − ... = z − A·p`
either way. A V-cycle on the increment from zero adds the same correction as a V-cycle on the
total, because the V-cycle only adds corrections computed from r.

**Saves:** 2 full passes per Biot update, plus the accumulator array and the scaling passes.

#### 1.5 Residual norms and flux fix from the boundary cells only (Biot-Savart)

**BSBC.** After each Biot update:

- `fix_resid!` computes `sum(r)` over the whole grid: a reduction and a host sync;
- the convergence test computes L₁ and L∞ of r: two more full reductions and syncs.

**TurboLily.**

- A Biot update changes the residual only in the cells next to the domain boundary (the set B).
  The last multigrid increment kernel already reports Σ|r| over all cells and max|r| over the
  cells *not* in B. After the update:
  - Σ|r| = Σ|r|(multigrid) − Σ_B|r_old| + Σ_B|r_new|;
  - max|r| = max(max over non-B cells, max_B|r_new|).
- For the flux fix, Σr before the update is zero, because the previous fix made it zero and
  multigrid does not change Σr. So Σr after the update is just the net change of the boundary
  flux, which the boundary kernel sums over the targets.

**Why equal.**

- The norm updates are exact, since only cells in B change.
- Σr is invariant under multigrid: Σ_I (Aϵ)_I = 0, because each interior face contributes
  with both signs and L = 0 on the domain faces.

**Relies on:** non-periodic domains (`TurboLily.BiotSimulation` does not support periodic
directions, see 3.4) and L = 0 on the domain faces.

**Saves:** 3 full reductions and 2 host syncs per Biot update. The remaining reductions run
over about N^(2/3) cells.

#### 1.6 Pressure solved in physical units (both)

**WL.** `mom_project!` does `b.x .*= Δt` (a full pass), solves `A x = z`, applies `Δt·∇p`
through x, then `b.x ./= Δt` (another full pass).

**TurboLily.** Solves `A p = z/Δt` directly. The residual tolerances are divided by Δt, so
`max|Δt·r| < tol` is the same test.

**Saves:** 2 full passes per projection.

#### 1.7 Residual correction folded into the next kernel (both)

**WL.** `solver!` → `residual!`:

1. r = z − A·x in one pass, then `sum(r)` (a reduction and a host sync);
2. if the mean is not negligible, r −= mean in a second pass;
3. then L₁(r) and L∞(r), as two more reductions with syncs.

**TurboLily.**

- The residual kernel computes `div(u)` itself, so the source z is never stored. It also writes
  per-block partial sums of r.
- The number of solved rows is computed once per `update!` and kept on the device.
- The mean correction is applied on the fly by the first Jacobi kernel of the first V-cycle,
  which also produces the initial L₁ and L∞.

**Saves:** 2 passes, 3 reductions and 3 host syncs per solve.

#### 1.8 `measure!` rewrites only the body band (both)

**WL.** `measure!`:

1. resets V to 0, μ₀ to 1 and μ₁ to 0 everywhere: memsets of D + D + D² values per cell;
2. evaluates the sdf everywhere into `σ`;
3. fully measures the cells with d² < (2+ϵ)², and sets μ₀ = 0 in the cells deeper inside.

`update!` then recomputes the multigrid coefficients with several KA kernels per level.

**TurboLily.** One kernel evaluates the sdf everywhere (still written to `σ`). It fully
measures the band cells. A 1-byte array remembers, for each cell, whether it was in the band at
the previous measure, and if not, on which side of the body it was. A cell outside the band is
reset to the far-field values (μ₀ ∈ {0,1}, μ₁ = 0, V = 0) only if it was in the band or has
changed side since. All other cells are left alone, since their values cannot have changed.
This holds for any body motion between two measures, also a jump in time. The same kernel
appends the BDIM face lists (see 2.1). The multigrid coefficients, regular-cell
flags (2.2) and the solved-row count are updated by fused kernels, with one host sync per
measure.

**Saves:** the D² + 2D value memsets and several kernels per measure. A static body
remeasured every step (the `sim_step!` default) costs about 1.5 ms at 4.2M cells.

### 2. Computational changes (same arithmetic, faster execution)

#### 2.1 Momentum step

**WL.**

- `conv_diff!` loops over all (component i, direction j) pairs. Per pair there are four
  kernels: lower boundary, interior flux written to `σ`, interior flux subtracted from the
  neighbour, upper boundary. That is 36 launches in 3D, with the flux stored and read back.
- `accelerate!` is another pass.
- `BDIM!` makes 2 passes over all faces, reading μ₀, μ₁ (D² values per cell) and V.
- `scale_u!` is another pass, and `u⁰ .= u` is a copy.

**TurboLily.**

- One kernel per phase. Each thread computes, for its cell and all D components, the fluxes
  through its 2D faces, the acceleration and the update `u⁰ + Δt·r`. Each face flux is evaluated
  by both adjacent cells instead of being stored: computing is cheaper than the store and load.
- In each flux the upwind stencil is selected first, so `λ` (QUICK) is evaluated once instead
  of once per branch. The far point is loaded through a single selected address.
- Domain-boundary faces use the general flux, interior faces a branch-free one. The y and z
  boundary layers run as separate thin kernels.
- The full BDIM update (μ₀, μ₁, V) runs only on two precomputed lists:
  - faces with μ₀ ≠ 1 and μ₁ = 0, where faces fully inside the body reduce to u = V;
  - faces with μ₁ ≠ 0, the shell.

  Everywhere else BDIM is the identity. μ₁ = 0 wherever μ₀ = 1, because the kernel moment
  `kern₁(±1) = 0`.
- `u⁰ ← u` and the corrector output are pointer swaps.

#### 2.2 Compressed Poisson stencil

**WL.** Every multigrid operator (mult, Jacobi, Gauss-Seidel, increment) reads L (D values per
cell), D and iD in every cell.

**TurboLily.** A 1-byte flag per cell marks cells where all 2D faces have L equal to the level's
fluid value (1 on the fine level, a fixed constant on coarse levels). There D and iD are
constants and nothing is loaded. Flagged cells are most of the domain, everything not next to
the body or the domain boundary. This reduces the bytes per cell of every multigrid operator
from about 6–7 values to about 2–3.

#### 2.3 Multigrid kernels

| WL | TurboLily |
|---|---|
| `Jacobi!`: ϵ = r·iD (pass), `perBC!`, increment (pass) | one kernel: reads r·iD of the neighbours on the fly; periodic wrap is in the index |
| `prolongate!` (pass), then `increment!` (pass) | one kernel: the piecewise-constant prolongation is read from the coarse level on the fly |
| `GaussSeidelRB!`: init ϵ = r·iD (pass), `perBC!`, 4 half sweeps | init fused into the first half sweep |
| `L₁` and `L∞` after each V-cycle: 2 reductions, 2 host syncs | computed inside the final increment kernel; 1 host read per V-cycle |
| adaptive ω updated on the host and passed to kernels | updated on the device by a 1-thread kernel |
| each V-cycle is about 70 launches with host overhead | the whole V-cycle + smoothing is replayed as one cached CUDA graph |

#### 2.4 Biot-Savart kernels

- **FMM interaction.** Each source x-line is split around the near-field hole, so the inner
  loop has no exclusion test, and the target's velocity component is a compile-time constant,
  so only the two vorticity components that enter are loaded. This idea was proposed in the
  [BiotSavartBCs#14](https://github.com/WaterLily-jl/BiotSavartBCs.jl/pull/14), but that part was not merged: BiotSavartBCs keeps the per-source loop
  with a `S ∉ Rinner` test. The split keeps the summation order, so the two differ only by
  rounding. TurboLily also computes 1/r³ with `rsqrt`. On coarse levels, where there are few
  targets with large source boxes, P = 2–32 threads share one target and reduce with warp
  shuffles.
- **Residual source and flux fix.** The residual kernel computes the source from `u` directly,
  so z is never stored. The boundary-velocity kernel also updates the residual next to the
  boundary and sums the net flux change.
- **`pflowBC!`.** One kernel per direction (tangential components, then normal) instead of one
  per (direction, component, side).

#### 2.5 General GPU details

- Reductions write per-block partial sums, which one small kernel then reduces in Float64.
  There are no atomics on a single address.
- All index arithmetic is Int32. Note that Julia's `sum` over Int32 widens to Int64.
- Loops over the dimensions are fully unrolled, so tuples indexed by the dimension stay in
  registers.
- Kernels on a domain face run threads over the two in-plane directions (WaterLily does the
  same since [#346](https://github.com/WaterLily-jl/WaterLily.jl/pull/346)).
- The BDIM band lists are built with warp-aggregated atomics instead of `findall`.

### 3. Where results are not identical to the reference

#### 3.1 Floating-point rounding (all cases)

Groups 1 and 2 change the order of floating-point operations:

- block-wise Float64 reductions;
- `rsqrt` and a different summation order in the FMM;
- vorticity and pressure accumulated incrementally.

Velocities therefore agree with the reference to rounding, not bit for bit. The 3D flows
amplify these differences exactly as they amplify a 1e-6 perturbation of the reference itself
(see Validation). Consequences visible in the output:

- **Partially blocked cells.** Pressure in cells with some faces closed by the body is
  ill-conditioned and can differ by up to about 0.1. It differs by similar amounts between two
  runs of the reference. Cells with all faces open agree to about 1e-5.
- **Pressure gauge.** With Biot-Savart BCs the pressure can differ from BSBC by a uniform
  constant of about 1e-3. This does not affect the velocity or the pressure force on a closed
  body.
- **Σr drift.** 1.5 assumes Σr stays exactly zero through the multigrid. In Float32 it drifts
  slightly; BSBC recomputes it and TurboLily does not. The resulting difference in the flux
  fix is around 1e-9.

#### 3.2 Periodic directions use a slightly different smoother

**WL.** `GaussSeidelRB!` fills the periodic ghost cells once (`perBC!`), then runs 4 half
sweeps. Sweeps 2–4 therefore read stale ghost values on periodic boundaries.

**TurboLily.** On levels where every periodic length is even, neighbours are wrapped across
periodic boundaries in every half sweep, so they always read current values. On a level with
an odd periodic length, the first and last interior cells are neighbours across the wrap and
have the same red-black colour. A wrapped read would then race with the update of that
neighbour. These levels therefore use WaterLily's method: ghosts filled once, no wrap. Odd
levels occur also for even grids: a periodic 48 or 96 coarsens to a 3-cell level.

**Effect.** This is a different, slightly better smoother. Results agree to the solver tolerance
(differences of about 2e-4 after a few steps) rather than to rounding. With a tight tolerance
in both codes they agree to 1e-6. On the Taylor-Green vortex TurboLily needs one V-cycle
fewer per solve.

#### 3.3 `udf` callbacks

As in WaterLily, the predictor callback sees `flow.u` with a zero interior, and the corrector
callback sees the projected predictor velocity. The velocity to advect with is the third
argument, `udf(flow, u, t; kwargs...)`. TurboLily may leave different values than WaterLily
in the tangential ghost cells of `flow.u`.

#### 3.4 Not implemented

These raise an `ArgumentError`:

- `fmm=false` (the tree sum).
- Biot-Savart BCs on grids too small for a second Biot-Savart level, e.g. 8×8.
- Periodic directions with Biot-Savart BCs ([BiotSavartBCs#19](https://github.com/WaterLily-jl/BiotSavartBCs.jl/pull/19)).

Also not implemented: `T=Float64` and non-CUDA arrays.

Not supported: bodies wrapped in
[WaterLilyNarrowBand.jl](https://github.com/WaterLily-jl/WaterLilyNarrowBand.jl)'s
`NarrowBand` cache. TurboLily already measures only near the body, and an experimental
integration gave little gain: about 5% per step for a 16-ring composite body and 69% slower
for a sphere.

## Multi-GPU

A single simulation runs on one GPU. Measured on a workstation with 4× RTX 2080 Ti (PCIe, no
NVLink):

- Peer-to-peer access between the GeForce cards is disabled. GPU-to-GPU copies go through
  the host at about 1.7 GB/s, or 6 GB/s with pinned staging, with 25-100 µs latency.
  Exchanging halos inside every multigrid smoothing step would cost more than one whole
  V-cycle takes on one GPU (1.45 ms at 4.2M cells). So domain decomposition only pays off for
  problems that do not fit in one card's 11 GB, which is roughly 60M cells.
- Reactant.jl (XLA) runs on these sm_75 cards. A 7-point stencil sweep in Reactant takes
  about 3× as long as a hand-written CUDA kernel. Sharding it over 4 GPUs gave no speedup, and
  the sharded result differed from the single-device one by 2e-3.
- Independent simulations, one per GPU from one Julia process, work as they are: see
  `examples/multigpu_sweep.jl`, started with `julia -t 4`.

## Tests

The tests step every case with TurboLily on the GPU and with WaterLily or BiotSavartBCs on the
CPU, so they need a CUDA GPU:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The benchmark scripts are not part of this repository. [`RESULTS.md`](RESULTS.md) has the full
tables, the method of each benchmark, and earlier comparisons against WaterLily and
BiotSavartBCs pull requests.
