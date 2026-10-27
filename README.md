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

It requires WaterLily 1.10 and BiotSavartBCs 1.1: TurboLily follows the latest minor versions.

Tested with:

- WaterLily 1.10.0 and master (52b2207), with BiotSavartBCs 1.1.0;
- Julia 1.13 and CUDA.jl 6.4;
- NVIDIA RTX 2080 Ti (compute capability 7.5) and A10 (8.6) GPUs.

### Other WaterLily packages

Bodies from other packages work if they can be measured on the GPU. Bodies that hold arrays
must be built with `mem=CuArray`, because TurboLily ignores the `mem` of the simulation. These
were checked over five steps against WaterLily, with the same body:

| package | status |
|---|---|
| [ParametricBodies.jl](https://github.com/WaterLily-jl/ParametricBodies.jl): `ParametricBody`, `HashedBody` | agrees with WaterLily on the CPU to rounding (2D, static and moving) |
| [WaterLilyMeshBodies.jl](https://github.com/WaterLily-jl/WaterLilyMeshBodies.jl): `MeshBody` | agrees with WaterLily on the GPU to rounding (3D). `MeshBody` itself measures slightly differently on the CPU and the GPU (μ₀ differs by 3e-4). |
| [Pathlines.jl](https://github.com/WaterLily-jl/Pathlines.jl) | not tested |
| [WaterLilyNarrowBand.jl](https://github.com/WaterLily-jl/WaterLilyNarrowBand.jl) | not supported (see 4.4) |
| [InterfaceAdvection.jl](https://github.com/WaterLily-jl/InterfaceAdvection.jl), [LilyPad.jl](https://github.com/WaterLily-jl/LilyPad.jl) | not applicable: they have their own flow or simulation type |

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
WaterLily 1.10.0, or BiotSavartBCs 1.1.0 for the `-biotsavart` cases, on the same GPU. One
NVIDIA A10 (22.5 GB), Float32, times in ms per step. Each row is the largest size of the case
that has a checkpoint in the suite. All sizes are in [`RESULTS.md`](RESULTS.md).

| case | grid | cells | reference | TurboLily | speedup |
|---|---|---|---|---|---|
| tgv | 256³ | 16.8M | 75.9 | 18.0 | 4.2× |
| tgv-periodic | 128³ | 2.1M | 12.4 | 2.4 | 5.1× |
| sphere | 512×192×192 | 18.9M | 85.1 | 21.1 | 4.0× |
| sphere-biotsavart | 256×96×96 | 2.4M | 16.8 | 4.3 | 3.9× |
| cylinder (moving) | 288×192×64 | 3.5M | 22.2 | 6.4 | 3.5× |
| donut | 128×64×64 | 524k | 5.4 | 1.3 | 4.2× |
| jelly-biotsavart (moving) | 128×128×512 | 8.4M | 68.3 | 24.4 | 2.8× |

Over all sizes, from 66k cells up, the speedup is 2.8–7.6×; from 2M cells up it is 2.8–5.1×.
On large grids both codes are limited by memory bandwidth, and on small grids (below about 1M
cells) by per-step overhead. Both codes need about the same GPU memory. The
`cylinder-biotsavart` case is not supported (see 4.4). Where BiotSavartBCs 1.1 uses the same
algorithm as TurboLily, TurboLily is faster through its kernels (see section 2).

## Validation

Every configuration is stepped side by side with the reference implementation, and the results
must agree to rounding: `test/runtests.jl` contains checks against WaterLily 1.10 and BiotSavartBCs
1.1. These cover 2D and 3D, `exitBC`, periodic
directions (including a fully periodic Taylor-Green vortex and grids with odd periodic
multigrid levels), `uBC(i,x,t)`, `g`, `λ=cds`,
moving and deforming bodies (WaterLily's pulsating-circle test bodies) with `remeasure=true`,
`measure!` after a jump in time, `udf` (body force, the Smagorinsky `sgs!`, and the state of
`flow.u` that the callback sees), thin grids such as 2×128, `setproperties(flow; uBC)`,
Biot-Savart with `nonbiotfaces` and `symmetry` planes (2D and 3D), and the BiotSavartBCs
potential-flow test. The pressure-iteration counts are identical, except with periodic
directions (see 4.2).

What the comparisons establish:

- The reference runs on the CPU, TurboLily on the GPU.
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
step is Δt≈5.5, and after 5 steps WaterLily with a 1e-6 velocity perturbation differs from
WaterLily by 2.0e-4 (TurboLily: 1.95e-4).

The velocity differences are of the same size as those between two runs of the reference code
when one run gets a 1e-6 perturbation. The 3D flows amplify rounding differences in both codes
in the same way.

Long runs of a sphere at Re=3700 (diameter 48 cells), 192×96×96, t = 0…20 (WaterLily 1.10.0,
BiotSavartBCs 1.1.0, RTX 2080 Ti):

| | wall time | steps | mean C_d (t>10) | std C_d (t>10) |
|---|---|---|---|---|
| BiotSavartBCs | 30.0 s | 1849 | 0.3750 | 0.0158 |
| TurboLily.BiotSimulation | 8.4 s (3.6×) | 1850 | 0.3757 | 0.0158 |
| WaterLily | 22.2 s | 2238 | 0.6278 | 0.0152 |
| TurboLily.Simulation | 7.0 s (3.2×) | 2227 | 0.6313 | 0.0190 |

The differences are of the same size as the spread of each code under a 1e-6 perturbation of
the initial velocity. In the same setup, from t>10:

| | steps | mean C_d | std C_d |
|---|---|---|---|
| WaterLily | 2238 | 0.6278 | 0.0152 |
| WaterLily + 1e-6 | 2230 | 0.6311 | 0.0149 |
| TurboLily.Simulation | 2227 | 0.6313 | 0.0190 |
| TurboLily.Simulation + 1e-6 | 2248 | 0.6304 | 0.0147 |
| BiotSavartBCs | 1850 | 0.3760 | 0.0142 |
| BiotSavartBCs + 1e-6 | 1850 | 0.3759 | 0.0144 |

Two unperturbed BiotSavartBCs runs also differ (0.3750 and 0.3760), so they are not
bitwise reproducible.

## What changed relative to the reference

The discretisation and the solvers are WaterLily's and BiotSavartBCs': QUICK/CDS convection,
BDIM, the predictor-corrector step, the multigrid V-cycle (Jacobi pre-smoothing, red-black
Gauss-Seidel post-smoothing, adaptive ω), the stopping criteria, and the
FMℓM Biot-Savart sum. The changes fall into four groups:

1. algorithmic changes, which skip work that cannot change the result, and which the references
   do not have;
2. algorithms shared with BiotSavartBCs 1.1, where TurboLily is faster only through its
   kernels;
3. computational changes, which do the same arithmetic faster;
4. places where the results are not identical to the reference.

Below, "WL" means WaterLily 1.10.0 and "BSBC" means BiotSavartBCs 1.1.0. N is the number of
cells. A "pass" is one sweep over a full-grid array.

### Where the speedup comes from

Both codes were instrumented phase by phase, and single TurboLily optimizations were switched off,
for the sphere at Re=3700 on one A10 (grids of 66k to 33.6M cells). The time
saved per step splits as follows:

| | share of the time saved, from 4.2M cells up | at 66k cells |
|---|---|---|
| kernel efficiency (group 3, and the kernels of group 2) | 77–92 % | 25 % (WaterLily BCs), 51 % (Biot-Savart BCs) |
| algorithmic changes (group 1) | 8 % (WaterLily BCs), 10–11 % (Biot-Savart BCs), 14–17 % (moving body) | 3 % (WaterLily BCs), 7–20 % (Biot-Savart BCs) |
| host gaps (fewer launches and host syncs, CUDA graphs) | 0–6 % | 72 % (WaterLily BCs), 42 % (Biot-Savart BCs) |

The moving-body cases were measured from 524k cells up. With Biot-Savart BCs the algorithmic
share is a range: it depends on whether the skipped work is priced at the reference's kernel
speed or at TurboLily's, and the kernel-efficiency share moves by the same amount (the table
shows it for TurboLily's speed).

Both codes are limited by memory bandwidth on grids above about 1M cells. The reference kernels
run close to the bandwidth of a plain copy; they are slower because they read and write the grid
many more times. The measured savings below are per step at 4.2M cells, with 33.6M cells in
parentheses.

The largest single changes, by time saved. Each was measured in the configuration it applies to,
so the savings are not additive.

| change | section | saved | applies to |
|---|---|---|---|
| fused momentum step | 3.1 | 8.6 ms (65 ms) per step | both |
| FMM kernel | 3.4 | 4.9 ms (17 ms) per step | Biot-Savart |
| multigrid kernels and compressed stencil | 3.2, 3.3 | 1.45 ms (14 ms) per V-cycle | both |
| `measure!` only in the body band | 1.5 | 1.5 ms per step | moving body |
| residual norms and flux fix from the boundary cells | 1.2 | 0.74 ms (5.0 ms) per step | Biot-Savart |
| residual correction folded into the next kernel | 1.4 | 0.68 ms (5.0 ms) per step | both |
| level-1 vorticity only where the FMM reads it | 1.1 | 0.66 ms (6.0 ms) per step | Biot-Savart |
| pressure solved in physical units | 1.3 | 0.49 ms (4.0 ms) per step | both |

### 1. Algorithmic changes (same result in exact arithmetic, less work)

WL 1.10 and BSBC 1.1 do this work; TurboLily skips it. Each change relies on a property of the
discrete operators. The property is listed so that a future change to those operators knows what
it could break.

#### 1.1 Level-1 vorticity only where the FMM reads it (Biot-Savart)

**BSBC.** The first Biot update of each projection calls `fill_ω!` (later ones refresh only the
body box, see 2):

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
evaluation. At 4.2M cells the two kernels take about 0.07 ms and 0.16 ms. Measured: 0.66 ms (6.0 ms) per
step.

#### 1.2 Residual norms and flux fix from the boundary cells only (Biot-Savart)

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

**Relies on:** L = 0 on the non-periodic domain faces. A periodic face adds with both signs,
like an interior face.

**Saves:** 3 full reductions and 2 host syncs per Biot update, 0.74 ms (5.0 ms) per step. The remaining reductions run
over about N^(2/3) cells.

#### 1.3 Pressure solved in physical units (both)

**WL.** `mom_project!` does `b.x .*= Δt` (a full pass), solves `A x = z`, applies `Δt·∇p`
through x, then `b.x ./= Δt` (another full pass). BSBC scales `p` the same way.

**TurboLily.** Solves `A p = z/Δt` directly. The residual tolerances are divided by Δt, so
`max|Δt·r| < tol` is the same test.

**Saves:** 2 full passes per projection, 0.49 ms (4.0 ms) per step with WaterLily BCs and
0.47 ms (3.8 ms) with Biot-Savart BCs.

#### 1.4 Residual correction folded into the next kernel (both)

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

**Saves:** 2 passes, 3 reductions and 3 host syncs per solve, 0.68 ms (5.0 ms) per step.

#### 1.5 `measure!` rewrites only the body band (both)

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
appends the BDIM face lists (see 3.1). The multigrid coefficients, regular-cell
flags (3.2) and the solved-row count are updated by fused kernels, with one host sync per
measure.

**Saves:** the D² + 2D value memsets and several kernels per measure. A static body
remeasured every step (the `sim_step!` default) costs about 1.5 ms at 4.2M cells. For a moving
sphere it saves 1.5 ms per step at 4.2M cells, 9 % of the time saved.

### 2. Algorithms shared with BiotSavartBCs 1.1 (same work, faster kernels)

TurboLily uses these algorithms of BSBC 1.1 (Biot-Savart only):

- **Cumulative pressure.** The multigrid continues on the total pressure p, and `u −= Δt·L∇p`
  is applied once at the end of the projection.
- **Vorticity in the body box.** After the first Biot update of a projection, the vorticity of
  `u − Δt·L∇p` is recomputed only in the body box: the bounding box of the cells with
  0 < μ₀ < 1, plus one cell, coarsened on every level. Outside it curl(L∇p) is zero.
- **FMM in the body box.** Later updates re-sum only the targets whose sources on their level
  overlap the body box; the other targets keep their sums from the first update.

Differences in TurboLily's implementation:

- `u − Δt·L∇p` is evaluated on the fly and never written, with p in physical units (1.3);
- level-1 vorticity is computed only in the layer that the FMM reads (1.1);
- a target is also re-summed when one of its symmetry images sees the body box.

**Cost.** At 4.2M cells (33.6M cells), a vorticity update takes 0.09 ms in BSBC and 0.11 ms in
TurboLily (0.43 and 0.44 ms). A later FMM update takes 0.74 and 0.12 ms (1.34 and 0.32 ms); the
difference is the FMM kernel (3.4).

### 3. Computational changes (same arithmetic, faster execution)

#### 3.1 Momentum step

**WL.**

- `u⁰ .= u` is a copy and `fill!(u, 0)` another pass; `conv_diff!` clears `r`.
- `conv_diff!` loops over all (component i, direction j) pairs. Per pair there are two
  kernels: the interior flux written to `σ` and added, and the same flux read back and
  subtracted from the neighbour. One kernel per direction handles the boundary faces. That is
  21 launches in 3D, with the flux stored and read back.
- `accelerate!` is another pass.
- `BDIM!` makes 2 passes over all faces, reading μ₀, μ₁ (D² values per cell) and V.

**TurboLily.**

- One kernel per phase. Each thread computes, for its cell and all D components, the fluxes
  through its 2D faces, the acceleration and the update `u⁰ + Δt·r`. Each face flux is evaluated
  by both adjacent cells instead of being stored: computing is cheaper than the store and load.
- In each flux the upwind stencil is selected first, so `λ` (QUICK) is evaluated once instead
  of once per branch. Both far-point candidates are loaded before the selection, so the load
  does not wait for the sign of the face velocity.
- The core kernel covers the full x range. Its x faces use a branch-free boundary flux, and the
  periodic flags and the grid strides are compile-time constants (type parameters), which keeps
  it at 64 registers. The y and z boundary layers run as one thin kernel.
- The core kernel is limited by its floating-point instructions, not by memory: about 440 per
  cell, 144 of them the minima and maxima of the QUICK limiter (RTX 2080 Ti).
- The full BDIM update (μ₀, μ₁, V) runs only on two precomputed lists:
  - faces with μ₀ ≠ 1 and μ₁ = 0, where faces fully inside the body reduce to u = V;
  - faces with μ₁ ≠ 0, the shell.

  Everywhere else BDIM is the identity. μ₁ = 0 wherever μ₀ = 1, because the kernel moment
  `kern₁(±1) = 0`. A shell face is handled by a group of 8 lanes, one per f̃ value it needs,
  combined by warp shuffles in a fixed order.
- `u⁰ ← u` and the corrector output are pointer swaps.

**Measured:** the momentum step takes 9.6 ms in WL and 1.0 ms in TurboLily (74 and 8.7 ms):
43–58 % of all the time saved.

#### 3.2 Compressed Poisson stencil

**WL.** Every multigrid operator (mult, Jacobi, Gauss-Seidel, increment) reads L (D values per
cell), D and iD in every cell.

**TurboLily.** A 1-byte flag per cell marks cells where all 2D faces have L equal to the level's
fluid value (1 on the fine level, a fixed constant on coarse levels). There D and iD are
constants and nothing is loaded. Flagged cells are most of the domain, everything not next to
the body or the domain boundary. This reduces the bytes per cell of every multigrid operator
from about 6–7 values to about 2–3. A second bit marks cells whose neighbours are flagged too,
so the Jacobi step and the first Gauss-Seidel sweep read no neighbour flags. All kernels load
their stencil values before branching on the flags, so the loads do not wait for the flag.

**Measured:** saves 1.9 ms (21 ms) per step. Without it the TurboLily step is 37 % (55 %) slower.

#### 3.3 Multigrid kernels

| WL | TurboLily |
|---|---|
| `Jacobi!`: ϵ = r·iD (pass), `perBC!`, increment (pass) | one kernel: reads r·iD of the neighbours on the fly; periodic wrap is in the index |
| `restrict!` of the residual after the Jacobi step (pass) | in the Jacobi kernel, by shared-memory block sums |
| `prolongate!` (pass), then `increment!` (pass) | one kernel: the piecewise-constant prolongation is read from the coarse level on the fly |
| `GaussSeidelRB!`: init ϵ = r·iD (pass), `perBC!`, 4 half sweeps | init fused into the first half sweep; half sweeps 3 and 4 in one pass, a wavefront in z |
| one launch per kernel on every level | levels of ≤ 2048 cells in one single-block kernel |
| `L₁` and `L∞` after each V-cycle: 2 reductions, 2 host syncs | computed inside the final increment kernel; 1 host read per V-cycle |
| adaptive ω updated on the host and passed to kernels | updated on the device by a 1-thread kernel |
| each V-cycle is about 70 launches with host overhead | the whole V-cycle + smoothing is replayed as one cached CUDA graph |

**Measured:** one V-cycle with its smoothing and norms takes 2.92 ms in WL and 1.47 ms in
TurboLily (24.5 and 10.2 ms), with the same iteration counts. CUDA graphs save 0.08 ms per step
from 4.2M cells up, and 30 % of TurboLily's step at 66k cells.

#### 3.4 Biot-Savart kernels

- **FMM interaction.** Each source x-line is split around the near-field hole, so the inner
  loop has no exclusion test, and the target's velocity component is a compile-time constant,
  so only the two vorticity components that enter are loaded. BiotSavartBCs tests every source
  with `S ∉ Rinner` instead. The split keeps the summation order, so the two differ only by
  rounding. TurboLily also computes 1/r³ with `rsqrt`. On coarse levels, where there are few
  targets with large source boxes, several threads (up to 16, chosen from measurements) share
  one target and reduce with warp shuffles.
- **Residual source and flux fix.** The residual kernel computes the source from `u` directly,
  so z is never stored. The boundary-velocity kernel also updates the residual next to the
  boundary and sums the net flux change. The boundary kernels read precomputed adjacent-cell
  indices, with no integer divisions.
- **`pflowBC!`.** One kernel per direction (tangential components, then normal), as in BSBC.

**Measured:** the FMM kernel saves 4.9 ms (17 ms) per step, 25 % of the time saved at 4.2M
cells.

#### 3.5 General GPU details

- Reductions write per-block partial sums, which are reduced in Float64 in two levels (Float64
  runs at 1/32 rate on GeForce cards). There are no atomics on a single address.
- The host reads the sums and maxima of a reduction in one copy, after a stream sync that
  blocks in the driver. CUDA.jl's default wait wakes the task from a worker thread, which leaves
  the GPU idle for 50–200 µs per read.
- The corrector projection also computes the CFL flux of the projected velocity, and the
  residual and projection kernels write the x faces of the domain BCs. The convective exit
  runs over the whole exit plane in three launches.
- All index arithmetic is Int32. Note that Julia's `sum` over Int32 widens to Int64.
  WaterLily's `@loop` gets each work-item's Cartesian index from KernelAbstractions with Int64
  divisions; on streaming kernels at 18.9M cells this costs about a third of the bandwidth
  (≈350 against ≈530 GB/s).
- Loops over the dimensions are fully unrolled, so tuples indexed by the dimension stay in
  registers.
- Kernels on a domain face run threads over the two in-plane directions, as in WaterLily.
- The BDIM band lists are built with warp-aggregated atomics instead of `findall`.

### 4. Where results are not identical to the reference

#### 4.1 Floating-point rounding (all cases)

Groups 1–3 change the order of floating-point operations:

- block-wise Float64 reductions;
- `rsqrt` and a different summation order in the FMM;
- pressure accumulated over the projection.

Velocities therefore agree with the reference to rounding, not bit for bit. The 3D flows
amplify these differences exactly as they amplify a 1e-6 perturbation of the reference itself
(see Validation). Consequences visible in the output:

- **Partially blocked cells.** Pressure in cells with some faces closed by the body is
  ill-conditioned and can differ by up to about 0.1. It differs by similar amounts between two
  runs of the reference. Cells with all faces open agree to about 1e-5.
- **Pressure gauge.** With Biot-Savart BCs the pressure can differ from BSBC by a uniform
  constant of about 1e-3. This does not affect the velocity or the pressure force on a closed
  body.
- **Σr drift.** 1.2 assumes Σr stays exactly zero through the multigrid. In Float32 it drifts
  slightly; BSBC recomputes it and TurboLily does not. The resulting difference in the flux
  fix is around 1e-9.

#### 4.2 Periodic directions use a slightly different smoother

**WL.** `GaussSeidelRB!` fills the periodic ghost cells once (`perBC!`), then runs 4 half
sweeps. Sweeps 2–4 therefore read stale ghost values on periodic boundaries.

**TurboLily.** On levels where every periodic length is even, neighbours are wrapped across
periodic boundaries in every half sweep, so they always read current values. On a level with
an odd periodic length, the first and last interior cells are neighbours across the wrap and
have the same red-black colour. A wrapped read would then race with the update of that
neighbour. These levels therefore use WaterLily's method: ghosts filled once, no wrap. Odd
levels occur also for even grids: a periodic 48 or 96 coarsens to a 3-cell level.

**Effect.** Results agree to the solver tolerance rather than to rounding. Against WaterLily
1.10:

| case | steps | max \|Δu\|, default tol | max \|Δu\|, tol=1e-6 | V-cycles per solve at tol=1e-6, WL / TurboLily |
|---|---|---|---|---|
| Taylor-Green vortex 32³, fully periodic | 10 | 1.9e-3 | 1.5e-5 | 5.3 / 4.3 |
| sphere 64×32×32, periodic in z, `exitBC` | 5 | 2.3e-4 | 3.8e-5 | 5.2 / 4.7 |

At the default tolerance both codes need the same number of V-cycles (1.05 and 1.6 per solve).

#### 4.3 `udf` callbacks

As in WaterLily, the predictor callback sees `flow.u` with a zero interior, and the corrector
callback sees the projected predictor velocity. The velocity to advect with is the third
argument, `udf(flow, u, t; kwargs...)`. TurboLily may leave different values than WaterLily
in the tangential ghost cells of `flow.u`.

#### 4.4 Not implemented

These raise an `ArgumentError`:

- `fmm=false` (the tree sum).
- Biot-Savart BCs on grids too small for a second Biot-Savart level, e.g. 8×8.
- Periodic directions with Biot-Savart BCs. BiotSavartBCs supports one periodic direction in
  3D; TurboLily's FMM has no periodic image sum.

Also not implemented: `T=Float64` and non-CUDA arrays.

Not supported: bodies wrapped in
[WaterLilyNarrowBand.jl](https://github.com/WaterLily-jl/WaterLilyNarrowBand.jl)'s
`NarrowBand` cache. TurboLily already measures only near the body (1.5), so the cache would
add little.

## Multi-GPU

A single simulation runs on one GPU. Measured on a workstation with 4× RTX 2080 Ti (PCIe, no
NVLink):

- Peer-to-peer access between the GeForce cards is disabled. GPU-to-GPU copies go through
  the host at about 1.7 GB/s, or 6 GB/s with pinned staging, with 25-100 µs latency.
  Exchanging halos inside every multigrid smoothing step would cost more than one whole
  V-cycle takes on one GPU (1.45 ms at 4.2M cells). So domain decomposition only pays off for
  problems that do not fit in one card's 11 GB, which is roughly 60M cells.
- Independent simulations, one per GPU from one Julia process, work as they are: see
  `examples/multigpu_sweep.jl`, started with `julia -t 4`.

## Tests

The tests need a CUDA GPU. `test/runtests.jl` steps every case with TurboLily on the GPU and
with WaterLily or BiotSavartBCs on the CPU, and requires agreement to rounding.
`test/waterlily_tests.jl` runs WaterLily's own simulation tests (Taylor-Green decay, body
forces, rotating frame, added mass, moving and deforming bodies, forces, `MeanFlow`) on
`TurboLily.Simulation`.

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

CI on GitHub Actions only checks that the dependencies resolve, on Julia 1.11 and 1.13. GitHub's
runners have no GPU, and WaterLily's CUDA extension does not load without one, so the tests
are run by hand.

The benchmark scripts are not part of this repository. [`RESULTS.md`](RESULTS.md) has the full
table and its method.

## License

MIT, see [`LICENSE`](LICENSE). Parts of TurboLily are derived from WaterLily.jl and
BiotSavartBCs.jl by Gabriel Weymouth, both MIT-licensed. TurboLily was written with
[Claude Code](https://claude.com/claude-code).
