# TurboLily

CUDA rewrite of the [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl) time step and of the
[BiotSavartBCs.jl](https://github.com/weymouth/BiotSavartBCs.jl) pressure solver. It keeps WaterLily's
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

## Speed

All results are from one RTX 2080 Ti, Float32, in ms per time step. The flow is a sphere at
Re=3700 (`D = N/2`, 256×128×128, 4.2M cells) unless noted. The comparison is against
WaterLily#337 + BiotSavartBCs#13+#14 on the same GPU. See `RESULTS.md` for the full tables,
including the earlier comparison against WaterLily master / BiotSavartBCs main.

| case | stack | TurboLily | speedup |
|---|---|---|---|
| WaterLily, static body | 18.9 | 6.0 | 3.2× |
| WaterLily, moving body (`remeasure=true`) | 22.8 | 8.4 | 2.7× |
| WaterLily, 512×256×256 (33.6M cells) | 143.0 | 41.4 | 3.5× |
| WaterLily, TGV 256³ | 74.0 | 20.1 | 3.7× |
| Biot-Savart, static body | 26.7 | 8.7 | 3.1× |
| Biot-Savart, moving body | 29.8 | 10.8 | 2.8× |
| Biot-Savart, 480×192×192 (17.7M cells) | 98.9 | 30.8 | 3.2× |

The speedup grows with problem size, and it is largest for Biot-Savart BCs. The long runs
below show 4.8× and 5.4× over a developed flow.

## Validation

Every configuration is stepped side by side with the reference implementation, and the results
must agree to rounding: `test/runtests.jl` contains 52 checks. These cover 2D and 3D, `exitBC`,
periodic directions (including a fully periodic Taylor-Green vortex), `uBC(i,x,t)`, `g`, a moving body with `remeasure=true`, `udf` (body force
and the Smagorinsky `sgs!`), Biot-Savart with `nonbiotfaces`, and the BiotSavartBCs
potential-flow test. The pressure-iteration counts are identical.

The velocity differences are of the same size as those between two runs of the reference code
when one run gets a 1e-6 perturbation. The 3D flows amplify rounding differences in both codes
in the same way.

Long runs of a sphere at Re=3700, 192×96×96, t = 0…20:

| | wall time | steps | mean C_d (t>10) |
|---|---|---|---|
| BiotSavartBCs | 35.9 s | 1160 | 0.3867 |
| TurboLily.BiotSimulation | 6.7 s (5.4×) | 1165 | 0.3865 |
| WaterLily | 13.0 s | 852 | 0.4140 |
| TurboLily.Simulation | 2.7 s (4.8×) | 852 | 0.4142 |

## What changed relative to the reference

The discretisation and the solvers are WaterLily's and BiotSavartBCs': QUICK/CDS convection,
BDIM, the predictor-corrector step, the multigrid V-cycle with Jacobi pre-smoothing, red-black
Gauss-Seidel post-smoothing and adaptive ω, the stopping criteria (BiotSavartBCs#13), and the
FMℓM Biot-Savart sum. The changes fall into three groups.

### 1. Algorithmic changes (same result in exact arithmetic, less work)

These skip work that provably cannot change the result. Each relies on a property of the
discrete operators, and the property must be preserved by future changes to those operators.

- **Vorticity only near the body (Biot-Savart).** During a projection `u = u* − Δt·L∇p`. The
  centred curl of a discrete gradient is identically zero, so ω changes only in cells whose
  curl stencil reads a face with L ≠ 1, that is, next to the body. Only those cells, and their
  level-2 parents, are recomputed after each pressure update.
  *Relies on:* this exact curl stencil and velocity update; μ₀ being exactly 1 away from the
  body; interior `u` not changing between Biot updates.
  `bench_stack/check_vort.jl` compares this against the direct computation.
- **Incremental FMM.** The Biot-Savart sum is linear in ω. After the first full evaluation in
  a projection, later updates sum only Δω, clipped to the bounding box of the changed cells on
  each level.
- **Level-1 vorticity only in the boundary layer.** The level-1 interactions read sources at
  most 6 cells (3D) or 10 cells (2D) from the boundary, and level 2 is summed directly from the
  curl. *Relies on:* the layer width matching the FMM near-field radius (`close`), with one
  cell of margin.
- **Cumulative pressure, one projection at the end.** The multigrid continues on the total p
  instead of resetting x = 0 each Biot iteration, and `u` is projected once at the end. The
  residual `z − A·p` is the same in both forms.
- **Boundary-only residual updates (Biot-Savart).** A Biot update changes the residual only in
  the cells next to the boundary. So Σ|r| and max|r| are updated from those cells: the
  multigrid reports the max over all other cells. Σr is invariant under multigrid (Σ(Aϵ) = 0 by
  telescoping, with L = 0 on the domain faces), so the global flux fix needs only the net
  boundary flux change.
- **Pressure solved in physical units.** `A p = z/Δt` with the tolerances scaled by Δt,
  instead of solving for `x = Δt·p`. This removes two scaling passes per projection.
- **Lazy residual correction.** WaterLily's `residual!` subtracts the mean residual from the
  solved rows. TurboLily applies that correction inside the first Jacobi kernel, instead of
  in a separate pass.
- **`measure!` touches only the body band.** Cells far from the body keep μ₀ ∈ {0,1}, μ₁ = 0
  and V = 0, so only cells that are in the band now or were in it at the last measure are
  rewritten. WaterLily resets μ₀, μ₁ and V everywhere. *Relies on:* the assumption listed
  in group 3.

### 2. Computational changes (same arithmetic, faster execution)

- **Fused kernels.**
  - Momentum: one kernel per phase computes conv-diff, acceleration and the update for all
    components. BDIM runs only on precomputed lists of band faces, and faces inside the body
    reduce to `u = V`.
  - Multigrid: Jacobi in one kernel; prolongation fused with the increment; the first
    Gauss-Seidel half sweep fused with ϵ = r·iD; norms computed inside the increment.
  - Biot-Savart: residual and source in one kernel; vorticity curl and restriction in one.
- **Compressed Poisson stencil.** A 1-byte flag marks cells whose faces all have L = 1, so
  L, D and iD are not read from memory there. This cuts the traffic of every multigrid operator.
- **Branch-free fast paths.**
  - Convective fluxes: the upwind stencil is selected before `λ`, so `λ` is evaluated once
    per face. The general boundary-face code runs only on the domain-boundary faces.
  - FMM: x-lines are split around the near-field hole, with a compile-time component
    (from BiotSavartBCs#14).
  - BDIM: compile-time component dispatch.
- **Fewer copies and host synchronisations.**
  - `u⁰ ← u` is a pointer swap.
  - Reductions use per-block partials with no atomics, and are read by the host once per
    V-cycle. The adaptive ω and the solver state live on the device.
  - Each multigrid iteration is replayed as a CUDA graph.
  - Band lists are built with warp-aggregated atomics instead of `findall`.
- **GPU-specific details.** Int32 index arithmetic; loops over dimensions fully unrolled so
  no tuple ends up in local memory; face kernels laid out over the two in-plane directions
  (the same issue as WaterLily#337); FMM lanes split over sources on coarse levels with few
  targets.

### 3. Where results are not identical to the reference

- **Floating-point rounding.** Groups 1 and 2 change the order of operations: reductions use
  Float64 block sums, the FMM uses `rsqrt` and a different summation order, and vorticity
  and pressure are accumulated incrementally. Velocities therefore agree with the reference
  to rounding, not bit for bit. The 3D flows amplify these differences exactly as they
  amplify a 1e-6 perturbation of the reference itself.
  - The pressure in cells partly blocked by the body is ill-conditioned and can differ by
    up to about 0.1 (it also differs between two runs of the reference).
  - With Biot-Savart BCs the pressure can differ by a uniform constant of about 1e-3 (the
    gauge). This does not change the velocity or the pressure force on a closed body.
  - The incremental updates assume Σr stays zero through the multigrid instead of
    recomputing it as BiotSavartBCs does; the difference in the flux fix is around 1e-9.
- **Periodic directions use a slightly different smoother.** TurboLily's Gauss-Seidel wraps
  periodic neighbours in every half sweep. WaterLily refreshes the periodic ghosts only
  between smoother calls, so its sweeps read stale ghost values. Results agree to the
  solver tolerance (about 2e-4) rather than to rounding. With a tight tolerance in both codes
  they agree to 1e-6, and TurboLily needs one V-cycle fewer.
- **Body motion assumption in `measure!`.** A cell is rewritten only if it is in the body
  band (|d| < 2+ϵ) now or was at the previous measure. If a body moves more than about
  3 cells between two measures (a jump, not CFL-limited motion), a cell could go from deep
  inside to far outside without being rewritten, and would keep μ₀ = 0. WaterLily has no
  such assumption.
- **WaterLily version semantics.** TurboLily follows WaterLily master's `residual!` and
  #328's deforming-body source term. WaterLily#337 is based on an older master without them.
  For static bodies the source term is zero and the test suite agrees with both.
- **Not implemented:** `fmm=false` (the tree sum), and overriding `BiotSavartBCs.symmetry`
  for image vortices. Float32 and CUDA only.

## Multi-GPU

A single simulation runs on one GPU. Measured on this machine (4× RTX 2080 Ti, PCIe, no
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

## Setup and tests

The environment in `../bench` (`Project.toml`) develops `../dev/WL`, `../dev/BSBC` and this
package together with CUDA. All benchmark and validation scripts live there.

```
cd ../bench
julia --project=. ../TurboLily/test/runtests.jl     # 52 equivalence tests, about 4 min
julia --project=. bench.jl                          # WHICH=WL,T,B,TB  N=128
```
