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
Re=3700 (`D = N/2`, 256×128×128, 4.2M cells) unless noted. The comparison is against WaterLily
`main` and BiotSavartBCs `master` on the same GPU. See `RESULTS.md` for the full tables.

| case | reference | TurboLily | speedup |
|---|---|---|---|
| WaterLily, static body | 19.3 | 6.1 | 3.2× |
| WaterLily, moving body (`remeasure=true`) | 23.0 | 8.2 | 2.8× |
| WaterLily, 512×256×256 (33.6M cells) | 147.6 | 41.0 | 3.6× |
| Biot-Savart, static body | 53.9 | 10.8 | 5.0× |
| Biot-Savart, moving body | 66.8 | 14.4 | 4.6× |
| Biot-Savart, 480×192×192 (17.7M cells) | 179.0 | 34.1 | 5.2× |

The speedup grows with problem size, and it is largest for Biot-Savart BCs. The long runs
below show 4.8× and 5.4× over a developed flow.

## Validation

Every configuration is stepped side by side with the reference implementation, and the results
must agree to rounding: `test/runtests.jl` contains 49 checks. These cover 2D and 3D, `exitBC`,
periodic directions, `uBC(i,x,t)`, `g`, a moving body with `remeasure=true`, `udf` (body force
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

## How it is faster

The discretisation and the solvers are WaterLily's (QUICK/CDS convection, BDIM, the
predictor-corrector step, the multigrid V-cycle with Jacobi pre-smoothing, the red-black
Gauss-Seidel post-smoother and adaptive ω, the tolerances, and BiotSavartBCs' FMℓM). The
changes are in how the step runs on the GPU:

- **Momentum.** One kernel per phase computes conv-diff, acceleration and the BDIM
  update for all components. Each convective flux evaluates its upwind stencil once
  (selected before `λ`) through a branch-free interior path. BDIM runs only on precomputed
  lists of the body-band faces. `u⁰ ← u` is a pointer swap, not a copy.
- **Pressure.**
  - Cells where all faces have L=1 are flagged in a 1-byte array. The stencil there is
    implied, which cuts the memory traffic of every multigrid operator.
  - Fused kernels: Jacobi; prolongation and increment; residual with the global correction
    applied lazily; norms inside the increment. The pressure is solved for directly in
    physical units, so there are no `x .*= Δt` passes.
  - One host synchronisation per V-cycle. ω and the solver state live on the device, and
    each V-cycle is a cached CUDA graph.
- **Biot-Savart BCs.**
  - Within one projection the pressure is cumulative and `u` is projected only once at the
    end.
  - The vorticity of the projected field is updated only where `curl(L∇p) ≠ 0`. The centred
    curl of a discrete gradient is exactly zero wherever L=1, so this is only next to the body.
  - After the first FMM of each projection, the FMM is evaluated only on Δω, clipped to the
    body's bounding box on each level.
  - The global flux fix and Σr² are updated from the boundary cells, because Σr is invariant
    under the multigrid iterations.
  - Level-2 vorticity is summed straight from the curl of `u`. Level-1 vorticity is computed
    only in the boundary layer that the level-1 interactions read.
- **measure!.** One fused kernel measures the sdf everywhere but only rewrites cells that are,
  or were, in the body band. Band lists are built with warp-aggregated atomics, and the
  Poisson coefficients are updated on the device. A static body remeasured every step (the
  `sim_step!` default) costs about 1.5 ms at 4.2M cells.

## Differences from the reference

- Periodic directions: the Gauss-Seidel sweeps wrap periodic neighbours in every half sweep.
  WaterLily refreshes the periodic ghost cells only between smoother calls. The results agree
  to the solver tolerance, not to rounding.
- `fmm=false` (the tree sum) is not implemented, and overriding `BiotSavartBCs.symmetry` has
  no effect.
- Float32 and CUDA only.

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

## Tests

```
julia --project=<env with WaterLily, BiotSavartBCs, CUDA, TurboLily> test/runtests.jl
```
