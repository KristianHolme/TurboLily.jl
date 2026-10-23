# TurboLily

CUDA-only, Float32 rewrite of the WaterLily time step and the BiotSavartBCs projection, behind WaterLily's API (`TurboLily.Simulation`, `TurboLily.BiotSimulation`). See `README.md` for usage and speedups.

## Layout

- `src/`: `flow.jl` (momentum step), `poisson.jl` (multigrid), `biot.jl` (Biot-Savart FMM), `measure.jl`, `util.jl` (reductions, index helpers).
- `test/runtests.jl`: steps every case side by side with WaterLily / BiotSavartBCs on CPU and requires agreement to rounding level. Needs a GPU.
- `review/`: correctness review (`REPORT.md`) and its probe scripts. Its defects are fixed in 566b44c and fd27ed5.
- `reports/`: design notes, e.g. `kernelabstractions-port.md` (what ports to KA, and why), `narrowband-support.md` (narrowband is deliberately not supported).
- `papers/speedup-attribution.typ`: where the speedup comes from (algorithmic vs kernel work).
- `_research/`: benchmark and profiling scripts. Its `Project.toml` develops `dev/WaterLily.jl` and `dev/BiotSavartBCs.jl`. The `env_*` folders pin the comparison stacks (master, v1.10, PR branches).
- `dev/` (git-ignored): upstream clones. `WaterLily.jl` is on `perf/turbolily-kernels` and `BiotSavartBCs-perf` on `perf/turbolily-biot` (KA ports of TurboLily ideas, not for PRs as-is; upstream prefers small PRs). `dev/eco/` holds the WaterLily ecosystem packages used for compatibility checks.

## Kernel lessons (CUDA.jl 6.4, Julia 1.13, measured on RTX 2080 Ti)

- A runtime `Bool` that gates a shared-memory block reduction made the kernel 30× slower. Pass such flags as `Val`.
- `sum` over an `NTuple{Int32}` widens to Int64, so index arithmetic silently becomes 64-bit. Write the linear-index sum out explicitly.
- `ntuple(Val(D)) do …` closures that capture structs, and closures that capture a reassigned (boxed) variable, give allocas, local memory or InvalidIR. Write the code out explicitly.
- Loops over `1:D` with tuple indexing need full unroll loopinfo (`Expr(:loopinfo, (Symbol("llvm.loop.unroll.full"), 1))`); otherwise the tuples go to local memory.
- Branchy general stencil code in a hot kernel raises register use and costs 2.5×. Use a branch-free interior fast path and the general path only at boundaries.
- One thread per cell beat both a z-marching shared-memory kernel and naive temporal blocking of red-black GS: L1 captures the reuse, and occupancy matters more.
- CUDA graph capture must use `STREAM_CAPTURE_MODE_THREAD_LOCAL` when other threads use the GPU.
- GeForce cards here have no P2P, so multi-GPU domain decomposition does not pay off.
- Loads that follow a branch on a loaded flag wait for that flag. Load the stencil values before branching on cell flags (`Pre` in `poisson.jl`): 6–13% on the multigrid kernels.
- Above 64 registers a 256-thread kernel drops from 4 to 3 blocks per SM. Runtime `Bool` tuples (periodic flags) inflate registers; as type parameters (`Per`) the code compiles away. A `maxregs` cap instead spills.
- FP64 runs at 1/32 rate on GeForce: a single block summing 73k partials in Float64 took 66 µs. Reduce many partials in two levels (`finish!`).
- `copyto!` from a `CuArray` to an `Array` waits cooperatively (a worker thread wakes the task): 50–200 µs of GPU idle per small read. On the critical path, block in the driver (`hostcopy!`).
- Strided writes to x faces (BC!) are DRAM-sector bound (partial sectors, scattered rows); more threads do not help. Do them in a kernel that already holds those lines.
- Latency-bound per-item work (vorticity corrections, BDIM shell faces): give each item a group of lanes running the same code with data-dependent offsets, and combine by shuffles in a fixed order, so the result stays bitwise equal. Not for code with local memory (a body's `measure`, ~400 B/thread): more threads in flight thrash the cache (4× slower).
- Multigrid levels of ≤ 2k cells run faster as one single-block kernel with barriers (`k_bottom!`); on 16³ levels one block is slower than separate launches.
- Two red-black half sweeps fused in a z wavefront: the second pair is 15–25% faster than two launches, the first (init) pair is not faster.
- The momentum core is issue-bound (~700 SASS instructions per cell, half of them limiter min/max): computing each face flux once, even only in x by shuffles, was no faster.

After every kernel change, run `test/runtests.jl` against the reference codes.
