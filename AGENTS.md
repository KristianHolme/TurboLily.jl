# TurboLily

CUDA-only, Float32 rewrite of the WaterLily time step and the BiotSavartBCs projection, behind WaterLily's API (`TurboLily.Simulation`, `TurboLily.BiotSimulation`). See `README.md` for usage and speedups.

## Layout

- `src/`: `flow.jl` (momentum step), `poisson.jl` (multigrid), `biot.jl` (Biot-Savart FMM), `measure.jl`, `util.jl` (reductions, index helpers).
- `test/runtests.jl`: steps every case side by side with WaterLily / BiotSavartBCs on CPU and requires agreement to rounding level. Needs a GPU.
- `test/waterlily_tests.jl`: WaterLily's simulation-level tests (analytic checks) on `TurboLily.Simulation`.
- `.github/workflows/CI.yml`: dependency resolution only; hosted runners have no GPU.
- `examples/`: usage examples.
- `RESULTS.md`: benchmark tables and their method.

## Kernel lessons (CUDA.jl 6.4, Julia 1.13, RTX 2080 Ti)

- A runtime `Bool` that gates a shared-memory block reduction made the kernel 30× slower. Pass such flags as `Val`.
- `sum` over an `NTuple{Int32}` widens to Int64, so index arithmetic silently becomes 64-bit. Write the linear-index sum out explicitly.
- `ntuple(Val(D)) do …` closures that capture structs, and closures that capture a reassigned (boxed) variable, give allocas, local memory or InvalidIR. Write the code out explicitly.
- Loops over `1:D` with tuple indexing need full unroll loopinfo (`Expr(:loopinfo, (Symbol("llvm.loop.unroll.full"), 1))`); otherwise the tuples go to local memory.
- Branchy general stencil code in a hot kernel raises register use and costs 2.5×. Use a branch-free interior fast path and the general path only at boundaries.
- One thread per cell beat both a z-marching shared-memory kernel and naive temporal blocking of red-black GS: L1 captures the reuse, and occupancy matters more.
- Capture CUDA graphs on a private non-blocking stream in `STREAM_CAPTURE_MODE_RELAXED`, with GC disabled and CUDA.jl's compile lock held (`graphed!` in `poisson.jl`). Then other tasks that use the GPU cannot touch or depend on the capturing stream, and no finalizer or module load is captured.
- GeForce cards have no P2P, so multi-GPU domain decomposition does not pay off.
- Loads that follow a branch on a loaded flag wait for that flag. Load the stencil values before branching on cell flags (`Pre` in `poisson.jl`): 6–13% on the multigrid kernels.
- Above 64 registers a 256-thread kernel drops from 4 to 3 blocks per SM. Runtime `Bool` tuples (periodic flags) inflate registers; as type parameters (`Per`) the code compiles away. A `maxregs` cap instead spills.
- FP64 runs at 1/32 rate on GeForce: a single block summing 73k partials in Float64 took 66 µs. Reduce many partials in two levels (`finish!`).
- `copyto!` from a `CuArray` to an `Array` waits cooperatively (a worker thread wakes the task): 50–200 µs of GPU idle per small read. On the critical path, block in the driver (`hostcopy!`).
- Strided writes to x faces (BC!) are DRAM-sector bound (partial sectors, scattered rows); more threads do not help. Do them in a kernel that already holds those lines.
- Latency-bound per-item work (BDIM shell faces): give each item a group of lanes running the same code with data-dependent offsets, and combine by shuffles in a fixed order, so the result stays bitwise equal. Not for code with local memory (a body's `measure`, ~400 B/thread): more threads in flight thrash the cache (4× slower).
- Multigrid levels of ≤ 2k cells run faster as one single-block kernel with barriers (`k_bottom!`); on 16³ levels one block is slower than separate launches.
- Two red-black half sweeps fused in a z wavefront: the second pair is 15–25% faster than two launches, the first (init) pair is not faster.
- The momentum core is FP32-pipe bound: ~440 FP instructions per cell, 144 of them the `quick` limiter's min/max. Integer address math runs alongside: Int64 indices (constant offsets folded into the loads) cut it from 140 to 29 instructions and registers from 61 to 45, with no speedup. Computing each face flux once, even only in x by shuffles, was no faster either.
- Block shape sets how often the stencil's halo planes come from L2. With 32×4×2 blocks a ±2-cell stencil reads each z plane three times, and on planes of 65k cells or more (`nx·ny`) the reloads fall out of the 6 MB L2 of an A10 or 2080 Ti: the momentum core takes 0.120 instead of 0.092 ns/cell. `k_momc!` uses 32×1×8 blocks (2.26 → 1.68 ms at 512×192×192 on an A10, bit-identical), the streaming multigrid and projection kernels 32×8×1, both only for boxes with `BIG_PLANE` cells per x-y plane (`kernel_threads`); smaller boxes keep 32×4×2, and so do the Jacobi kernels (`FusedR`). A GH200 (50 MB L2) shows no difference. Block order matters too: blockIdx.x runs fastest, so the slowest block dimension should carry the deep tile. On an A10, replacing `quick` by a 3-operation λ with the same loads left the core at 2.25 ms with the old shape, so there the halo traffic, not the limiter, set the time.
- A block shape has to fit every architecture. `k_bottom!` needs 196 registers per thread on sm_90, so 512 threads failed on a GH200. `launch` checks blocks above `SAFE_THREADS` (256 threads always fit) against `CUDA.maxthreads` and halves `shrinkable` kernels' shape until it fits, otherwise it raises an error; `k_bottom!` uses `fit_threads` the same way. Querying the limit on every launch costs 1–3 % of a small step, so keep it off the 256-thread path.

After every kernel change, run `test/runtests.jl` against the reference codes.
