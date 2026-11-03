# WaterLily 1.11.0's own test files (test/waterlily/), run on TurboLily. Where the tests loop over
# `arrays`, the array type is CuArray; `mem=Array` simulations (the tests' CPU references) stay WaterLily.
# The copies differ from upstream only where TurboLily cannot do what the test does (each change is
# commented in place): Float32 instead of Float64, `@test_broken` for what is not supported.
# `test_alloc.jl` is not run: it needs WaterLily's SIMD backend.
using Test, StaticArrays, CUDA
import TurboLily, WaterLily

const WL_TESTDIR = joinpath(@__DIR__, "waterlily")
const WL_NAMES = [n for n in names(WaterLily) if n ∉ (:Simulation, :Flow, :WaterLily, :Bodies)]

@testset "WaterLily 1.11 test suite on TurboLily" begin
    for f in sort(filter(f -> startswith(f, "test_") && endswith(f, ".jl") && f != "test_alloc.jl", readdir(WL_TESTDIR)))
        M = Module(Symbol("WL_", f[1:end-3]), true)    # one module per file, as WaterLily's runner does
        Core.eval(M, quote
            using Test, StaticArrays, CUDA
            import TurboLily, WaterLily, GPUArrays
            $(Expr(:using, Expr(:(:), Expr(:., :WaterLily), [Expr(:., n) for n in WL_NAMES]...)))
            Simulation(args...; mem=Array, kw...) = mem === Array ? WaterLily.Simulation(args...; mem, kw...) :
                TurboLily.Simulation(args...; mem, kw...)
            Flow(args...; mem=Array, kw...) = mem === Array ? WaterLily.Flow(args...; mem, kw...) :
                TurboLily.TFlow(args...; mem, kw...)
            arrays = [CuArray]
        end)
        Base.include(M, joinpath(WL_TESTDIR, "helper.jl"))
        @testset "$f" begin
            Base.include(M, joinpath(WL_TESTDIR, f))
        end
    end
end
