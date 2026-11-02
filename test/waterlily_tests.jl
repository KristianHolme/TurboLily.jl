# WaterLily's own simulation-level tests (test_flow.jl, test_simulation.jl, test_float32.jl,
# test_metrics.jl of WaterLily 1.11), run with TurboLily.Simulation. They check the flow against
# analytic results, not against a reference run. Float32 throughout; where WaterLily's test
# uses Float64 the tolerance is noted.
using Test, TurboLily, WaterLily, CUDA, StaticArrays

const TS = TurboLily.Simulation
L₂(a) = WaterLily.L₂(a)

@testset "WaterLily tests on TurboLily" begin
    @testset "periodic Taylor-Green vortex decays as the exact solution" begin
        L = 64; κ = Float32(2π/L); ν = 1/(κ*1f8)
        TGV(i, xy, t) = (x = xy[1]*κ; y = xy[2]*κ; d = exp(-2κ^2*ν*t);
            i == 1 ? -sin(x)*cos(y)*d : cos(x)*sin(y)*d)
        sim = TS((L,L), TGV, L; U=1, ν, T=Float32, perdir=(1,2))
        sim_step!(sim, π/100)
        # capture the time, not `sim`: KernelAbstractions 0.10 compiles CPU kernels and needs isbits closures
        t = WaterLily.time(sim)
        ue = zeros(Float32, size(sim.flow.u)); apply!((i,x)->TGV(i, x, t), ue)
        u = Array(sim.flow.u)
        @test L₂(u[:,:,1] .- ue[:,:,1]) < 1e-4
        @test L₂(u[:,:,2] .- ue[:,:,2]) < 1e-4
    end

    @testset "increasing body force: g and udf" begin
        N = 8; jerk = 4; U = Float32(√N)
        mk(g) = TS((N,N), (U,0), N; ν=0.001, g, Δt=0.001, perdir=(1,), T=Float32)
        exact(sim) = U + 0.5f0*jerk*WaterLily.time(sim)^2
        sim = mk((i,x,t) -> i == 1 ? t*jerk : 0f0)
        sim_step!(sim, 1.0); u = Array(sim.flow.u)
        @test L₂(u[:,:,1] .- exact(sim)) < 1e-4 && L₂(u[:,:,2]) < 1e-4
        gravity!(flow, t; jerk=4) = for i in 1:last(size(flow.f))
            WaterLily.@loop flow.f[I,i] += i == 1 ? t*jerk : 0 over I ∈ CartesianIndices(Base.front(size(flow.f)))
        end
        sim = mk(nothing)
        sim_step!(sim, 1.0; udf=gravity!, jerk); u = Array(sim.flow.u)
        @test L₂(u[:,:,1] .- exact(sim)) < 1e-4 && L₂(u[:,:,2]) < 1e-4
    end

    @testset "boundary-layer profile is maintained" begin
        L = 32
        sim = TS((L,L), (i,x,t) -> i == 1 ? 4f0*(((x[2]+0.5f0)/2L) - ((x[2]+0.5f0)/2L)^2) : 0f0, L;
            ν=0.001, U=1, T=Float32)
        sim_step!(sim, 10); u = Array(sim.flow.u)
        @test all(u[1,:,1] .≈ u[end,:,1])
    end

    @testset "rotating reference frame: g and udf agree" begin
        N, x₀, ω = 8, SA_F32[4,4], 1f0/4
        function velocity(i, x, t)
            s, c = sincos(ω*t); y = ω*(x - x₀)
            i == 1 ? s*y[1] + c*y[2] : -c*y[1] + s*y[2]
        end
        g(i, x, t) = (i == 1 ? 2ω*velocity(2, x, t) : -2ω*velocity(1, x, t)) + ω^2*(x - x₀)[i]
        udf(a::AbstractFlow, t) = WaterLily.@loop a.f[Ii] += g(last(Ii), loc(Ii, eltype(a.f)), t) over Ii in CartesianIndices(a.f)
        simg = TS((N,N), velocity, N; g, U=1, T=Float32); sim_step!(simg)
        sim = TS((N,N), velocity, N; U=1, T=Float32); sim_step!(sim; udf)
        # WaterLily (Float64): equal and < 3e-3
        @test L₂(Array(simg.flow.p)) ≈ L₂(Array(sim.flow.p)) rtol=1e-4
        @test L₂(Array(sim.flow.p)) < 3e-3
    end

    @testset "circle in accelerating flow: added mass" begin
        R, H = 32, 16
        sim = TS(R .* (2H,2H), (i,x,t) -> i == 1 ? t : zero(t), R; U=1, T=Float32,
            body=AutoBody((x,t) -> √sum(abs2, x .- H*R) - R))
        sim_step!(sim)
        @test isapprox(WaterLily.pressure_force(sim)/(π*sim.L^2), [-1,0], atol=0.04)
        u = Array(sim.flow.u)
        @test maximum(u)/u[2,2,1] > 1.91   # ≈ 2U
        foreach(_ -> sim_step!(sim), 1:3)
        @test all(sim.pois.n .≤ 2)
    end

    @testset "anti-phase deforming pair: flux between the bodies" begin
        R, ε, Tc = 8f0, 0.25f0, 64f0
        s(t) = 1 - ε*(1 - cospi(2t/Tc)); ṡ(t) = -2ε*Float32(π)/Tc*sinpi(2t/Tc)
        circ(c, s) = AutoBody((x,t) -> √(x'*x) - R, (x,t) -> (x .- c)/s(t))
        sim = TS((128,64), (0,0), 2R; ν=2R/250, T=Float32,
            body=circ(SA_F32[32,32], s) + circ(SA_F32[96,32], t -> √(2 - s(t)^2)))
        foreach(_ -> sim_step!(sim), 1:8); t = WaterLily.time(sim.flow)
        @test sum(Array(sim.flow.u)[66,2:65,1]) ≈ 2Float32(π)*R^2*s(t)*ṡ(t) rtol=0.01
    end

    @testset "moving and deforming bodies (test_simulation.jl)" begin
        r = 8; ν = r/250f0; nm = r .* (4,4)
        circle(x, t) = √sum(abs2, x .- 2r) - r
        plate(x, t) = √sum(abs2, x - SA[clamp(x[1], -r+2, r-2), 0]) - 2
        function rotate(x, t)
            s, c = sincos(t/r + 1); R = SA[c s; -s c]
            R*(x .- 2r)
        end
        function bend(xy, t)
            x, y = xy .- 2r; κ = 2t/r^2 + 0.2f0/r
            SA[x + x^3*κ^2/6, y - x^2*κ/2]
        end
        sim = TS(nm, (1,0), r; body=AutoBody(circle), ν, T=Float32)
        @test sim_time(sim) == 0
        sim_step!(sim, 0.1, remeasure=false)
        @test sim_time(sim) ≥ 0.1 > sum(sim.flow.Δt[1:end-2])*sim.U/sim.L
        # poisson_kwargs reach both projections and not udf; the impulsive start needs more than one V-cycle
        sim = TS(nm, (1,0), r; body=AutoBody(circle), ν, T=Float32)
        sim_step!(sim; udf=(a,t)->nothing, poisson_kwargs=(; itmx=1))
        @test sim.pois.n == [1, 1]
        for exitBC in (true, false)
            # remeasure is exact when V = U = 1
            sim = TS(nm, (1,0), r; body=AutoBody(circle, (x,t) -> x - SA[t,0]), ν, T=Float32, exitBC)
            sim_step!(sim)
            @test all(Array(sim.flow.u)[:,r,1] .≈ 1)
            # accelerating from U=0 to U=1
            sim = TS(nm, (0,0), r; U=1, body=AutoBody(circle, (x,t) -> x - SA[2t^2,0]), ν, T=Float32, exitBC)
            sim_step!(sim)
            @test length(sim.pois.n) == 2 && all(sim.pois.n .< 5)
            @test maximum(Array(sim.flow.u)) > maximum(Array(sim.flow.V)) > 0
            # non-uniform V
            sim = TS(nm, (0,0), r; U=1, body=AutoBody(plate, rotate), ν, T=Float32, exitBC)
            sim_step!(sim)
            @test length(sim.pois.n) == 2 && all(sim.pois.n .< 5)
            @test 1 > sim.flow.Δt[end] > 0.5
            # divergent V
            sim = TS(nm, (0,0), r; U=1, body=AutoBody(plate, bend), ν, T=Float32, exitBC)
            sim_step!(sim)
            @test length(sim.pois.n) == 2 && all(sim.pois.n .< 5)
            @test 1.2 > sim.flow.Δt[end] > 0.8
            # mixed-type uBC and tuple u0 are converted to T
            sim = TS(nm, (1.0,0), r; u0=(1,0.5), T=Float32, exitBC)
            @test sim.flow.uBC === (1f0,0f0)
            @test all(Array(sim.flow.u)[3:end-2,3:end-2,2] .== 0.5f0)
        end
    end

    @testset "forces and moments match WaterLily on the CPU (test_float32.jl)" begin
        N = 64
        mk(ctor; mem=Array) = (L = N/4f0; c = N/2f0;
            ctor((N,N), (1f0,0f0), L; ν=L/200, T=Float32, mem, exitBC=true,
                body=AutoBody((x,t) -> √sum(abs2, x .- c) - L/2)))
        ref = mk(WaterLily.Simulation); sim = mk(TS)
        sim_step!(ref, 1f0; remeasure=false); sim_step!(sim, 1f0; remeasure=false)
        x₀ = SA_F32[N/2, N/2]
        @test sim.flow.Δt ≈ ref.flow.Δt
        @test maximum(abs, Array(sim.flow.u) .- ref.flow.u) < 1f-4
        Fp = WaterLily.pressure_force(sim); Fv = WaterLily.viscous_force(sim)
        @test Fp isa Vector{Float32} && Fv isa Vector{Float32}
        @test Fp ≈ WaterLily.pressure_force(ref) rtol=1e-4
        @test Fv ≈ WaterLily.viscous_force(ref) rtol=1e-4
        @test WaterLily.total_force(sim) ≈ WaterLily.total_force(ref) rtol=1e-4
        @test WaterLily.total_moment(x₀, sim) ≈ WaterLily.total_moment(x₀, ref) atol=1e-5
    end

    @testset "MeanFlow statistics (test_metrics.jl)" begin
        L = 32; T = Float32
        sim = TS((L,L), (i,x,t) -> i == 1 ? 4f0*(((x[2]+0.5f0)/2L) - ((x[2]+0.5f0)/2L)^2) : 0f0, L;
            ν=0.001, U=1, T)
        mf = MeanFlow(sim.flow; uu_stats=true)
        for t in range(0, 10; step=0.1)
            sim_step!(sim, t); WaterLily.update!(mf, sim.flow)
        end
        u, U = Array(sim.flow.u), Array(mf.U)
        @test all(isapprox.(u, U; atol=√eps(T)))
        @test all(isapprox.(Array(sim.flow.p), Array(mf.P); atol=√eps(T)))
        @test all(isapprox.(u[:,:,1] .* u[:,:,2], Array(mf.UU)[:,:,1,2]; atol=√eps(T)))
        @test WaterLily.time(sim.flow) == WaterLily.time(mf)
    end
end
