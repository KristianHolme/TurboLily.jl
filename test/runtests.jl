# TurboLily tests: every case is stepped side by side with the reference implementation
# (WaterLily / BiotSavartBCs on the same GPU) and must agree to rounding level.
using Test, TurboLily, WaterLily, BiotSavartBCs, CUDA, StaticArrays

const T = Float32
maxdiff(a, b) = maximum(abs, Array(a) .- Array(b))
# cells with every face open; pressure in partially blocked cells at the body surface is
# ill-conditioned and differs between any two runs (also between two BiotSavartBCs runs)
fluid(sim) = dropdims(minimum(Array(sim.flow.μ₀), dims=ndims(sim.flow.μ₀)), dims=ndims(sim.flow.μ₀)) .> 0
piters(sim) = sim.pois isa BiotSavartPoisson ? sim.pois.ml.n : sim.pois.n

# runs `nstep` steps of both simulations and returns (max |Δu|, max |Δp - mean| on fluid cells)
function compare(mk, nstep; remeasure=false, kw...)
    a = mk(:ref); b = mk(:turbo)
    @test b.flow isa TurboLily.TFlow
    for _ in 1:nstep
        sim_step!(a; remeasure, kw...); sim_step!(b; remeasure, kw...)
    end
    fl = fluid(a)
    dp = (Array(a.flow.p) .- Array(b.flow.p))[fl]
    return maxdiff(a.flow.u, b.flow.u), maximum(abs, dp .- sum(dp)/length(dp)), piters(a), piters(b)
end
ctor(which) = which == :ref ? WaterLily.Simulation : TurboLily.Simulation
bctor(which) = which == :ref ? BiotSavartBCs.BiotSimulation : TurboLily.BiotSimulation

circle(which; n=128, m=64, R=8f0, kw...) = (c = SA[m/2f0, m/2f0];
    ctor(which)((n,m),(1,0),R; ν=R/200, body=AutoBody((x,t)->√sum(abs2,x .- c)-R), T, kw...))
sphere(which; n=64, m=32, R=6f0, kw...) = (c = SA[m/2f0, m/2f0, m/2f0];
    ctor(which)((n,m,m),(1,0,0),R; ν=R/300, body=AutoBody((x,t)->√sum(abs2,x .- c)-R), T, kw...))
oscillating(which; n=96, m=48, R=6f0) = (c = SA[m/2f0, m/2f0]; z = 0f0;
    ctor(which)((n,m),(1,0),R; ν=R/200, T,
        body=AutoBody((x,t)->√sum(abs2,x .- c)-R, (x,t)->x - SA[z, 3f0*sin(0.2f0*t)])))
Ufun(i,x,t) = i==1 ? 1f0 - exp(-t/5f0) : 0f0
function tgv(which; L=32, Re=1600)
    κ = Float32(2π/L)
    function uλ(i, xyz)
        x, y, z = @. (xyz - 1)*κ
        return i == 1 ? -sin(x)*cos(y)*cos(z) : i == 2 ? cos(x)*sin(y)*cos(z) : zero(eltype(xyz))
    end
    ctor(which)((L,L,L), (0,0,0), 1/κ; U=1, u0=uλ, ν=1/(κ*Re), T, perdir=(1,2,3))
end

@testset "TurboLily vs WaterLily" begin
    # Periodic directions: TurboLily wraps neighbours in every Gauss-Seidel half sweep, while
    # WaterLily refreshes the periodic ghosts only between smoother calls, so periodic cases
    # agree to the solver tolerance (2e-3) rather than to rounding.
    for (name, mk, nstep, rem, tolu, tolp) in [
        ("2D circle", w->circle(w), 10, false, 1e-4, 1e-2),
        ("2D circle exitBC", w->circle(w; exitBC=true), 10, false, 1e-4, 1e-2),
        ("2D oscillating body", oscillating, 10, true, 1e-4, 1e-2),
        ("3D sphere", w->sphere(w), 5, false, 1e-4, 1e-2),
        ("3D sphere periodic z + exitBC", w->sphere(w; perdir=(3,), exitBC=true), 5, false, 2e-3, 1e-2),
        ("3D sphere uBC(t) + g", w->ctor(w)((64,32,32), Ufun, 6f0; U=1, ν=0.02, T,
            g=(i,x,t)->i==2 ? 0.01f0 : 0f0, body=AutoBody((x,t)->√sum(abs2,x .- SA[16f0,16f0,16f0])-6f0)), 5, false, 1e-4, 1e-2),
        ("3D periodic box, no body", w->ctor(w)((32,32,32),(1,0,0),8; ν=0.01, perdir=(2,3), T), 5, false, 2e-3, 1e-2),
        ("3D Taylor-Green vortex (fully periodic)", tgv, 10, false, 2e-3, 1e-2)]
        du, dp, na, nb = compare(mk, nstep; remeasure=rem)
        @testset "$name" begin
            @test du < tolu
            @test dp < tolp
            occursin("Taylor", name) || @test na == nb   # periodic GS converges in fewer iterations
        end
    end
    @testset "udf body force" begin
        grav!(flow, t; g=0.01f0) = (WaterLily.@loop flow.f[Ii] += g*(last(Ii)==2) over Ii in CartesianIndices(flow.f); nothing)
        du, dp, na, nb = compare(w->circle(w), 10; udf=grav!, g=0.02f0)
        @test du < 1e-4 && na == nb
    end
end

bcircle(which; m=128, D=32f0, kw...) = (c = SA[m/2f0, m/2f0];
    bctor(which)((m,m),(1,0),D; ν=D/1f3, body=AutoBody((x,t)->√sum(abs2,x .- c)-D/2), T, kw...))
bsphere(which; m=48, D=24f0, kw...) = (c = SA[m/2f0, m/2f0, m/2f0];
    bctor(which)((2m,m,m),(1,0,0),D; ν=D/1f3, body=AutoBody((x,t)->√sum(abs2,x .- c)-D/2), T, kw...))
bmoving(which; m=48, D=24f0) = (c = SA[m/2f0, m/2f0, m/2f0]; z = 0f0;
    bctor(which)((2m,m,m),(1,0,0),D; ν=D/1f3, T,
        body=AutoBody((x,t)->√sum(abs2,x .- c)-D/2, (x,t)->x - SA[z, 3f0*sin(0.1f0*t), z])))

@testset "TurboLily vs BiotSavartBCs" begin
    for (name, mk, rem) in [("2D circle", w->bcircle(w), false), ("3D sphere", w->bsphere(w), false),
                            ("3D sphere nonbiotfaces", w->bsphere(w; nonbiotfaces=(-2,)), false),
                            ("3D moving sphere", bmoving, true)]
        du, dp, na, nb = compare(mk, 5; remeasure=rem)
        @testset "$name" begin
            @test du < 1e-4
            @test dp < 1e-3
            @test na == nb
        end
    end
    @testset "potential flow around a circle" begin   # BiotSavartBCs test
        sim = TurboLily.BiotSimulation((512,512), (1,0), 256; T,
            body=AutoBody((x,t)->√sum(abs2,x .- 256)-128), ν=256/1e4)
        sim_step!(sim; remeasure=false)
        u = Array(sim.flow.u)
        @test abs(maximum(u[:,:,1]) - 2) < 0.02
        @test abs(maximum(u[:,:,2]) - 1) < 0.02
        @test abs(minimum(u[1,:,1]) - 0.75) < 0.02
    end
end
