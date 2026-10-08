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

# WaterLily's deforming-body test: a circle pulsating through a scaling map (V ≠ 0)
const Rd, εd, Tcd = 8f0, 0.25f0, 64f0
sd(t) = 1-εd*(1-cospi(2t/Tcd))
pulsing(c, s) = AutoBody((x,t)->√(x'*x)-Rd, (x,t)->(x .- c)/s(t))

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
        ("3D Taylor-Green vortex (fully periodic)", tgv, 10, false, 2e-3, 1e-2),
        # TurboLily includes the deforming-body source term of WaterLily#328; compare only
        # against a WaterLily that has it
        (isdefined(WaterLily, :δv) ? [
        ("2D deforming body", w->ctor(w)((64,64),(0,0),2Rd; ν=2Rd/250, T, body=pulsing(SA_F32[32,32], sd)), 8, true, 1e-4, 1e-2),
        ("2D anti-phase deforming pair", w->ctor(w)((128,64),(0,0),2Rd; ν=2Rd/250, T,
            body=pulsing(SA_F32[32,32], sd)+pulsing(SA_F32[96,32], t->√(2-sd(t)^2))), 8, true, 1e-4, 1e-2)] : [])...]
        du, dp, na, nb = compare(mk, nstep; remeasure=rem)
        @testset "$name" begin
            @test du < tolu
            @test dp < tolp
            occursin("Taylor", name) || @test na == nb   # periodic GS converges in fewer iterations
        end
    end
    @testset "update uBC with setproperties" begin   # WaterLily#340
        swap!(sim, uBC) = (sim.flow = WaterLily.setproperties(sim.flow; uBC); nothing)
        sim = TurboLily.Simulation((64,64), (1,0), 16; T)
        swap!(sim, (1f0,0f0))   # compile
        @test @allocated(swap!(sim, (2f0,0f0))) < sizeof(sim.flow.p)   # arrays are shared, not copied
        sim_step!(sim)
        @test all(Array(sim.flow.u)[2,2:end-1,1] .== 2)   # the kernels use the new uBC
        swap!(sim, (3,0))   # an Int tuple is converted to the flow's eltype
        @test sim.flow.uBC === (3f0,0f0)
        swap!(sim, (i,x,t)->i==1 ? 1.5f0 : 0f0)   # a function uBC rebuilds the acceleration term too
        sim_step!(sim)
        @test all(Array(sim.flow.u)[2,2:end-1,1] .== 1.5f0)
    end
    @testset "measure! after a jump in time" begin   # the body moves far past the measure band
        mk(w) = (c = SA[24f0, 32f0]; ctor(w)((128,64), (1,0), 8; T,
            body=AutoBody((x,t)->√sum(abs2, x .- c)-8f0, (x,t)->x - SA[t, 0f0])))
        a, b = mk(:ref), mk(:turbo)
        for t in (60f0, 0f0)
            measure!(a, t); measure!(b, t)
            @test maxdiff(a.flow.μ₀, b.flow.μ₀) < 1e-5
            @test maxdiff(a.flow.μ₁, b.flow.μ₁) < 1e-5
            @test maxdiff(a.flow.V, b.flow.V) < 1e-5
        end
        sim_step!(a); sim_step!(b)
        @test maxdiff(a.flow.u, b.flow.u) < 1e-4
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

# square plate on two symmetry planes (BiotSavartBCs README / former examples/square_sym.jl)
bplate(which; sym=(-2,-3)) = bctor(which)((64,32,32), (1,0,0), 16f0; ν=2*16f0/500, T, symmetry=sym,
    body=AutoBody(((x,y,z),t)->hypot(x-16f0, y-min(y,14f0), z-min(z,14f0))-2f0))
bhalfcircle(which) = bctor(which)((128,64),(1,0),32f0; ν=32f0/1f3, T, symmetry=(-2,),
    body=AutoBody((x,t)->√sum(abs2,x .- SA[64f0,0f0])-16f0))

@testset "TurboLily vs BiotSavartBCs" begin
    for (name, mk, rem) in [("2D circle", w->bcircle(w), false), ("3D sphere", w->bsphere(w), false),
                            ("3D sphere nonbiotfaces", w->bsphere(w; nonbiotfaces=(-2,)), false),
                            ("2D half circle, symmetry plane", bhalfcircle, false),
                            ("3D plate, two symmetry planes", w->bplate(w), false),
                            ("3D moving sphere", bmoving, true),
                            ("3D deforming sphere", w->bctor(w)((96,48,48),(1,0,0),2Rd; ν=2Rd/250, T,
                                body=pulsing(SA_F32[24,24,24], sd)), true)]
        du, dp, na, nb = compare(mk, 5; remeasure=rem)
        @testset "$name" begin
            @test du < 1e-4
            @test dp < 1e-3
            @test na == nb
        end
    end
    @testset "argument checks" begin   # as BiotSavartBCs.BiotSavartPoisson
        @test_throws ArgumentError bplate(:turbo; sym=(-2,2))   # one plane per direction
        @test_throws ArgumentError bplate(:turbo; sym=(4,))
        @test_throws ArgumentError TurboLily.BiotSimulation((64,32),(1,0),8f0; T, fmm=false, symmetry=(-2,))
        @test_throws ArgumentError TurboLily.BiotSimulation((64,32),(1,0),8f0; T, exitBC=true)
        @test_throws ArgumentError TurboLily.BiotSimulation((32,32,32),(1,0,0),8f0; T, perdir=(3,))
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
