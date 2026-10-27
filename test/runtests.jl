# TurboLily tests: every case is stepped side by side with the reference implementation
# (WaterLily / BiotSavartBCs on CPU, TurboLily on CUDA) and must agree to rounding level.
using Test, TurboLily, WaterLily, BiotSavartBCs, CUDA, StaticArrays

const T = Float32
maxdiff(a, b) = maximum(abs, Array(a) .- Array(b))
# Velocity difference on the faces a step defines: component i on the interior cells and on
# both boundary faces normal to i. Tangential ghost values are left out: neither code uses
# them, and their contents depend on which scratch buffer became `u`.
function udiff(a, b)
    A, B = Array(a), Array(b); N = size(A)[1:end-1]; D = length(N)
    maximum(1:D) do i
        R = CartesianIndices(ntuple(d -> d == i ? (2:N[d]) : (2:N[d]-1), D))
        maximum(I -> abs(A[I, i] - B[I, i]), R)
    end
end
# cells with no fully closed face (μ₀>0 on every face); pressure in cells with a closed face
# at the body surface is ill-conditioned and differs between any two runs (also between two
# BiotSavartBCs runs), so pressure is not compared there
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
    return udiff(a.flow.u, b.flow.u), maximum(abs, dp .- sum(dp)/length(dp)), piters(a), piters(b)
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
        ("2D circle λ=cds", w->circle(w; λ=WaterLily.cds), 10, false, 1e-4, 1e-2),
        ("2D oscillating body", oscillating, 10, true, 1e-4, 1e-2),
        ("3D sphere", w->sphere(w), 5, false, 1e-4, 1e-2),
        # 66³ cells: large enough for the fused Gauss-Seidel pair (non-periodic 3D levels)
        ("3D sphere 64³", w->sphere(w; n=64, m=64, R=10f0), 3, false, 1e-4, 1e-2),
        ("3D sphere periodic z + exitBC", w->sphere(w; perdir=(3,), exitBC=true), 5, false, 2e-3, 1e-2),
        # uBC starts at 0, so the second step is Δt≈5.5. After 5 steps WaterLily with a 1e-6 velocity
        # perturbation differs from WaterLily by 2.0e-4, so the velocity tolerance is wider here
        ("3D sphere uBC(t) + g", w->ctor(w)((64,32,32), Ufun, 6f0; U=1, ν=0.02, T,
            g=(i,x,t)->i==2 ? 0.01f0 : 0f0, body=AutoBody((x,t)->√sum(abs2,x .- SA[16f0,16f0,16f0])-6f0)), 5, false, 3e-4, 1e-2),
        ("3D periodic box, no body", w->ctor(w)((32,32,32),(1,0,0),8; ν=0.01, perdir=(2,3), T), 5, false, 2e-3, 1e-2),
        ("3D Taylor-Green vortex (fully periodic)", tgv, 10, false, 2e-3, 1e-2),
        ("2D deforming body", w->ctor(w)((64,64),(0,0),2Rd; ν=2Rd/250, T, body=pulsing(SA_F32[32,32], sd)), 8, true, 1e-4, 1e-2),
        ("2D anti-phase deforming pair", w->ctor(w)((128,64),(0,0),2Rd; ν=2Rd/250, T,
            body=pulsing(SA_F32[32,32], sd)+pulsing(SA_F32[96,32], t->√(2-sd(t)^2))), 8, true, 1e-4, 1e-2)]
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
        @test udiff(a.flow.u, b.flow.u) < 1e-4
    end
    @testset "udf body force" begin
        grav!(flow, t; g=0.01f0) = (WaterLily.@loop flow.f[Ii] += g*(last(Ii)==2) over Ii in CartesianIndices(flow.f); nothing)
        du, dp, na, nb = compare(w->circle(w), 10; udf=grav!, g=0.02f0)
        @test du < 1e-4 && na == nb
    end
    @testset "udf Smagorinsky sgs!" begin
        # (Cs·Δ)²|S|, written without views so that it also compiles in a GPU kernel
        function smag(I::CartesianIndex{D}; S, Cs, Δ) where D
            s = zero(eltype(S))
            for i in 1:D, j in 1:D
                s += S[I, i, j]^2
            end
            return (Cs*Δ)^2*sqrt(s)
        end
        a, b = circle(:ref), circle(:turbo)
        Sa = zeros(T, size(a.flow.p)..., 2, 2); Sb = CUDA.zeros(T, size(b.flow.p)..., 2, 2)
        for _ in 1:10
            sim_step!(a; remeasure=false, udf=WaterLily.sgs!, νₜ=smag, S=Sa, Cs=0.17f0, Δ=1f0)
            sim_step!(b; remeasure=false, udf=WaterLily.sgs!, νₜ=smag, S=Sb, Cs=0.17f0, Δ=1f0)
        end
        @test udiff(a.flow.u, b.flow.u) < 1e-4
        @test piters(a) == piters(b)
    end
    @testset "udf sees WaterLily's flow state" begin
        # WaterLily zeroes the interior of flow.u before the predictor udf
        interior(u) = (A = Array(u); A[ntuple(d -> 2:size(A, d)-1, ndims(A)-1)..., :])
        probe!(flow, u, t; rec) = (push!(rec, (maximum(abs, interior(flow.u)), maximum(abs, interior(u)))); nothing)
        ra, rb = Tuple{T,T}[], Tuple{T,T}[]
        a, b = circle(:ref), circle(:turbo)
        for _ in 1:3
            sim_step!(a; remeasure=false, udf=probe!, rec=ra)
            sim_step!(b; remeasure=false, udf=probe!, rec=rb)
        end
        @test length(ra) == length(rb) == 6
        @test all(k -> isapprox(collect(ra[k]), collect(rb[k]); atol=1e-4), eachindex(ra))
        @test all(k -> ra[k][1] == rb[k][1] == 0, 1:2:6)   # predictor: zero interior
        # a two-argument callback that reads flow.u
        drag!(flow, t) = (flow.f .-= 0.1f0 .* flow.u; nothing)
        du, dp, na, nb = compare(w->circle(w), 10; udf=drag!)
        @test du < 1e-4 && na == nb
    end
    @testset "poisson_kwargs go to the solver, not to udf" begin   # WaterLily#245
        grav!(flow, t; g=0.01f0) = (WaterLily.@loop flow.f[Ii] += g*(last(Ii)==2) over Ii in CartesianIndices(flow.f); nothing)
        b = circle(:turbo); sim_step!(b; remeasure=false)
        for kw in ((;), (; udf=grav!, g=0.02f0))
            WaterLily.mom_step!(b.flow, b.pois; poisson_kwargs=(; itmx=1), kw...)
            @test b.pois.n[end-1:end] == [1, 1]
            WaterLily.mom_step!(b.flow, b.pois; poisson_kwargs=(; tol=1f-6), kw...)
            @test all(>(1), b.pois.n[end-1:end])
        end
        c = TurboLily.BiotSimulation((64,64), (1,0), 8; ν=8/200f0, T,
            body=AutoBody((x,t)->√sum(abs2, x .- SA[32f0,32f0])-8f0))
        sim_step!(c; remeasure=false)
        WaterLily.mom_step!(c.flow, c.pois; poisson_kwargs=(; itmx=2))
        @test all(<=(2), c.pois.n[end-1:end])
        # through sim_step!, against a WaterLily whose sim_step! takes poisson_kwargs (after 1.10.0)
        if any(m -> :poisson_kwargs in Base.kwarg_decl(m), methods(WaterLily.sim_step!))
            du, dp, na, nb = compare(w->circle(w), 5; udf=grav!, g=0.02f0, poisson_kwargs=(; tol=1f-4))
            @test du < 1e-4 && na == nb
        end
    end
    @testset "thin grids: reduction scratch" begin
        # the block count of a launch depends on the rounded-up extent in each direction,
        # so thin grids need more partials than cld(cells, 64)
        for (dims, uBC) in (((2,128), (0,1)), ((2,2,128), (0,0,1)))
            n = dims .+ 2
            red = TurboLily.RedBuf(TurboLily.Grid(n))
            @test length(red.ps) >= TurboLily.nblocks(ntuple(_->1, length(n)), n)
            u0(i, x) = i == length(dims) ? 1 + 0.1f0*sinpi(x[end]/16) : 0f0
            du, dp, na, nb = compare(w->ctor(w)(dims, uBC, 8; T, ν=0.01f0, u0), 3)
            @test du < 1e-4
            @test na == nb
        end
    end
    @testset "odd periodic multigrid levels" begin
        # 48 coarsens to an odd 3-cell level, 63 is odd on every level; red-black sweeps
        # must not race across the periodic wrap
        for (dims, perdir) in (((48,48), (1,2)), ((63,64), (1,)))
            mk(w) = (c = SA[dims[1]/2f0, dims[2]/2f0];
                ctor(w)(dims, (1,0), 8; T, ν=0.05f0, perdir, body=AutoBody((x,t)->√sum(abs2, x .- c)-6f0)))
            du, dp, na, nb = compare(mk, 5)
            @test du < 2e-3
            @test dp < 1e-2
        end
    end
end

@testset "Poisson residual recomputed from scratch" begin
    # The solver tracks the residual incrementally; check the final x against r = z - A x
    # computed by WaterLily on the same arrays.
    for (dims, perdir) in (((128,64), ()), ((48,48), (1,2)), ((63,64), (1,)), ((32,32,32), (3,)))
        @testset "$dims perdir=$perdir" begin
            D = length(dims)
            c = SVector{D,T}(ntuple(d -> dims[d]/2f0, D))
            sim = TurboLily.Simulation(dims, ntuple(i -> Int(i == 1), D), 8; T, perdir,
                body=AutoBody((x,t)->√sum(abs2, x .- c)-6f0))
            p = sim.pois; lv = p.ml.levels[1]
            iD = Array(lv.iD)
            z = zeros(T, size(iD))
            for I in CartesianIndices(z)
                iD[I] != 0 && (z[I] = sin(T(LinearIndices(z)[I])))
            end
            solved = iD .!= 0
            z[solved] .-= sum(z[solved])/count(solved)
            copyto!(lv.z, z); fill!(lv.x, 0)
            WaterLily.solver!(p; tol=2e-3)
            @test p.n[end] < 32
            WaterLily.residual!(lv)
            @test maximum(abs, Array(lv.r)) < 2e-3
            @test sum(abs, Array(lv.r)) < WaterLily.l1n_tol(lv, 2e-3)
        end
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
        @test_throws ArgumentError TurboLily.BiotSimulation((64,32),(1,0),8f0; T, fmm=false)   # tree sum
        @test_throws ArgumentError TurboLily.BiotSimulation((8,8),(1,0),1; T)   # one Biot-Savart level
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

@testset "Red-black sweeps cover every interior cell" begin
    for dims in ((8, 10), (8, 9), (8, 10, 10), (8, 10, 9), (8, 9, 9))
        @testset "$dims" begin
            D = length(dims)
            sim = TurboLily.Simulation(dims, ntuple(i -> Int(i == 1), D), 1; T)
            lv = sim.pois.levels[1]
            r = zeros(T, size(lv.r))
            interior = CartesianIndices(map(n -> 2:(n - 1), size(r)))
            for I in interior
                r[I] = sin(T(LinearIndices(r)[I]))
            end
            copyto!(lv.r, r)
            fill!(lv.ϵ, 0)
            iD, L = Array(lv.op.iD), Array(lv.op.L)
            expected = r .* iD
            for sweep in 1:4
                for I in interior
                    isodd(sum(Tuple(I)) + sweep) || continue
                    s = zero(T)
                    for j in 1:D
                        δ = CartesianIndex(ntuple(d -> Int(d == j), D))
                        s += expected[I - δ] * L[Tuple(I)..., j] +
                            expected[I + δ] * L[Tuple(I + δ)..., j]
                    end
                    expected[I] = (r[I] - s) * iD[I]
                end
                TurboLily.launch(
                    TurboLily.k_gs!, TurboLily.gs_range(lv.op.g)...,
                    lv.ϵ, lv.r, lv.op, Int32(sweep), sweep == 1
                )
                @test maxdiff(lv.ϵ, expected) < 2.0f-6
            end
        end
    end
end

@testset "Fused Gauss-Seidel pair equals two half sweeps" begin
    # sizes that are not multiples of the 30×6 tiles or the 16-plane chunks; the body gives
    # cells with irregular coefficients
    for (dims, perdir) in (((64, 64, 62), ()), ((64, 72, 56), ()), ((64, 64, 64), (3,)), ((64, 64, 64), (1, 2, 3)), ((64, 72, 56), (2,)))
        @testset "$dims perdir=$perdir" begin
            c = SA[24f0, 30f0, 28f0]
            sim = TurboLily.Simulation(dims, (1, 0, 0), 8; T, perdir, body = AutoBody((x, t) -> √sum(abs2, x .- c) - 10f0))
            lv = sim.pois.levels[1]; op = lv.op
            @test TurboLily.usegs2(op)
            r = zeros(T, size(lv.r))
            interior = CartesianIndices(map(n -> 2:(n - 1), size(r)))
            for I in interior
                r[I] = sin(T(LinearIndices(r)[I]))
            end
            copyto!(lv.r, r)
            lo, hi = TurboLily.gs_range(op.g)
            fill!(lv.ϵ, 0)
            for k in 1:4
                TurboLily.launch(TurboLily.k_gs!, lo, hi, lv.ϵ, lv.r, op, Int32(k), k == 1)
            end
            ref = Array(lv.ϵ)
            fill!(lv.ϵ, 0); fill!(lv.z, 0)
            for k in 1:2
                TurboLily.launch(TurboLily.k_gs!, lo, hi, lv.z, lv.r, op, Int32(k), k == 1)
            end
            TurboLily.gs2!(lv.z, lv.ϵ, lv.r, op, 3)
            @test Array(lv.ϵ)[interior] == ref[interior]
        end
    end
end

@testset "Odd periodic smoother is race-free" begin
    # Serial reference: WaterLily.GaussSeidelRB! (ghosts filled once, then 4 half sweeps).
    # Every thread layout must give it exactly; a race across the wrap would not.
    for (dims, perdir) in (((65, 64), (1,)), ((64, 65), (2,)), ((1025, 64), (1,)), ((9, 16, 11), (1, 3)))
        @testset "$dims perdir=$perdir" begin
            D = length(dims)
            sim = TurboLily.Simulation(dims, ntuple(i -> Int(i == 1), D), 1; T, perdir)
            lv = sim.pois.levels[1]; op = lv.op; g = op.g
            @test TurboLily.oddper(op)
            r = zeros(T, size(lv.r))
            interior = CartesianIndices(map(n -> 2:(n - 1), size(r)))
            for I in interior
                r[I] = sin(T(LinearIndices(r)[I]))
            end
            copyto!(lv.r, r)
            iD, L = Array(op.iD), Array(op.L)
            expected = r .* iD
            for j in perdir   # perBC!
                n = size(r, j)
                sel(k) = ntuple(d -> d == j ? (k:k) : (:), D)
                expected[sel(1)...] .= expected[sel(n-1)...]
                expected[sel(n)...] .= expected[sel(2)...]
            end
            for sweep in 1:4, I in interior
                isodd(sum(Tuple(I)) + sweep) || continue
                s = zero(T)
                for j in 1:D
                    δ = CartesianIndex(ntuple(d -> Int(d == j), D))
                    s += expected[I - δ] * L[Tuple(I)..., j] + expected[I + δ] * L[Tuple(I + δ)..., j]
                end
                expected[I] = (r[I] - s) * iD[I]
            end
            lo, hi = Int32.(TurboLily.gs_range(g)[1]), Int32.(TurboLily.gs_range(g)[2])
            ext = (Int.(hi .- lo .+ 1)..., ntuple(_ -> 1, 3 - D)...)
            nop = TurboLily.noper(op)
            for th in (D == 2 ? ((32, 8, 1), (16, 16, 1), (64, 4, 1), (128, 2, 1)) : ((32, 4, 2), (8, 8, 4), (4, 4, 16)))
                fill!(lv.ϵ, 0)
                TurboLily.launch(TurboLily.k_rid!, TurboLily.inside_lo(g), TurboLily.inside_hi(g), lv.ϵ, lv.r, op)
                TurboLily.perBC!(lv.ϵ, g, op.per)
                for k in 1:4
                    @cuda threads=th blocks=cld.(ext, th) TurboLily.k_gs!(lo, hi, lv.ϵ, lv.r, nop, Int32(k), false)
                end
                @test maxdiff(Array(lv.ϵ)[interior], expected[interior]) < 2.0f-6
            end
        end
    end
end

@testset "Biot increments after body motion" begin
    for D in (2, 3)
        @testset "$(D)D" begin
            dims = ntuple(d -> d == 1 ? 64 : 32, D)
            c = SVector{D, T}(ntuple(_ -> 8.0f0, D))
            direction = SVector{D, T}(ntuple(d -> T(d == 1), D))
            body = AutoBody(
                (x, t) -> sqrt(sum(abs2, x - c)) - 6.0f0,
                (x, t) -> x - t * direction
            )
            sim = TurboLily.BiotSimulation(dims, ntuple(i -> Int(i == 1), D), 12; T, body)
            b, a = sim.pois, sim.flow
            u = reshape(sin.(T.(1:length(a.u))), size(a.u))
            p = reshape(cos.(T.(1:length(a.p))), size(a.p))
            copyto!(a.u, u)
            for t in (0.0f0, 12.0f0, 80.0f0, 0.0f0)
                measure!(sim, t)
                copyto!(a.p, p)
                TurboLily.fill_ω!(b, a.f, a.u)
                TurboLily.correct_ω!(b, a.f, a.u, a.p, 0.25f0, false)
                previous = map(Array, (a.f, b.ω...))
                a.p .*= 2.0f0
                TurboLily.correct_ω!(b, a.f, a.u, a.p, 0.25f0, true)
                for l in 1:2
                    current = l == 1 ? a.f : b.ω[1]
                    @test maxdiff(b.dω[l], Array(current) .- previous[l]) == 0
                end
                full = map(copy, (a.f, b.ω...))
                for l in 3:length(b.g)
                    TurboLily.launch(
                        TurboLily.k_vrestrict!, TurboLily.inside_lo(b.g[l]),
                        TurboLily.inside_hi(b.g[l]), full[l], b.g[l], full[l - 1], b.g[l - 1]
                    )
                    @test maxdiff(b.dω[l], Array(full[l]) .- previous[l]) < 2.0f-5
                end
                if t == 80.0f0
                    @test b.nA == [0, 0]
                else
                    @test all(>(0), b.nA)
                end
            end
        end
    end
end

include("waterlily_tests.jl")
