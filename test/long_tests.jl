# Optional long tests: TurboLily against WaterLily / BiotSavartBCs at a tight solver tolerance
# (tol=1e-5), over more steps than runtests.jl. At the default tolerance (2e-3) the two multigrid
# solvers stop at different iterates, so runtests.jl can only require agreement to O(tol); here
# both solvers converge far enough that a discretization difference of more than ~1e-4 shows.
# Some cases amplify any difference over many steps (uBC(t) with Δt≈5.5, the Taylor-Green vortex),
# so each case also steps a second WaterLily run whose initial velocity is perturbed by 1e-6, and
# TurboLily must stay within `RATIO` times that run's distance (or within the fixed floor).
#
# Run with the rest of the suite:  julia --project -e 'using Pkg; Pkg.test(test_args=["long"])'
# Run alone (GPU host):            julia --project test/long_tests.jl
isdefined(@__MODULE__, :compare) || include("setup.jl")

const TIGHT = (; tol=1f-5, itmx=64)
const RATIO = 4

# deterministic perturbation of size `ε` on the interior faces of `sim.flow.u`
function perturb!(sim, ε)
    u = sim.flow.u; N = size(u)[1:end-1]
    for I in CartesianIndices(u)
        all(2 .<= Tuple(I)[1:end-1] .<= N .- 1) && (u[I] += ε*sin(T(LinearIndices(u)[I])))
    end
end
# `compare` with a third, perturbed reference run: returns (du, dp) TurboLily-vs-reference,
# (du, dp) perturbed-vs-reference, and the mean V-cycles per solve of the reference and TurboLily
function compare_sens(mk, nstep; remeasure=false, kw...)
    a = mk(:ref); q = mk(:ref); b = mk(:turbo)
    perturb!(q, 1f-6)
    for _ in 1:nstep
        sim_step!(a; remeasure, kw...); sim_step!(q; remeasure, kw...); sim_step!(b; remeasure, kw...)
    end
    fl = fluid(a)
    pd(x, y) = (d = (Array(x.flow.p) .- Array(y.flow.p))[fl]; maximum(abs, d .- sum(d)/length(d)))
    mean(n) = sum(n)/length(n)
    return (udiff(a.flow.u, b.flow.u), pd(a, b)), (udiff(a.flow.u, q.flow.u), pd(a, q)), mean(piters(a)), mean(piters(b))
end
function check_case(group, name, mk, nstep, rem; tolu=1e-4, tolp=5e-3)
    (du, dp), (su, sp), na, nb = compare_sens(mk, nstep; remeasure=rem, poisson_kwargs=TIGHT)
    @info "long $group: $name" nstep du perturbed_du=su dp perturbed_dp=sp vcycles=(na, nb)
    @testset "$name" begin
        @test du < max(tolu, RATIO*su)
        @test dp < max(tolp, RATIO*sp)
    end
end

# BiotSavartBCs takes the solver tolerance as positional arguments of `mom_project!`, so
# `sim_step!(...; poisson_kwargs)` would fall through to WaterLily's projection. Forward the
# keywords (more specific than BiotSavartBCs' own method in `t`, so nothing is overwritten).
WaterLily.mom_project!(a::WaterLily.AbstractFlow, b::BiotSavartPoisson, w::Int, t::Real; tol=2e-3, itmx=32) =
    WaterLily.mom_project!(a, b, w, t, tol, itmx)

@testset "Long tests (tol=1e-5)" begin
@testset "TurboLily vs WaterLily" begin
    for (name, mk, nstep, rem) in [
        ("2D circle", w->circle(w), 40, false),
        ("2D circle exitBC", w->circle(w; exitBC=true), 40, false),
        ("2D circle λ=cds", w->circle(w; λ=WaterLily.cds), 40, false),
        ("2D oscillating body", oscillating, 40, true),
        ("2D deforming body", w->ctor(w)((64,64),(0,0),2Rd; ν=2Rd/250, T, body=pulsing(SA_F32[32,32], sd)), 40, true),
        ("2D anti-phase deforming pair", w->ctor(w)((128,64),(0,0),2Rd; ν=2Rd/250, T,
            body=pulsing(SA_F32[32,32], sd)+pulsing(SA_F32[96,32], t->√(2-sd(t)^2))), 40, true),
        ("2D periodic y", w->ctor(w)((48,48), (1,0), 8; T, ν=0.05f0, perdir=(2,),
            body=AutoBody((x,t)->√sum(abs2, x .- SA[24f0,24f0])-6f0)), 40, false),
        ("3D sphere", w->sphere(w), 20, false),
        ("3D sphere 64³", w->sphere(w; n=64, m=64, R=10f0), 8, false),
        ("3D sphere, large x-y planes", w->ctor(w)((256,192,16), (1,0,0), 6f0; ν=0.02, T,
            body=AutoBody((x,t)->√sum(abs2,x .- SA[64f0,96f0,8f0])-6f0)), 6, false),
        ("3D sphere periodic z + exitBC", w->sphere(w; perdir=(3,), exitBC=true), 20, false),
        ("3D sphere uBC(t) + g", w->ctor(w)((64,32,32), Ufun, 6f0; U=1, ν=0.02, T,
            g=(i,x,t)->i==2 ? 0.01f0 : 0f0, body=AutoBody((x,t)->√sum(abs2,x .- SA[16f0,16f0,16f0])-6f0)), 20, false),
        ("3D periodic box, no body", w->ctor(w)((32,32,32),(1,0,0),8; ν=0.01, perdir=(2,3), T), 20, false),
        ("3D Taylor-Green vortex (fully periodic)", tgv, 20, false)]
        check_case("WaterLily", name, mk, nstep, rem)
    end
end

@testset "TurboLily vs BiotSavartBCs" begin
    for (name, mk, nstep, rem) in [
        ("2D circle", w->bcircle(w), 20, false),
        ("3D sphere", w->bsphere(w), 10, false),
        ("3D sphere nonbiotfaces", w->bsphere(w; nonbiotfaces=(-2,)), 10, false),
        ("2D half circle, symmetry plane", bhalfcircle, 20, false),
        ("3D plate, two symmetry planes", w->bplate(w), 10, false),
        ("3D moving sphere", bmoving, 10, true),
        ("3D deforming sphere", w->bctor(w)((96,48,48),(1,0,0),2Rd; ν=2Rd/250, T,
            body=pulsing(SA_F32[24,24,24], sd)), 10, true)]
        check_case("BiotSavartBCs", name, mk, nstep, rem)
    end
end
end
