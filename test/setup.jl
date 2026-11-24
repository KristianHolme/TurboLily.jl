# Shared helpers and simulation constructors for runtests.jl and long_tests.jl.
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
