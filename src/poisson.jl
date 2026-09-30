using WaterLily: MultiLevelPoisson, Poisson, coarsen_mask
using Adapt

"""
    Op

Kernel-side view of one Poisson level `A = L + D + L'`. Cells flagged `reg[I]≠0` have
`L=Lreg` on all 2d faces, so `D=Dreg` and `iD=iDreg`; the solver kernels then skip
reading `L`, `D` and `iD`, which dominate the memory traffic away from the body.
"""
struct Op{D,T,S<:AbstractArray{T},V<:AbstractArray{T},R<:AbstractArray{UInt8}}
    L :: V
    Dg :: S
    iD :: S
    reg :: R
    Lreg :: T
    Dreg :: T
    iDreg :: T
    g :: Grid{D}
    per :: NTuple{D,Bool}
end
Adapt.@adapt_structure Op

@inline isreg(op::Op, l) = @inbounds op.reg[l] != 0x00
@inline getiD(op::Op, l) = isreg(op, l) ? op.iDreg : @inbounds(op.iD[l])

"Array as a neighbour-value functor `(I,l)->x[l]`."
struct AV{S}; x::S; end
@inline (f::AV)(I, l) = @inbounds f.x[l]

# Σ_nb L·f (off-diagonal part of A) with neighbour values from f(I,l)
@inline function offdiag_f(op::Op{D}, I, l, f::F) where {D,F}
    g, per = op.g, op.per
    s = zero(op.Lreg)
    if isreg(op, l)
        @unroll for j in i32:Int32(D)
            s += f(shift(I,j,-1), nbm(g,I,l,j,per))*op.Lreg + f(shift(I,j,1), nbp(g,I,l,j,per))*op.Lreg
        end
    else
        @unroll for j in i32:Int32(D)
            cj = (j-i32)*g.nc
            s += f(shift(I,j,-1), nbm(g,I,l,j,per))*@inbounds(op.L[l+cj]) +
                 f(shift(I,j,1), nbp(g,I,l,j,per))*@inbounds(op.L[l+g.s[j]+cj])
        end
    end
    return s
end
# A·f at cell I
@inline function mult_f(op::Op{D}, I, l, f::F) where {D,F}
    g, per = op.g, op.per
    if isreg(op, l)
        s = f(I,l)*op.Dreg
        @unroll for j in i32:Int32(D)
            s += f(shift(I,j,-1), nbm(g,I,l,j,per))*op.Lreg + f(shift(I,j,1), nbp(g,I,l,j,per))*op.Lreg
        end
    else
        s = f(I,l)*@inbounds(op.Dg[l])
        @unroll for j in i32:Int32(D)
            cj = (j-i32)*g.nc
            s += f(shift(I,j,-1), nbm(g,I,l,j,per))*@inbounds(op.L[l+cj]) +
                 f(shift(I,j,1), nbp(g,I,l,j,per))*@inbounds(op.L[l+g.s[j]+cj])
        end
    end
    return s
end
@inline mult(op::Op, I, l, x::AbstractArray) = mult_f(op, I, l, AV(x))

"""
    Level

One multigrid level. Arrays alias the corresponding `WaterLily.Poisson` level;
`r` and `ϵ` may be swapped by the fused Jacobi kernel.
"""
mutable struct Level{D,T,S<:AbstractArray{T},V<:AbstractArray{T},R}
    op :: Op{D,T,S,V,R}
    x :: S
    ϵ :: S
    r :: S
    z :: S
    c :: NTuple{D,Bool}     # coarsening mask to the next level
end
function Level(p::Poisson{T}, c, per) where T
    g = Grid(p.x)
    reg = similar(p.x, UInt8); fill!(reg, 0x00)
    op = Op(p.L, p.D, p.iD, reg, one(T), -T(2ndims(p.x)), inv(-T(2ndims(p.x))), g, per)
    Level(op, p.x, p.ϵ, p.r, p.z, c)
end

function k_regflag!(lo, hi, reg, L, g::Grid{D}, Lreg) where D
    I, ok = cellidx(lo, hi); ok || return
    l = lin(g, I)
    b = true
    @unroll for j in i32:Int32(D)
        cj = (j-i32)*g.nc
        @inbounds b &= (L[l+cj] == Lreg) & (L[l+g.s[j]+cj] == Lreg)
    end
    @inbounds reg[l] = b ? 0x01 : 0x00
    return nothing
end
"Recompute the regular-cell flags from `L` (after `L` changed)."
function set_reg!(lv::Level{D,T}) where {D,T}
    op = lv.op; g = op.g
    Lreg = maximum(op.L)
    fill!(op.reg, 0x00)
    launch(k_regflag!, inside_lo(g), inside_hi(g), op.reg, op.L, g, Lreg)
    Dreg = -T(2D)*Lreg
    lv.op = Op(op.L, op.Dg, op.iD, op.reg, Lreg, Dreg, inv(Dreg), g, op.per)
end

"""
    TPoisson

Multigrid pressure solver with the same algorithm as `WaterLily.MultiLevelPoisson`
(Jacobi pre-smoothing, red-black Gauss-Seidel post-smoothing, adaptive ω), with
fused CUDA kernels, compressed regular-cell stencils and one host synchronisation
per V-cycle.
"""
struct TPoisson{T,S,V,D,R} <: AbstractPoisson{T,S,V}
    x :: S
    L :: V
    z :: S
    ml :: MultiLevelPoisson{T,S,V}
    levels :: Vector{Level{D,T,S,V,R}}
    per :: NTuple{D,Bool}
    n :: Vector{Int16}
    perdir :: NTuple
    nsolve :: Base.RefValue{Int}     # number of solved rows (count(!iszero, iD))
    red :: RedBuf
end
function TPoisson(x::AbstractArray{T,D}, L, z; perdir=()) where {T,D}
    ml = MultiLevelPoisson(x, L, z; perdir)
    nl = length(ml.levels)
    per = ntuple(j->j ∈ perdir, D)
    levels = [Level(ml.levels[l], l < nl ? coarsen_mask(ml.levels[l].x, ml.levels[l+1].x) : ntuple(_->false,D), per) for l in 1:nl]
    p = TPoisson{T,typeof(x),typeof(L),D,typeof(levels[1].op.reg)}(x, L, z, ml, levels, per, Int16[], perdir, Ref(0), RedBuf(Grid(x)))
    finish_update!(p)
end
TPoisson(flow::AbstractFlow) = TPoisson(flow.p, flow.μ₀, flow.σ; perdir=flow.perdir)
function finish_update!(p::TPoisson)
    foreach(set_reg!, p.levels)
    p.nsolve[] = count(!iszero, p.levels[1].op.iD)
    return p
end
function WaterLily.update!(p::TPoisson)
    WaterLily.update!(p.ml)
    finish_update!(p)
end

# ---------------------------------------------------------------- kernels
# Jacobi: ϵ=r·iD, r ← r-Aϵ (written to rn), x += ϵ
struct RiD{S,O}; r::S; op::O; end
@inline (f::RiD)(I, l) = @inbounds(f.r[l])*getiD(f.op, l)
function k_jacobi!(lo, hi, rn, r, x, op::Op)
    I, ok = cellidx(lo, hi); ok || return
    l = lin(op.g, I)
    f = RiD(r, op)
    e = f(I, l)
    @inbounds rn[l] = r[l] - mult_f(op, I, l, f)
    @inbounds x[l] += e
    return nothing
end
# restriction: coarse r = Σ fine r over the 2ᵈ block (semi-coarsening mask c)
function k_restrict!(lo, hi, rc, gc::Grid{D}, rf, gf::Grid{D}, c) where D
    I, ok = cellidx(lo, hi); ok || return
    J0 = ntuple(d-> c[d] ? Int32(2)*I[d]-Int32(2) : I[d], Val(D))
    l0 = lin(gf, J0)
    s = 0f0
    if D == 3
        for dz in 0:Int(c[3]), dy in 0:Int(c[2]), dx in 0:Int(c[1])
            @inbounds s += rf[l0 + dx*gf.s[1] + dy*gf.s[2] + dz*gf.s[3]]
        end
    else
        for dy in 0:Int(c[2]), dx in 0:Int(c[1])
            @inbounds s += rf[l0 + dx*gf.s[1] + dy*gf.s[2]]
        end
    end
    @inbounds rc[lin(gc, I)] = s
    return nothing
end
# piecewise-constant prolongation of the coarse solution
@inline function down(I::NTuple{D,Int32}, c) where D
    ntuple(d-> c[d] ? (I[d]+Int32(2))÷Int32(2) : I[d], Val(D))
end
@inline function wrapI(g::Grid{D}, I, per) where D
    ntuple(Val(D)) do d
        (per[d] && I[d]==i32) ? g.n[d]-i32 : (per[d] && I[d]==g.n[d]) ? Int32(2) : I[d]
    end
end
struct Prol{D,S,C}; cx::S; gc::Grid{D}; g::Grid{D}; c::C; per::NTuple{D,Bool}; end
@inline (f::Prol)(I, l) = @inbounds f.cx[lin(f.gc, down(wrapI(f.g, I, f.per), f.c))]
# increment with ϵ = prolongation(cx): r -= ωAϵ, x += ωϵ
function k_prol_incr!(lo, hi, r, x, op::Op, cx, gc::Grid, c, ω)
    I, ok = cellidx(lo, hi); ok || return
    l = lin(op.g, I)
    f = Prol(cx, gc, op.g, c, op.per)
    e = f(I, l)
    @inbounds r[l] -= ω*mult_f(op, I, l, f)
    @inbounds x[l] += ω*e
    return nothing
end
# increment with stored ϵ (optionally the block partials of Σ|r|, max|r|)
function k_incr!(lo, hi, ps, pm, r, x, ϵ, op::Op, ω, norms::Bool)
    I, ok = cellidx(lo, hi)
    a = 0f0
    if ok
        l = lin(op.g, I)
        v = @inbounds(r[l]) - ω*mult(op, I, l, ϵ)
        @inbounds r[l] = v
        @inbounds x[l] += ω*ϵ[l]
        a = abs(v)
    end
    norms && block_partial!(ps, pm, a, a)
    return nothing
end
# red-black Gauss-Seidel half sweep on ϵ (WaterLily.gauss_rb indexing over half_rangek).
# `init`: first sweep, also sets ϵ=r·iD on the partner cell and reads neighbours as r·iD.
function k_gs!(lo, hi, ϵ, r, op::Op{D}, k0::Int32, init::Bool) where D
    Iv, ok = cellidx(lo, hi); ok || return
    g = op.g
    fs = Int32(0)
    @unroll for d in i32:Int32(D-1)
        fs += Iv[d]
    end
    p = (fs + k0) % Int32(2)
    k = 2Iv[D] - i32 - p
    I = setj(Iv, D, k)
    l = lin(g, I)
    if init
        kp = 2Iv[D] - i32 - (i32 - p)       # partner cell in the pair {2kk-2, 2kk-1}
        if kp <= g.n[D] - i32
            lp = lin(g, setj(Iv, D, kp))
            @inbounds ϵ[lp] = r[lp]*getiD(op, lp)
        end
        if 2Iv[D] == g.n[D] - i32           # odd size: last cell is outside the sweep range
            lx = lin(g, setj(Iv, D, g.n[D] - i32))
            @inbounds ϵ[lx] = r[lx]*getiD(op, lx)
        end
        @inbounds ϵ[l] = (r[l] - offdiag_f(op, I, l, RiD(r, op)))*getiD(op, l)
    else
        @inbounds ϵ[l] = (r[l] - offdiag_f(op, I, l, AV(ϵ)))*getiD(op, l)
    end
    return nothing
end
# perBC! for a scalar field
function k_perbc!(lo, hi, j, a, g::Grid)
    I, ok = faceidx(lo, hi, j); ok || return
    l1 = lin(g, I); sj = g.s[j]; nj = g.n[j]
    @inbounds a[l1] = a[l1 + (nj-Int32(2))*sj]
    @inbounds a[l1 + (nj-i32)*sj] = a[l1 + sj]
    return nothing
end
function perBC!(a, g::Grid{D}, per) where D
    for j in 1:D
        per[j] || continue
        launch_face(k_perbc!, ntuple(_->1,D), ntuple(d-> d==j ? 1 : Int(g.n[d]), D), j, a, g)
    end
end

# ---------------------------------------------------------------- lazily corrected residual
# rc = iD==0 ? 0 : r-s, with s = Σr/nsolve applied only when |s|>tiny (WaterLily.residual!)
struct Rc{S,O}; r::S; op::O; s::Float32; apply::Bool; end
@inline function (f::Rc)(I, l)
    v = @inbounds f.r[l]
    return f.apply ? (getiD(f.op, l) == 0 ? 0f0 : v - f.s) : v
end
struct RcD{S,O}; rc::Rc{S,O}; end
@inline (f::RcD)(I, l) = f.rc(I, l)*getiD(f.rc.op, l)
@inline function make_rc(r, op, sums, nsolve, tiny)
    s = Float32(@inbounds(sums[1])) / nsolve
    return Rc(r, op, s, abs(s) > tiny)
end
# Jacobi on the corrected residual; partial Σ|rc|, max|rc| (the initial residual norms)
function k_jacobi_fix!(lo, hi, ps, pm, rn, r, x, op::Op, sums, nsolve, tiny)
    I, ok = cellidx(lo, hi)
    a = 0f0
    if ok
        l = lin(op.g, I)
        rc = make_rc(r, op, sums, nsolve, tiny)
        v = rc(I, l)
        f = RcD(rc)
        @inbounds rn[l] = v - mult_f(op, I, l, f)
        @inbounds x[l] += v*getiD(op, l)
        a = abs(v)
    end
    block_partial!(ps, pm, a, a)
    return nothing
end

# ---------------------------------------------------------------- solver pieces
function jacobi!(lv::Level)
    g = lv.op.g
    launch(k_jacobi!, inside_lo(g), inside_hi(g), lv.ϵ, lv.r, lv.x, lv.op)
    lv.r, lv.ϵ = lv.ϵ, lv.r
end
gs_range(g::Grid{D}) where D = (inside_lo(g), ntuple(d-> d==D ? Int(g.n[d])÷2 : Int(g.n[d])-1, D))
function smooth!(lv::Level{D,T}, ω, red::RedBuf, norms::Bool=false; it=4) where {D,T}
    g = lv.op.g
    lo, hi = gs_range(g)
    for i in 1:it
        launch(k_gs!, lo, hi, lv.ϵ, lv.r, lv.op, Int32(i), i==1)
    end
    if norms
        red_launch(k_incr!, inside_lo(g), inside_hi(g), red, 3, lv.r, lv.x, lv.ϵ, lv.op, T(ω), true)
    else
        launch(k_incr!, inside_lo(g), inside_hi(g), red.ps, red.pm, lv.r, lv.x, lv.ϵ, lv.op, T(ω), false)
    end
end
# V-cycle below level l; `fix` = (nsolve, tiny): the fine residual still needs the
# global correction, which is applied inside the fine Jacobi (norms into slot 2)
function vcycle!(p::TPoisson{T}, l, ω, fix=nothing) where T
    fine, coarse = p.levels[l], p.levels[l+1]
    if fix === nothing
        jacobi!(fine)
    else
        g = fine.op.g
        red_launch(k_jacobi_fix!, inside_lo(g), inside_hi(g), p.red, 2, fine.ϵ, fine.r, fine.x, fine.op, p.red.s, fix...)
        fine.r, fine.ϵ = fine.ϵ, fine.r
    end
    gc = coarse.op.g
    launch(k_restrict!, inside_lo(gc), inside_hi(gc), coarse.r, gc, fine.r, fine.op.g, fine.c)
    fill!(coarse.x, 0)
    l+1 < length(p.levels) && vcycle!(p, l+1, ω)
    smooth!(coarse, ω, p.red)
    g = fine.op.g
    launch(k_prol_incr!, inside_lo(g), inside_hi(g), fine.r, fine.x, fine.op, coarse.x, gc, fine.c, T(ω))
end

"""
    solve!(p::TPoisson, scale; tol=2e-3, itmx=32)

Multigrid iterations on `A x = z` where the level-1 residual `r` (uncorrected, with its
sum in `p.red.s[1]`) was set by the caller. Convergence is tested on `scale·r`, so a
caller solving for `p = x/Δt` passes `scale = Δt` and WaterLily's tolerances keep their meaning.
"""
function solve!(p::TPoisson{T}, scale; tol=2e-3, itmx=32) where T
    top = p.levels[1]
    r₁tol = WaterLily.l1n_tol(p.ml.levels[1], tol)/scale; r∞tol = tol/scale
    fix = (Float32(p.nsolve[]), T(2eps(T)/scale))
    ω = T(1); nᵖ = 0; r₁ = Inf
    while nᵖ < itmx
        vcycle!(p, 1, ω, nᵖ == 0 ? fix : nothing)
        smooth!(top, ω, p.red, true)
        fetch!(p.red)
        nᵖ == 0 && (r₁ = getsum(p.red,2); @log ", 0, $(getmax(p.red,2)*scale), $(r₁*scale), $ω\n")
        rnew, r∞ = getsum(p.red,3), getmax(p.red,3); nᵖ += 1
        @log ", $nᵖ, $(r∞*scale), $(rnew*scale), $ω\n"
        ω = rnew ≥ r₁ ? T(max(0.2, 0.9ω)) : T(min(1.0, 1.02ω))
        r₁ = rnew
        (r₁ < r₁tol && r∞ < r∞tol) && break
    end
    perBC!(top.x, top.op.g, p.per)
    push!(p.n, nᵖ)
end
function k_residual!(lo, hi, ps, pm, r, z, x, op::Op)
    I, ok = cellidx(lo, hi)
    s = 0f0
    if ok
        l = lin(op.g, I)
        v = getiD(op, l) == 0 ? 0f0 : @inbounds(z[l]) - mult(op, I, l, x)
        @inbounds r[l] = v
        s = v
    end
    block_partial!(ps, pm, s, 0f0)
    return nothing
end
function WaterLily.solver!(p::TPoisson; tol=2e-3, itmx=32)
    lv = p.levels[1]; g = lv.op.g
    red_launch(k_residual!, inside_lo(g), inside_hi(g), p.red, 1, lv.r, lv.z, lv.x, lv.op)
    solve!(p, 1; tol, itmx)
end

# ---------------------------------------------------------------- projection
# r = iD==0 ? 0 : (div u - δv·div V)/Δt - A p, partial Σr
function k_resid_src!(lo, hi, ps, pm, r, x, u, V, op::Op{D}, idt, useV::Bool) where D
    I, ok = cellidx(lo, hi)
    s = 0f0
    if ok
        g = op.g
        l = lin(g, I)
        v = 0f0
        if getiD(op, l) != 0
            du = 0f0; dv = 0f0
            @unroll for i in i32:Int32(D)
                ci = (i-i32)*g.nc
                @inbounds du += u[l+g.s[i]+ci] - u[l+ci]
                useV && (@inbounds dv += V[l+g.s[i]+ci] - V[l+ci])
            end
            z = useV ? du - (1 + (isreg(op,l) ? op.Dreg : @inbounds(op.Dg[l]))/(2D))*dv : du
            v = z*idt - mult(op, I, l, x)
        end
        @inbounds r[l] = v
        s = v
    end
    block_partial!(ps, pm, s, 0f0)
    return nothing
end
function k_proj!(lo, hi, u, x, op::Op{D}, dt) where D
    I, ok = cellidx(lo, hi); ok || return
    g = op.g
    l = lin(g, I)
    r = isreg(op, l)
    @unroll for i in i32:Int32(D)
        ci = (i-i32)*g.nc
        L = r ? op.Lreg : @inbounds(op.L[l+ci])
        @inbounds u[l+ci] -= dt*L*(x[l]-x[l-g.s[i]])
    end
    return nothing
end
function WaterLily.mom_project!(a::TFlow{D,T}, b::TPoisson, w, t) where {D,T}
    dt = T(w)*a.Δt[end]; g = a.grid; lv = b.levels[1]
    red_launch(k_resid_src!, inside_lo(g), inside_hi(g), b.red, 1, lv.r, lv.x, a.u, a.V, lv.op, inv(dt), !a.Vzero)
    solve!(b, dt)
    launch(k_proj!, inside_lo(g), inside_hi(g), a.u, b.x, lv.op, dt)
    BC!(a, a.u, t)
end
