using WaterLily: MultiLevelPoisson, Poisson, coarsen_mask

"""
    Level

One multigrid level. Arrays alias the corresponding `WaterLily.Poisson` level;
`r` and `ϵ` may be swapped by the fused Jacobi kernel.
"""
mutable struct Level{D,T,S<:AbstractArray{T},V<:AbstractArray{T}}
    L :: V
    D :: S
    iD :: S
    x :: S
    ϵ :: S
    r :: S
    z :: S
    g :: Grid{D}
    c :: NTuple{D,Bool}     # coarsening mask to the next level
end
Level(p::Poisson, c) = Level(p.L, p.D, p.iD, p.x, p.ϵ, p.r, p.z, Grid(p.x), c)

"""
    TPoisson

Multigrid pressure solver with the same algorithm as `WaterLily.MultiLevelPoisson`
(Jacobi pre-smoothing, red-black Gauss-Seidel post-smoothing, adaptive ω), but with
fused CUDA kernels and one host synchronisation per V-cycle.
"""
struct TPoisson{T,S,V,D} <: AbstractPoisson{T,S,V}
    x :: S
    L :: V
    z :: S
    ml :: MultiLevelPoisson{T,S,V}
    levels :: Vector{Level{D,T,S,V}}
    per :: NTuple{D,Bool}
    n :: Vector{Int16}
    perdir :: NTuple
    nsolve :: Base.RefValue{Int}     # number of solved rows (count(!iszero, iD))
    red :: RedBuf
end
function TPoisson(x::AbstractArray{T,D}, L, z; perdir=()) where {T,D}
    ml = MultiLevelPoisson(x, L, z; perdir)
    nl = length(ml.levels)
    levels = [Level(ml.levels[l], l < nl ? coarsen_mask(ml.levels[l].x, ml.levels[l+1].x) : ntuple(_->false,D)) for l in 1:nl]
    per = ntuple(j->j ∈ perdir, D)
    p = TPoisson{T,typeof(x),typeof(L),D}(x, L, z, ml, levels, per, Int16[], perdir, Ref(0), RedBuf())
    count_solved!(p)
    p
end
TPoisson(flow::AbstractFlow) = TPoisson(flow.p, flow.μ₀, flow.σ; perdir=flow.perdir)
count_solved!(p::TPoisson) = (p.nsolve[] = count(!iszero, p.levels[1].iD); p)
function WaterLily.update!(p::TPoisson)
    WaterLily.update!(p.ml)
    count_solved!(p)
end

# ---------------------------------------------------------------- operators
@inline function mult(I, l, g::Grid{D}, L, Dg, x, per) where D
    s = @inbounds x[l]*Dg[l]
    @unroll for j in i32:Int32(D)
        cj = (j-1)*g.nc
        @inbounds s += x[nbm(g,I,l,j,per)]*L[l+cj] + x[nbp(g,I,l,j,per)]*L[l+g.s[j]+cj]
    end
    s
end
# A applied to ϵ=r·iD computed on the fly
@inline function mult_riD(I, l, g::Grid{D}, L, Dg, r, iD, per) where D
    s = @inbounds r[l]*iD[l]*Dg[l]
    @unroll for j in i32:Int32(D)
        cj = (j-1)*g.nc
        lm, lp = nbm(g,I,l,j,per), nbp(g,I,l,j,per)
        @inbounds s += r[lm]*iD[lm]*L[l+cj] + r[lp]*iD[lp]*L[l+g.s[j]+cj]
    end
    s
end
# A applied to the prolongation of the coarse solution cx
@inline function down(I::NTuple{D,Int32}, c) where D
    ntuple(d-> c[d] ? (I[d]+Int32(2))÷Int32(2) : I[d], Val(D))
end
@inline function wrapI(g::Grid{D}, I, per) where D
    ntuple(Val(D)) do d
        (per[d] && I[d]==i32) ? g.n[d]-i32 : (per[d] && I[d]==g.n[d]) ? Int32(2) : I[d]
    end
end
@inline prol(g, I, per, cx, gc, c) = @inbounds cx[lin(gc, down(wrapI(g,I,per), c))]
@inline function mult_prol(I, l, g::Grid{D}, L, Dg, cx, gc, c, per) where D
    s = prol(g, I, per, cx, gc, c)*@inbounds(Dg[l])
    @unroll for j in i32:Int32(D)
        cj = (j-1)*g.nc
        @inbounds s += prol(g, shift(I,j,-1), per, cx, gc, c)*L[l+cj] +
                       prol(g, shift(I,j,1), per, cx, gc, c)*L[l+g.s[j]+cj]
    end
    s
end

# ---------------------------------------------------------------- kernels
# residual: r = iD==0 ? 0 : z-Ax ; accumulates Σr
function k_residual!(lo, hi, outs, outm, r, z, x, L, Dg, iD, g::Grid, per)
    I, ok = cellidx(lo, hi)
    s = 0f0
    if ok
        l = lin(g, I)
        v = @inbounds iD[l] == 0 ? 0f0 : z[l] - mult(I, l, g, L, Dg, x, per)
        @inbounds r[l] = v
        s = v
    end
    block_red_flat!(outs, outm, 1, s, 0f0)
    nothing
end
# global correction r -= s (only if |s|>2eps) and norms Σ|r|, max|r|
function k_fixres!(lo, hi, outs, outm, r, iD, g::Grid, nsolve, tiny)
    I, ok = cellidx(lo, hi)
    a = 0f0
    if ok
        l = lin(g, I)
        s = Float32(@inbounds(outs[1]) / nsolve)
        v = @inbounds r[l]
        if abs(s) > tiny
            v = @inbounds iD[l] == 0 ? 0f0 : v - s
            @inbounds r[l] = v
        end
        a = abs(v)
    end
    block_red_flat!(outs, outm, 2, a, a)
    nothing
end
# Jacobi: ϵ=r·iD, r ← r-Aϵ (written to rn), x += ϵ
function k_jacobi!(lo, hi, rn, r, x, L, Dg, iD, g::Grid, per)
    I, ok = cellidx(lo, hi); ok || return
    l = lin(g, I)
    @inbounds e = r[l]*iD[l]
    @inbounds rn[l] = r[l] - mult_riD(I, l, g, L, Dg, r, iD, per)
    @inbounds x[l] += e
    nothing
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
    nothing
end
# increment with ϵ = prolongation(cx): r -= ωAϵ, x += ωϵ
function k_prol_incr!(lo, hi, r, x, L, Dg, g::Grid, cx, gc::Grid, c, per, ω)
    I, ok = cellidx(lo, hi); ok || return
    l = lin(g, I)
    e = prol(g, I, per, cx, gc, c)
    @inbounds r[l] -= ω*mult_prol(I, l, g, L, Dg, cx, gc, c, per)
    @inbounds x[l] += ω*e
    nothing
end
# increment with stored ϵ (optionally accumulating norms of the new r)
function k_incr!(lo, hi, outs, outm, r, x, ϵ, L, Dg, g::Grid, per, ω, norms::Bool)
    I, ok = cellidx(lo, hi)
    a = 0f0
    if ok
        l = lin(g, I)
        v = @inbounds r[l] - ω*mult(I, l, g, L, Dg, ϵ, per)
        @inbounds r[l] = v
        @inbounds x[l] += ω*ϵ[l]
        a = abs(v)
    end
    norms && block_red_flat!(outs, outm, 2, a, a)
    nothing
end
# red-black Gauss-Seidel half sweep on ϵ (WaterLily.gauss_rb indexing over half_rangek).
# `init`: first sweep, also sets ϵ=r·iD on the partner cell and reads neighbours as r·iD.
@inline function gauss(I, l, g::Grid{D}, r, L, iD, x, per) where D
    s = @inbounds r[l]
    @unroll for j in i32:Int32(D)
        cj = (j-1)*g.nc
        @inbounds s -= x[nbm(g,I,l,j,per)]*L[l+cj] + x[nbp(g,I,l,j,per)]*L[l+g.s[j]+cj]
    end
    s*@inbounds(iD[l])
end
@inline function gauss_init(I, l, g::Grid{D}, r, L, iD, per) where D
    s = @inbounds r[l]
    @unroll for j in i32:Int32(D)
        cj = (j-1)*g.nc
        lm, lp = nbm(g,I,l,j,per), nbp(g,I,l,j,per)
        @inbounds s -= r[lm]*iD[lm]*L[l+cj] + r[lp]*iD[lp]*L[l+g.s[j]+cj]
    end
    s*@inbounds(iD[l])
end
function k_gs!(lo, hi, ϵ, r, L, iD, g::Grid{D}, per, k0::Int32, init::Bool) where D
    Iv, ok = cellidx(lo, hi); ok || return
    fs = Int32(0)
    @unroll for d in i32:Int32(D-1)
        fs += Iv[d]
    end
    p = (fs + k0) % Int32(2)
    k = 2Iv[D] - i32 - p
    I = setj(Iv, D, k)
    l = lin(g, I)
    if init
        # partner cell in the pair {2kk-2, 2kk-1}
        kp = 2Iv[D] - i32 - (i32 - p)
        if kp <= g.n[D] - i32
            lp = lin(g, setj(Iv, D, kp))
            @inbounds ϵ[lp] = r[lp]*iD[lp]
        end
        # odd size: the last interior cell is outside the red-black range
        if 2Iv[D] == g.n[D] - i32
            lx = lin(g, setj(Iv, D, g.n[D] - i32))
            @inbounds ϵ[lx] = r[lx]*iD[lx]
        end
        @inbounds ϵ[l] = gauss_init(I, l, g, r, L, iD, per)
    else
        @inbounds ϵ[l] = gauss(I, l, g, r, L, iD, ϵ, per)
    end
    nothing
end
# perBC! for a scalar field
function k_perbc!(lo, hi, a, g::Grid, j)
    I, ok = cellidx(lo, hi); ok || return
    l1 = lin(g, I); sj = g.s[j]; nj = g.n[j]
    @inbounds a[l1] = a[l1 + (nj-Int32(2))*sj]
    @inbounds a[l1 + (nj-i32)*sj] = a[l1 + sj]
    nothing
end
function perBC!(a, g::Grid{D}, per) where D
    for j in 1:D
        per[j] || continue
        launch(k_perbc!, ntuple(_->1,D), ntuple(d-> d==j ? 1 : Int(g.n[d]), D), a, g, j)
    end
end

# ---------------------------------------------------------------- solver pieces
function residual!(p::TPoisson{T}) where T
    lv = p.levels[1]; g = lv.g; b = reset!(p.red)
    launch(k_residual!, inside_lo(g), inside_hi(g), b.s, b.m, lv.r, lv.z, lv.x, lv.L, lv.D, lv.iD, g, p.per)
    launch(k_fixres!, inside_lo(g), inside_hi(g), b.s, b.m, lv.r, lv.iD, g, Float64(p.nsolve[]), 2eps(T))
end
function jacobi!(lv::Level, per)
    g = lv.g
    launch(k_jacobi!, inside_lo(g), inside_hi(g), lv.ϵ, lv.r, lv.x, lv.L, lv.D, lv.iD, g, per)
    lv.r, lv.ϵ = lv.ϵ, lv.r
end
gs_range(g::Grid{D}) where D = (inside_lo(g), ntuple(d-> d==D ? Int(g.n[d])÷2 : Int(g.n[d])-1, D))
function smooth!(lv::Level{D,T}, per, ω, red::RedBuf, norms::Bool=false; it=4) where {D,T}
    g = lv.g
    lo, hi = gs_range(g)
    for i in 1:it
        launch(k_gs!, lo, hi, lv.ϵ, lv.r, lv.L, lv.iD, g, per, Int32(i), i==1)
    end
    launch(k_incr!, inside_lo(g), inside_hi(g), red.s, red.m, lv.r, lv.x, lv.ϵ, lv.L, lv.D, g, per, T(ω), norms)
end
function vcycle!(p::TPoisson{T}, l, ω) where T
    fine, coarse = p.levels[l], p.levels[l+1]
    jacobi!(fine, p.per)
    launch(k_restrict!, inside_lo(coarse.g), inside_hi(coarse.g), coarse.r, coarse.g, fine.r, fine.g, fine.c)
    fill!(coarse.x, 0)
    l+1 < length(p.levels) && vcycle!(p, l+1, ω)
    smooth!(coarse, p.per, ω, p.red)
    g = fine.g
    launch(k_prol_incr!, inside_lo(g), inside_hi(g), fine.r, fine.x, fine.L, fine.D, g, coarse.x, coarse.g, fine.c, p.per, T(ω))
end
function norms!(b::RedBuf)
    fetch!(b)
    getsum(b,2), getmax(b,2)
end

function WaterLily.solver!(p::TPoisson{T}; tol=2e-3, itmx=32) where T
    top = p.levels[1]
    r₁tol = WaterLily.l1n_tol(p.ml.levels[1], tol); r∞tol = tol
    residual!(p); r₁, r∞ = norms!(p.red); ω = T(1)
    nᵖ = 0; @log ", $nᵖ, $r∞, $r₁, $ω\n"
    while nᵖ < itmx
        vcycle!(p, 1, ω)
        b = reset!(p.red)
        smooth!(top, p.per, ω, b, true)
        rnew, r∞ = norms!(b); nᵖ += 1
        @log ", $nᵖ, $r∞, $rnew, $ω\n"
        ω = rnew ≥ r₁ ? T(max(0.2, 0.9ω)) : T(min(1.0, 1.02ω))
        r₁ = rnew
        (r₁ < r₁tol && r∞ < r∞tol) && break
    end
    perBC!(top.x, top.g, p.per)
    push!(p.n, nᵖ)
end

# ---------------------------------------------------------------- projection
function k_src!(lo, hi, z, x, u, V, Dg, g::Grid{D}, dt, useV::Bool) where D
    I, ok = cellidx(lo, hi); ok || return
    l = lin(g, I)
    @inbounds x[l] *= dt
    all(ntuple(d->I[d]>i32 && I[d]<g.n[d], Val(D))) || return
    du = 0f0; dv = 0f0
    @unroll for i in i32:Int32(D)
        ci = (i-1)*g.nc
        @inbounds du += u[l+g.s[i]+ci] - u[l+ci]
        useV && (@inbounds dv += V[l+g.s[i]+ci] - V[l+ci])
    end
    @inbounds z[l] = useV ? du - (1 + Dg[l]/(2D))*dv : du
    nothing
end
function k_proj!(lo, hi, u, L, x, g::Grid{D}) where D
    I, ok = cellidx(lo, hi); ok || return
    l = lin(g, I)
    @unroll for i in i32:Int32(D)
        ci = (i-1)*g.nc
        @inbounds u[l+ci] -= L[l+ci]*(x[l]-x[l-g.s[i]])
    end
    nothing
end
function WaterLily.mom_project!(a::TFlow{D,T}, b::TPoisson, w, t) where {D,T}
    dt = T(w)*a.Δt[end]; g = a.grid
    launch(k_src!, ntuple(_->1,D), size(a.p), b.z, b.x, a.u, a.V, b.levels[1].D, g, dt, !a.Vzero)
    WaterLily.solver!(b)
    launch(k_proj!, inside_lo(g), inside_hi(g), a.u, b.L, b.x, g)
    b.x ./= dt
    BC!(a, a.u, t)
end
