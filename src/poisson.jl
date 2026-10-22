using WaterLily: MultiLevelPoisson, Poisson, coarsen_mask
using Adapt

"""
    Op

Kernel-side view of one Poisson level `A = L + D + L'`. Cells with bit `0x01` of `reg[I]` have
`L=Lreg` on all 2d faces, so `D=Dreg` and `iD=iDreg`; the solver kernels then skip
reading `L`, `D` and `iD`, which dominate the memory traffic away from the body. Bit `0x02`
marks a regular cell whose 2d neighbours are regular too, so a stencil over `r·iD` needs no
flag of a neighbour.
"""
struct Op{D,T,S<:AbstractArray{T},V<:AbstractArray{T},R<:AbstractArray{UInt8},P<:Per}
    L :: V
    Dg :: S
    iD :: S
    reg :: R
    Lreg :: T
    Dreg :: T
    iDreg :: T
    g :: Grid{D}
    per :: P
end
Adapt.@adapt_structure Op

@inline isreg(op::Op, l) = @inbounds (op.reg[l] & 0x01) != 0x00
@inline isdeep(op::Op, l) = @inbounds (op.reg[l] & 0x02) != 0x00
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
mutable struct Level{D,T,S<:AbstractArray{T},V<:AbstractArray{T},R,P}
    op :: Op{D,T,S,V,R,P}
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
# bit 0x02: the cell and its 2d neighbours (wrapped in periodic directions) are regular.
# Only bit 0x02 is written, so concurrent reads of bit 0x01 are unaffected.
function k_deepflag!(lo, hi, reg, g::Grid{D}, per) where D
    I, ok = cellidx(lo, hi); ok || return
    l = lin(g, I)
    b = @inbounds(reg[l]) & 0x01
    @unroll for j in i32:Int32(D)
        b &= @inbounds(reg[nbm(g,I,l,j,per)]) & @inbounds(reg[nbp(g,I,l,j,per)])
    end
    b != 0x00 && (@inbounds reg[l] = 0x03)
    return nothing
end
set_deep!(op::Op) = launch(k_deepflag!, inside_lo(op.g), inside_hi(op.g), op.reg, op.g, op.per)
"Recompute the regular-cell flags from `L` (after `L` changed)."
function set_reg!(lv::Level{D,T}) where {D,T}
    op = lv.op; g = op.g
    Lreg = maximum(op.L)
    fill!(op.reg, 0x00)
    launch(k_regflag!, inside_lo(g), inside_hi(g), op.reg, op.L, g, Lreg)
    set_deep!(op)
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
struct TPoisson{T,S,V,D,R,P} <: AbstractPoisson{T,S,V}
    x :: S
    L :: V
    z :: S
    ml :: MultiLevelPoisson{T,S,V}
    levels :: Vector{Level{D,T,S,V,R,P}}
    per :: P
    n :: Vector{Int16}
    perdir :: NTuple
    nsolve :: Base.RefValue{Int}     # number of solved rows (count(!iszero, iD))
    red :: RedBuf
    ωd :: CuVector{Float32}          # [ω, r₁]: relaxation and previous Σ|r| (device side)
    fixd :: CuVector{Float32}        # [s, tiny] for the lazy residual correction, s = Σr/nsolve
    nsd :: CuVector{Float64}         # number of solved rows, count(iD≠0), on the device
    graphs :: Dict{Any,CuGraphExec}
    seen :: Set{Any}
end
function TPoisson(x::AbstractArray{T,D}, L, z; perdir=()) where {T,D}
    ml = MultiLevelPoisson(x, L, z; perdir)
    nl = length(ml.levels)
    per = Per(ntuple(j->j ∈ perdir, D))
    levels = [Level(ml.levels[l], l < nl ? coarsen_mask(ml.levels[l].x, ml.levels[l+1].x) : ntuple(_->false,D), per) for l in 1:nl]
    p = TPoisson{T,typeof(x),typeof(L),D,typeof(levels[1].op.reg),typeof(per)}(x, L, z, ml, levels, per, Int16[], perdir, Ref(0), RedBuf(Grid(x)),
        CUDA.ones(Float32, 2), CUDA.zeros(Float32, 2), CUDA.zeros(Float64, 1), Dict{Any,CuGraphExec}(), Set{Any}())
    finish_update!(p)
end
TPoisson(flow::AbstractFlow) = TPoisson(flow.p, flow.μ₀, flow.σ; perdir=flow.perdir)
function finish_update!(p::TPoisson)
    foreach(set_reg!, p.levels)       # sets Lreg per level (the fluid value of L)
    return WaterLily.update!(p)
end

# ---------------------------------------------------------------- kernels
# Jacobi: ϵ=r·iD, r ← r-Aϵ (written to rn), x += ϵ
struct RiD{S,O}; r::S; op::O; end
@inline (f::RiD)(I, l) = @inbounds(f.r[l])*getiD(f.op, l)
"`RiD` around a cell flagged `isdeep`, where every iD is iDreg."
struct RiDc{S,T}; r::S; iD::T; end
@inline (f::RiDc)(I, l) = @inbounds(f.r[l])*f.iD
function k_jacobi!(lo, hi, rn, r, x, op::Op, assign::Bool, fr)
    I, ok = cellidx(lo, hi)
    w = 0f0
    if ok
        l = lin(op.g, I)
        if isdeep(op, l)
            f = RiDc(r, op.iDreg)
            e = f(I, l)
            w = @inbounds(r[l]) - mult_f(op, I, l, f)
        else
            f = RiD(r, op)
            e = f(I, l)
            w = @inbounds(r[l]) - mult_f(op, I, l, f)
        end
        @inbounds rn[l] = w
        @inbounds x[l] = assign ? e : x[l] + e
    end
    restrict_block!(fr, I, ok, w)
    return nothing
end
"""
    FusedR(rc, cx, gc)

Restriction fused into the fine Jacobi kernel (`fr` argument; `nothing` = separate
`k_restrict!`). A `launch` block (32×4×2 or 32×8) starting at the even index 2 holds whole
2ᵈ coarsening blocks, so the block sums its new residuals in shared memory, in the order of
`k_restrict!`, writes the coarse residual and zeroes the coarse solution.
"""
struct FusedR{D,S}; rc::S; cx::S; gc::Grid{D}; end
Adapt.@adapt_structure FusedR
@inline restrict_block!(::Nothing, I, ok, w) = nothing
@inline function restrict_block!(fr::FusedR{D}, I, ok, w) where D
    sh = CuStaticSharedArray(Float32, 256)
    t = flat_tid()
    @inbounds sh[t+i32] = ok ? w : 0f0
    sync_threads()
    lead = ok
    @unroll for d in i32:Int32(D)
        lead &= iseven(I[d])
    end
    if lead
        bx = blockDim().x; bxy = blockDim().x*blockDim().y
        s = 0f0
        if D == 3
            for dz in Int32(0):i32, dy in Int32(0):i32, dx in Int32(0):i32
                @inbounds s += sh[t + i32 + dx + dy*bx + dz*bxy]
            end
        else
            for dy in Int32(0):i32, dx in Int32(0):i32
                @inbounds s += sh[t + i32 + dx + dy*bx]
            end
        end
        lc = lin(fr.gc, ntuple(d-> (I[d]+Int32(2)) >> 1, Val(D)))
        @inbounds fr.rc[lc] = s
        @inbounds fr.cx[lc] = 0f0
    end
    return nothing
end
"`FusedR` for the restriction from `fine` to `coarse`, or `nothing` if a block can't hold whole coarse cells."
function fused_restrict(fine::Level{D}, coarse::Level{D}) where D
    gf, gc = fine.op.g, coarse.op.g
    ok = all(fine.c) && all(d-> gf.n[d]-2 == 2*(gc.n[d]-2), 1:D) && D in (2, 3)
    return ok ? FusedR(coarse.r, coarse.x, gc) : nothing
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
struct Prol{D,S,C,P}; cx::S; gc::Grid{D}; g::Grid{D}; c::C; per::P; end
@inline (f::Prol)(I, l) = @inbounds f.cx[lin(f.gc, down(wrapI(f.g, I, f.per), f.c))]
# increment with ϵ = prolongation(cx): r -= ωAϵ, x += ωϵ
function k_prol_incr!(lo, hi, r, x, op::Op, cx, gc::Grid, c, ωd)
    I, ok = cellidx(lo, hi); ok || return
    ω = @inbounds ωd[1]
    l = lin(op.g, I)
    f = Prol(cx, gc, op.g, c, op.per)
    e = f(I, l)
    @inbounds r[l] -= ω*mult_f(op, I, l, f)
    @inbounds x[l] += ω*e
    return nothing
end
# increment with stored ϵ (optionally the block partials of Σ|r|, max|r|)
# increment with stored ϵ (optionally the block partials of Σ|r| and max|r|; with `exclb`
# the max skips the cells next to the domain boundary, which a Biot-Savart update changes)
function k_incr!(lo, hi, ps, pm, r, x, ϵ, op::Op{D}, ωd, ::Val{norms}, ::Val{exclb}) where {D, norms, exclb}
    I, ok = cellidx(lo, hi)
    ω = @inbounds ωd[1]
    a = 0f0; m = 0f0
    if ok
        l = lin(op.g, I)
        v = @inbounds(r[l]) - ω*mult(op, I, l, ϵ)
        @inbounds r[l] = v
        @inbounds x[l] += ω*ϵ[l]
        a = abs(v)
        edge = false
        if exclb
            @unroll for d in i32:Int32(D)
                edge |= (I[d] == Int32(2)) | (I[d] == op.g.n[d] - i32)
            end
        end
        m = edge ? 0f0 : a
    end
    norms && block_partial!(ps, pm, a, m)
    return nothing
end
# Red-black Gauss-Seidel half sweep on ϵ, with pairs along the last direction.
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
    end
    k <= g.n[D] - i32 || return             # the last pair can have only one interior cell
    if init
        if isdeep(op, l)
            @inbounds ϵ[l] = (r[l] - offdiag_f(op, I, l, RiDc(r, op.iDreg)))*op.iDreg
        else
            @inbounds ϵ[l] = (r[l] - offdiag_f(op, I, l, RiD(r, op)))*getiD(op, l)
        end
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
"`RcD` around a cell flagged `isdeep`: every iD is iDreg ≠ 0."
struct RcDc{S,T}; r::S; s::T; apply::Bool; iD::T; end
@inline (f::RcDc)(I, l) = (f.apply ? @inbounds(f.r[l]) - f.s : @inbounds(f.r[l]))*f.iD
@inline function make_rc(r, op, fixd)
    s = @inbounds fixd[1]
    return Rc(r, op, s, abs(s) > @inbounds(fixd[2]))
end
# s = Σr/nsolve, once per solve (a Float64 division per thread is slow on GeForce cards)
function k_mkfix!(fixd, sums, nsd)
    @inbounds fixd[1] = Float32(sums[1] / nsd[1])
    return nothing
end
# Jacobi on the corrected residual; partial Σ|rc|, max|rc| (the initial residual norms)
function k_jacobi_fix!(lo, hi, ps, pm, rn, r, x, op::Op, fixd, fr)
    I, ok = cellidx(lo, hi)
    a = 0f0; w = 0f0
    if ok
        l = lin(op.g, I)
        rc = make_rc(r, op, fixd)
        v = rc(I, l)
        if isdeep(op, l)
            w = v - mult_f(op, I, l, RcDc(r, rc.s, rc.apply, op.iDreg))
            @inbounds x[l] += v*op.iDreg
        else
            w = v - mult_f(op, I, l, RcD(rc))
            @inbounds x[l] += v*getiD(op, l)
        end
        @inbounds rn[l] = w
        a = abs(v)
    end
    restrict_block!(fr, I, ok, w)
    block_partial!(ps, pm, a, a)
    return nothing
end

# ---------------------------------------------------------------- solver pieces
function jacobi!(lv::Level, assign::Bool=false, fr=nothing)
    g = lv.op.g
    launch(k_jacobi!, inside_lo(g), inside_hi(g), lv.ϵ, lv.r, lv.x, lv.op, assign, fr)
    lv.r, lv.ϵ = lv.ϵ, lv.r
end
gs_range(g::Grid{D}) where D = (inside_lo(g), ntuple(d-> d==D ? cld(Int(g.n[d]),2) : Int(g.n[d])-1, D))
function k_rid!(lo, hi, ϵ, r, op::Op)
    I, ok = cellidx(lo, hi); ok || return
    l = lin(op.g, I)
    @inbounds ϵ[l] = r[l]*getiD(op, l)
    return nothing
end
# An odd periodic length puts the first and last interior cells, which are neighbours across
# the wrap, in the same red-black colour. Reading the wrapped neighbour would then race with
# its update, so such levels read the periodic ghosts, filled once before the sweeps
# (as WaterLily.GaussSeidelRB!).
oddper(op::Op{D}) where D = any(d-> op.per[d] && isodd(op.g.n[d]), 1:D)
noper(op::Op{D}) where D = Op(op.L, op.Dg, op.iD, op.reg, op.Lreg, op.Dreg, op.iDreg, op.g, Per(ntuple(_->false, D)))
function smooth!(lv::Level{D,T}, ωd, red::RedBuf, norms::Bool=false; it=4, exclb=false) where {D,T}
    g = lv.op.g
    lo, hi = gs_range(g)
    if oddper(lv.op)
        launch(k_rid!, inside_lo(g), inside_hi(g), lv.ϵ, lv.r, lv.op)
        perBC!(lv.ϵ, g, lv.op.per)
        op = noper(lv.op)
        for i in 1:it
            launch(k_gs!, lo, hi, lv.ϵ, lv.r, op, Int32(i), false)
        end
    else
        for i in 1:it
            launch(k_gs!, lo, hi, lv.ϵ, lv.r, lv.op, Int32(i), i==1)
        end
    end
    if norms
        red_launch(k_incr!, inside_lo(g), inside_hi(g), red, 3, lv.r, lv.x, lv.ϵ, lv.op, ωd, Val(true), Val(exclb))
    else
        launch(k_incr!, inside_lo(g), inside_hi(g), red.ps, red.pm, lv.r, lv.x, lv.ϵ, lv.op, ωd, Val(false), Val(false))
    end
end
# V-cycle below level l; `fix` = (nsolve, tiny): the fine residual still needs the
# global correction, which is applied inside the fine Jacobi (norms into slot 2)
function vcycle!(p::TPoisson{T}, l, fix=nothing; assign=false) where T
    fine, coarse = p.levels[l], p.levels[l+1]
    fr = fused_restrict(fine, coarse)
    if fix === nothing
        jacobi!(fine, assign, fr)
    else
        g = fine.op.g
        @cuda k_mkfix!(p.fixd, p.red.s, p.nsd)
        red_launch(k_jacobi_fix!, inside_lo(g), inside_hi(g), p.red, 2, fine.ϵ, fine.r, fine.x, fine.op, p.fixd, fr)
        fine.r, fine.ϵ = fine.ϵ, fine.r
    end
    gc = coarse.op.g
    if fr === nothing
        launch(k_restrict!, inside_lo(gc), inside_hi(gc), coarse.r, gc, fine.r, fine.op.g, fine.c)
        fill!(coarse.x, 0)
    end
    l+1 < length(p.levels) && vcycle!(p, l+1)
    smooth!(coarse, p.ωd, p.red)
    g = fine.op.g
    launch(k_prol_incr!, inside_lo(g), inside_hi(g), fine.r, fine.x, fine.op, coarse.x, gc, fine.c, p.ωd)
end

# ω update on the device (WaterLily.solver! rule); r₁ comes from slot 2 on the first iteration
function k_omega!(ωd, sums, first::Bool)
    ω = @inbounds ωd[1]
    r₁ = first ? Float32(@inbounds(sums[2])) : @inbounds(ωd[2])
    rnew = Float32(@inbounds(sums[3]))
    @inbounds ωd[1] = rnew ≥ r₁ ? max(0.2f0, 0.9f0ω) : min(1f0, 1.02f0ω)
    @inbounds ωd[2] = rnew
    return nothing
end
swap_levels!(p::TPoisson) = for l in 1:length(p.levels)-1
    lv = p.levels[l]; lv.r, lv.ϵ = lv.ϵ, lv.r
end
"""
    graphed!(f, p, key)

Run `f()` (a fixed sequence of launches that swaps the level buffers like one V-cycle)
through a cached CUDA graph: plain execution the first time a (key, buffer state) is
seen, capture on the second, graph replay afterwards.
"""
# (internal to CUDA.jl, looked up defensively)
const CUFUNCTION_LOCK = isdefined(CUDA, :cufunction_lock) ? CUDA.cufunction_lock :
    isdefined(CUDA, :CUDACore) && isdefined(CUDA.CUDACore, :cufunction_lock) ?
    CUDA.CUDACore.cufunction_lock : ReentrantLock()
function graphed!(f::F, p::TPoisson, key) where F
    k = (key, objectid(p.levels[1].r))
    ex = get(p.graphs, k, nothing)
    if ex !== nothing
        CUDA.launch(ex); swap_levels!(p)
    elseif k in p.seen
        state = [(lv.r, lv.ϵ) for lv in p.levels]
        # Capture on a private non-blocking stream in relaxed mode, so that CUDA calls of
        # other tasks (possibly scheduled on this thread, possibly on the legacy stream)
        # cannot touch or depend on the capturing stream. Nothing runs during a capture;
        # the graph is launched on the task's stream. No GC during the capture: a
        # finalizer freeing GPU memory would be captured too.
        # CUDA.jl's kernel compile/load lock is held too: loading a module while a stream
        # is capturing fails, and CUDA.jl's retry synchronizes the device.
        cs = CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
        gc = GC.enable(false)
        g = try
            lock(CUFUNCTION_LOCK) do
                CUDA.stream!(cs) do
                    CUDA.capture(f; flags=CUDA.STREAM_CAPTURE_MODE_RELAXED, throw_error=false)
                end
            end
        catch
            nothing
        finally
            GC.enable(gc)
        end
        if g === nothing   # capture failed: restore the buffers and run normally
            for (lv, (r, ϵ)) in zip(p.levels, state); lv.r, lv.ϵ = r, ϵ; end
            f()
        else
            ex = CUDA.instantiate(g)
            p.graphs[k] = ex
            CUDA.launch(ex)
        end
    else
        push!(p.seen, k); f()
    end
    return nothing
end
# one multigrid iteration: V-cycle, fine smoothing with norms (Σ|r|, max|r|) into slot 3, ω update
function iteration!(p::TPoisson; fix=false, assign=false, exclb=false, adapt=true)
    graphed!(p, (fix, assign, exclb, adapt)) do
        vcycle!(p, 1, fix ? true : nothing; assign)
        smooth!(p.levels[1], p.ωd, p.red, true; exclb)
        adapt && @cuda k_omega!(p.ωd, p.red.s, fix)
    end
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
    copyto!(p.fixd, Float32[0, 2eps(T)/scale])
    fill!(p.ωd, 1)
    nᵖ = 0
    while nᵖ < itmx
        iteration!(p; fix = nᵖ == 0)
        fetch!(p.red)
        nᵖ == 0 && @log ", 0, $(getmax(p.red,2)*scale), $(getsum(p.red,2)*scale)\n"
        rnew, r∞ = getsum(p.red,3), getmax(p.red,3); nᵖ += 1
        @log ", $nᵖ, $(r∞*scale), $(rnew*scale)\n"
        (rnew < r₁tol && r∞ < r∞tol) && break
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
# (WaterLily ≥ 1.9.1: domain BCs at t and the convective exit (predictor) are applied here,
# and the step is Δt/w with w=1 in the predictor and w=2 in the corrector)
function WaterLily.mom_project!(a::TFlow{D,T}, b::TPoisson, w::Int, t) where {D,T}
    BC!(a, a.u, t)
    w == 1 && a.exitBC && exitBC!(a, a.u, a.u⁰, a.Δt[end])
    dt = a.Δt[end]/T(w); g = a.grid; lv = b.levels[1]
    red_launch(k_resid_src!, inside_lo(g), inside_hi(g), b.red, 1, lv.r, lv.x, a.u, a.V, lv.op, inv(dt), !a.Vzero)
    solve!(b, dt)
    launch(k_proj!, inside_lo(g), inside_hi(g), a.u, b.x, lv.op, dt)
    BC!(a, a.u, t)
end
