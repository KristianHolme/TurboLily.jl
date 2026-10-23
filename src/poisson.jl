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

"""
    Pre(f, op, I, l)

The values of the neighbour functor `f` at cell `l` and its 2d neighbours (wrapped as
`nbm`/`nbp`), loaded up front. A kernel that branches on a cell flag before `mult_f` would
otherwise issue the stencil loads only after the flag load. A neighbour is identified from
its index tuple, which folds to a constant once the stencil loop is unrolled.
"""
struct Pre{D,T}; I::NTuple{D,Int32}; c::T; m::NTuple{D,T}; p::NTuple{D,T}; end
@inline function (f::Pre{3})(Ip, lp)
    Ip[1] != f.I[1] && return Ip[1] < f.I[1] ? f.m[1] : f.p[1]
    Ip[2] != f.I[2] && return Ip[2] < f.I[2] ? f.m[2] : f.p[2]
    Ip[3] != f.I[3] && return Ip[3] < f.I[3] ? f.m[3] : f.p[3]
    return f.c
end
@inline function (f::Pre{2})(Ip, lp)
    Ip[1] != f.I[1] && return Ip[1] < f.I[1] ? f.m[1] : f.p[1]
    Ip[2] != f.I[2] && return Ip[2] < f.I[2] ? f.m[2] : f.p[2]
    return f.c
end
@inline function Pre(f::F, op::Op{3}, I, l) where F
    g, per = op.g, op.per
    m = (f(shift(I,1,-1), nbm(g,I,l,1,per)), f(shift(I,2,-1), nbm(g,I,l,2,per)), f(shift(I,3,-1), nbm(g,I,l,3,per)))
    p = (f(shift(I,1,1), nbp(g,I,l,1,per)), f(shift(I,2,1), nbp(g,I,l,2,per)), f(shift(I,3,1), nbp(g,I,l,3,per)))
    return Pre(I, f(I, l), m, p)
end
@inline function Pre(f::F, op::Op{2}, I, l) where F
    g, per = op.g, op.per
    m = (f(shift(I,1,-1), nbm(g,I,l,1,per)), f(shift(I,2,-1), nbm(g,I,l,2,per)))
    p = (f(shift(I,1,1), nbp(g,I,l,1,per)), f(shift(I,2,1), nbp(g,I,l,2,per)))
    return Pre(I, f(I, l), m, p)
end
"Preloaded values times a constant (r·iDreg around a cell flagged `isdeep`)."
struct Scaled{P,T}; p::P; s::T; end
@inline (f::Scaled)(Ip, lp) = f.p(Ip, lp)*f.s
"Preloaded values, corrected and scaled ((r-s)·iDreg around a cell flagged `isdeep`)."
struct ShiftScaled{P,T}; p::P; s::T; apply::Bool; iD::T; end
@inline (f::ShiftScaled)(Ip, lp) = (f.apply ? f.p(Ip, lp) - f.s : f.p(Ip, lp))*f.iD

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
# Jacobi at cell I: ϵ=r·iD, new residual r-Aϵ into rn (returned), x += ϵ
@inline function jacobi_cell!(I, rn, r, x, op::Op, assign::Bool)
    l = lin(op.g, I)
    pr = Pre(AV(r), op, I, l)
    if isdeep(op, l)
        f = Scaled(pr, op.iDreg)
        e = f(I, l)
        w = pr.c - mult_f(op, I, l, f)
    else
        f = RiD(r, op)
        e = f(I, l)
        w = @inbounds(r[l]) - mult_f(op, I, l, f)
    end
    @inbounds rn[l] = w
    @inbounds x[l] = assign ? e : x[l] + e
    return w
end
function k_jacobi!(lo, hi, rn, r, x, op::Op, assign::Bool, fr)
    I, ok = cellidx(lo, hi)
    w = ok ? jacobi_cell!(I, rn, r, x, op, assign) : 0f0
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
    restrict_cell!(I, rc, gc, rf, gf, c)
    return nothing
end
@inline function restrict_cell!(I, rc, gc::Grid{D}, rf, gf::Grid{D}, c) where D
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
    prol_cell!(I, r, x, op, cx, gc, c, @inbounds(ωd[1]))
    return nothing
end
@inline function prol_cell!(I, r, x, op::Op, cx, gc::Grid, c, ω)
    l = lin(op.g, I)
    f = Pre(Prol(cx, gc, op.g, c, op.per), op, I, l)
    e = f.c
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
        v = incr_cell!(I, r, x, ϵ, op, ω)
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
@inline function incr_cell!(I, r, x, ϵ, op::Op, ω)
    l = lin(op.g, I)
    pe = Pre(AV(ϵ), op, I, l)
    v = @inbounds(r[l]) - ω*mult_f(op, I, l, pe)
    @inbounds r[l] = v
    @inbounds x[l] += ω*pe.c
    return v
end
# Red-black Gauss-Seidel half sweep on ϵ, with pairs along the last direction.
# `init`: first sweep, also sets ϵ=r·iD on the partner cell and reads neighbours as r·iD.
function k_gs!(lo, hi, ϵ, r, op::Op{D}, k0::Int32, init::Bool) where D
    Iv, ok = cellidx(lo, hi); ok || return
    gs_cell!(Iv, ϵ, r, op, k0, init)
    return nothing
end
@inline function gs_cell!(Iv, ϵ, r, op::Op{D}, k0::Int32, init::Bool) where D
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
        pr = Pre(AV(r), op, I, l)
        if isdeep(op, l)
            @inbounds ϵ[l] = (pr.c - offdiag_f(op, I, l, Scaled(pr, op.iDreg)))*op.iDreg
        else
            @inbounds ϵ[l] = (r[l] - offdiag_f(op, I, l, RiD(r, op)))*getiD(op, l)
        end
    else
        pe = Pre(AV(ϵ), op, I, l)
        @inbounds ϵ[l] = (r[l] - offdiag_f(op, I, l, pe))*getiD(op, l)
    end
    return nothing
end
# ---------------------------------------------------------------- fused Gauss-Seidel pair
# Half sweeps k0 (red) and k0+1 (black) in one pass, on 3D non-periodic levels: blocks of
# 32×8 threads own an x-y tile with a one-cell halo and march a chunk of z planes, updating
# red at plane z and black at plane z-1 (in-plane neighbours in shared memory, the column in
# registers, the next planes prefetched). ϵ is read from `ϵi` and written to `ϵo`, so blocks
# never see each other's updates. offdiag_f is called with the neighbour values, so the
# result equals two k_gs! launches. In a periodic direction (even length) the halo column or
# plane at index 1 or n stands for the wrapped cell n-1 or 2, which nbm/nbp would read.
const GS2_TX, GS2_TY, GS2_CH = 32, 8, 16
const GS2_CELLS = 2^18
"Neighbour values of cell I for offdiag_f, identified by the neighbour's index tuple."
struct NBV{T}; I::NTuple{3,Int32}; xm::T; xp::T; ym::T; yp::T; zm::T; zp::T; end
@inline (f::NBV)(Ip, lp) = Ip[1] != f.I[1] ? (Ip[1] < f.I[1] ? f.xm : f.xp) :
                           Ip[2] != f.I[2] ? (Ip[2] < f.I[2] ? f.ym : f.yp) : (Ip[3] < f.I[3] ? f.zm : f.zp)
"A one-element stand-in for the regular flags: the prefetched flag of the cell being updated."
struct Flag <: AbstractVector{UInt8}; v::UInt8; end
@inline Base.getindex(f::Flag, i::Integer) = f.v
Base.size(::Flag) = (1,)
@inline withflag(op::Op, f::UInt8) = Op(op.L, op.Dg, op.iD, Flag(f), op.Lreg, op.Dreg, op.iDreg, op.g, op.per)
# index k in direction j as the cell it stands for (wrapped in a periodic direction)
@inline wrapk(k, n, per::Bool) = per ? (k < Int32(2) ? k + n - Int32(2) : k > n - i32 ? k - n + Int32(2) : k) : k
@inline function gs2_ld(ϵi, r, op, act, lc, z, n3, sz)
    z = wrapk(z, n3, op.per[3])
    ok = act & (z >= i32) & (z <= n3)
    l = lc + (z - i32)*sz
    return (ok ? @inbounds(r[l]) : 0f0), (ok ? @inbounds(op.reg[l]) : 0x00), (ok ? @inbounds(ϵi[l]) : 0f0)
end
function k_gs2!(ϵi, ϵo, r, op::Op{3}, k0::Int32)
    TX, TY, CH = Int32(GS2_TX), Int32(GS2_TY), Int32(GS2_CH)
    g = op.g; n1, n2, n3 = g.n; sy, sz = g.s[2], g.s[3]
    Sin = CuStaticSharedArray(Float32, (GS2_TX, GS2_TY))
    SE = CuStaticSharedArray(Float32, (GS2_TX, GS2_TY, 2))
    tx = threadIdx().x; ty = threadIdx().y
    x = (blockIdx().x - i32)*(TX - Int32(2)) + tx          # tx=1 is the halo column x0-1
    y = (blockIdx().y - i32)*(TY - Int32(2)) + ty
    zc0 = Int32(2) + (blockIdx().z - i32)*CH
    zc1 = min(zc0 + CH - i32, n3 - i32)
    act = (x <= n1) & (y <= n2)
    outt = act & (x >= Int32(2)) & (x <= n1 - i32) & (y >= Int32(2)) & (y <= n2 - i32) &
           (tx > i32) & (tx < TX) & (ty > i32) & (ty < TY)
    # the last active column or row (index n) has its upper neighbour outside the tile too
    xl = tx == i32; xr = (tx == TX) | (x == n1)
    yl = ty == i32; yr = (ty == TY) | (y == n2)
    x = wrapk(x, n1, op.per[1]); y = wrapk(y, n2, op.per[2])     # the cell this column stands for
    inxy = act & (x >= Int32(2)) & (x <= n1 - i32) & (y >= Int32(2)) & (y <= n2 - i32)
    # offset of the neighbour outside the tile (edge threads), wrapped as nbm/nbp
    ho = xl ? (op.per[1] & (x == Int32(2)) ? (n1-Int32(3)) : -i32) :
         xr ? (op.per[1] & (x == n1 - i32) ? -(n1-Int32(3)) : i32) : Int32(0)
    vo = yl ? (op.per[2] & (y == Int32(2)) ? (n2-Int32(3))*sy : -sy) :
         yr ? (op.per[2] & (y == n2 - i32) ? -(n2-Int32(3))*sy : sy) : Int32(0)
    lc = act ? lin(g, (x, y, i32)) : i32                 # this column at z=1
    _, _, em = gs2_ld(ϵi, r, op, act, lc, zc0 - Int32(2), n3, sz)
    r0, f0, e0 = gs2_ld(ϵi, r, op, act, lc, zc0 - i32, n3, sz)
    r1, f1, ep = gs2_ld(ϵi, r, op, act, lc, zc0, n3, sz)
    z0 = wrapk(zc0 - i32, n3, op.per[3])
    l0 = lc + (z0 - i32)*sz
    ok0 = inxy & (z0 >= Int32(2))
    hh = (ok0 & (ho != 0)) ? @inbounds(ϵi[l0 + ho]) : 0f0
    hv = (ok0 & (vo != 0)) ? @inbounds(ϵi[l0 + vo]) : 0f0
    Em2 = 0f0; Em1 = 0f0; rprev = 0f0; fprev = 0x00
    for zz in zc0-i32:zc1+i32
        zw = wrapk(zz, n3, op.per[3])                       # the plane zz stands for
        l = lc + (zw - i32)*sz
        @inbounds Sin[tx, ty] = e0
        r2, f2, epn = gs2_ld(ϵi, r, op, act, lc, zz + Int32(2), n3, sz)
        zn = wrapk(zz + i32, n3, op.per[3])
        okn = inxy & (zn >= Int32(2)) & (zn <= n3 - i32)
        ln = lc + (zn - i32)*sz
        hhn = (okn & (ho != 0)) ? @inbounds(ϵi[ln + ho]) : 0f0
        hvn = (okn & (vo != 0)) ? @inbounds(ϵi[ln + vo]) : 0f0
        sync_threads()
        E = e0
        if inxy & (zw >= Int32(2)) & (zw <= n3 - i32) & isodd(x + y + zw + k0)
            xm = xl ? hh : @inbounds(Sin[tx-1, ty])
            xp = (xr & !xl) ? hh : @inbounds(Sin[tx+1, ty])
            ym = yl ? hv : @inbounds(Sin[tx, ty-1])
            yp = (yr & !yl) ? hv : @inbounds(Sin[tx, ty+1])
            I = (x, y, zw); o = withflag(op, f0)
            E = (r0 - offdiag_f(o, I, l, NBV(I, xm, xp, ym, yp, em, ep)))*getiD(o, l)
        end
        b = (zz & i32) + i32
        @inbounds SE[tx, ty, b] = E
        sync_threads()
        zb = zz - i32
        if outt & (zb >= zc0)
            lb = lc + (zb - i32)*sz
            v = Em1
            if iseven(x + y + zb + k0)
                bo = Int32(3) - b
                I = (x, y, zb); o = withflag(op, fprev)
                v = (rprev - offdiag_f(o, I, lb, NBV(I, @inbounds(SE[tx-1, ty, bo]), @inbounds(SE[tx+1, ty, bo]),
                        @inbounds(SE[tx, ty-1, bo]), @inbounds(SE[tx, ty+1, bo]), Em2, E)))*getiD(o, lb)
            end
            @inbounds ϵo[lb] = v
        end
        Em2 = Em1; Em1 = E; em = e0; e0 = ep; ep = epn
        rprev = r0; fprev = f0; r0 = r1; f0 = f1; r1 = r2; f1 = f2; hh = hhn; hv = hvn
    end
    return nothing
end
"Use the fused pair on this level: 3D, at least GS2_CELLS cells, no odd periodic length."
usegs2(op::Op{3}) = prod(Int.(op.g.n)) >= GS2_CELLS && !oddper(op)
usegs2(op::Op) = false
function gs2!(ϵi, ϵo, r, op::Op{3}, k0)
    g = op.g
    bl = (cld(Int(g.n[1]) - 2, GS2_TX - 2), cld(Int(g.n[2]) - 2, GS2_TY - 2), cld(Int(g.n[3]) - 2, GS2_CH))
    @cuda maxregs=64 threads=(GS2_TX, GS2_TY) blocks=bl k_gs2!(ϵi, ϵo, r, op, Int32(k0))
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
        pr = Pre(AV(r), op, I, l)
        rc = make_rc(r, op, fixd)
        v = rc(I, l)
        if isdeep(op, l)
            w = v - mult_f(op, I, l, ShiftScaled(pr, rc.s, rc.apply, op.iDreg))
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
    elseif it == 4 && usegs2(lv.op)
        # sweeps 1 and 2 on z (scratch during the solve), the fused pair 3,4 from z into ϵ
        launch(k_gs!, lo, hi, lv.z, lv.r, lv.op, Int32(1), true)
        launch(k_gs!, lo, hi, lv.z, lv.r, lv.op, Int32(2), false)
        gs2!(lv.z, lv.ϵ, lv.r, lv.op, 3)
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
    if l+1 == bottom_index(p)
        bottom!(p, l+1)
    else
        l+1 < length(p.levels) && vcycle!(p, l+1)
        smooth!(coarse, p.ωd, p.red)
    end
    g = fine.op.g
    launch(k_prol_incr!, inside_lo(g), inside_hi(g), fine.r, fine.x, fine.op, coarse.x, gc, fine.c, p.ωd)
end

# ---------------------------------------------------------------- small coarse levels
# On levels of a few thousand cells every kernel is a few µs of launch overhead. The V-cycle
# below such a level runs as one single-block kernel, with block barriers between the passes
# of the separate kernels and the same cell functions, so the results are identical.
const BOTTOM_CELLS = 2048
"Kernel-side view of one level for `k_bottom!` (`ϵ`, `r` as before the V-cycle's swap)."
struct DLevel{O,S,C}; op::O; x::S; ϵ::S; r::S; c::C; end
Adapt.@adapt_structure DLevel
@inline boxlen(e::NTuple{3,Int32}) = e[1]*e[2]*e[3]
@inline boxlen(e::NTuple{2,Int32}) = e[1]*e[2]
@inline function unravel(k::Int32, lo::NTuple{3,Int32}, e::NTuple{3,Int32})
    q = k ÷ e[1]
    return (lo[1] + k - q*e[1], lo[2] + q % e[2], lo[3] + q ÷ e[2])
end
@inline unravel(k::Int32, lo::NTuple{2,Int32}, e::NTuple{2,Int32}) = (lo[1] + k % e[1], lo[2] + k ÷ e[1])
@inline inner(g::Grid{D}) where D = (ntuple(_->Int32(2), Val(D)), ntuple(d-> g.n[d]-Int32(2), Val(D)))
# each pass: thread t takes cells t, t+nt, … of its box, then the block synchronises
@inline function b_jacobi!(op::Op, rn, r, x)
    lo, e = inner(op.g); k = threadIdx().x - i32
    while k < boxlen(e)
        jacobi_cell!(unravel(k, lo, e), rn, r, x, op, false)
        k += blockDim().x
    end
    sync_threads()
end
@inline function b_restrict_zero!(rc, cx, gc::Grid, rf, gf::Grid, c)
    lo, e = inner(gc); k = threadIdx().x - i32
    while k < boxlen(e)
        restrict_cell!(unravel(k, lo, e), rc, gc, rf, gf, c)
        k += blockDim().x
    end
    k = threadIdx().x
    while k <= length(cx)
        @inbounds cx[k] = 0f0
        k += blockDim().x
    end
    sync_threads()
end
@inline function b_smooth!(op::Op{D}, ϵ, r, x, ω) where D
    g = op.g
    lo = ntuple(_->Int32(2), Val(D))
    e = ntuple(d-> d == D ? cld(g.n[d], Int32(2)) - i32 : g.n[d] - Int32(2), Val(D))
    for i in i32:Int32(4)
        k = threadIdx().x - i32
        while k < boxlen(e)
            gs_cell!(unravel(k, lo, e), ϵ, r, op, i, i == i32)
            k += blockDim().x
        end
        sync_threads()
    end
    lo, e = inner(g); k = threadIdx().x - i32
    while k < boxlen(e)
        incr_cell!(unravel(k, lo, e), r, x, ϵ, op, ω)
        k += blockDim().x
    end
    sync_threads()
end
@inline function b_prol!(op::Op, r, x, cx, gc::Grid, c, ω)
    lo, e = inner(op.g); k = threadIdx().x - i32
    while k < boxlen(e)
        prol_cell!(unravel(k, lo, e), r, x, op, cx, gc, c, ω)
        k += blockDim().x
    end
    sync_threads()
end
# vcycle!(levels[1]) followed by smooth!(levels[1]), as the host recursion does them
function k_bottom!(lv::NTuple{K,DLevel}, ωd) where K
    ω = @inbounds ωd[1]
    @unroll for k in 1:K-1
        a = lv[k]; b = lv[k+1]
        b_jacobi!(a.op, a.ϵ, a.r, a.x)                  # new residual in ϵ: the roles swap
        b_restrict_zero!(b.r, b.x, b.op.g, a.ϵ, a.op.g, a.c)
    end
    c = lv[K]
    b_smooth!(c.op, c.ϵ, c.r, c.x, ω)
    @unroll for kk in 1:K-1
        k = K - kk
        a = lv[k]; b = lv[k+1]
        b_prol!(a.op, a.ϵ, a.x, b.x, b.op.g, a.c, ω)
        b_smooth!(a.op, a.r, a.ϵ, a.x, ω)
    end
    return nothing
end
"First level from which every coarser level is small enough for `k_bottom!` (0: none)."
function bottom_index(p::TPoisson)
    nl = length(p.levels); lb = nl + 1
    while lb > 2 && prod(p.levels[lb-1].op.g.n) <= BOTTOM_CELLS && !oddper(p.levels[lb-1].op)
        lb -= 1
    end
    return lb <= nl ? lb : 0
end
function bottom!(p::TPoisson, lb)
    lv = Tuple(DLevel(v.op, v.x, v.ϵ, v.r, v.c) for v in p.levels[lb:end])
    @cuda threads=512 k_bottom!(lv, p.ωd)
    for l in lb:length(p.levels)-1
        v = p.levels[l]; v.r, v.ϵ = v.ϵ, v.r
    end
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
# The x faces of BC!(u, bc, t; saveexit=exit) on this cell's row, for the cells at x=2 and
# x=n-1, with the values `v` just projected (k_bc!'s expressions for a non-periodic j=1).
# BC! then only needs its y and z passes: those overwrite every x-face value of the ghost
# rows, so the x pass is only needed on the interior rows.
struct XBC{B,T}; bc::B; t::T; exit::Bool; end
@inline projall(u, x, op::Op{3}, l, r, dt) =
    (projv(u, x, op, l, i32, r, dt), projv(u, x, op, l, Int32(2), r, dt), projv(u, x, op, l, Int32(3), r, dt))
@inline projall(u, x, op::Op{2}, l, r, dt) = (projv(u, x, op, l, i32, r, dt), projv(u, x, op, l, Int32(2), r, dt))
@inline xfaces!(::Nothing, u, I, l, g, v) = nothing
@inline function xfaces!(f::XBC{B,T}, u, I::NTuple{D,Int32}, l, g::Grid{D}, v) where {B,T,D}
    bc, t = f.bc, f.t
    n1 = g.n[1]
    if I[1] == Int32(2)
        I1 = setj(I, 1, 1)
        @unroll for i in i32:Int32(D)
            ci = (i-i32)*g.nc
            if i == i32
                @inbounds u[l-i32+ci] = bc(i, loc(i,I1,T), t)
                @inbounds u[l+ci] = bc(i, loc(i,I,T), t)
            else
                @inbounds u[l-i32+ci] = bc(i, loc(i,I1,T), t) + v[i] - bc(i, loc(i,I,T), t)
            end
        end
    end
    if I[1] == n1 - i32
        In = setj(I, 1, n1)
        @unroll for i in i32:Int32(D)
            ci = (i-i32)*g.nc
            if i == i32
                f.exit || (@inbounds u[l+i32+ci] = bc(i, loc(i,In,T), t))
            else
                @inbounds u[l+i32+ci] = bc(i, loc(i,In,T), t) + v[i] - bc(i, loc(i,I,T), t)
            end
        end
    end
    return nothing
end
# Face values of u at cell I. With an XBC (BC! not yet applied), the values BC! will leave:
# bc on non-periodic boundary faces, the exit face as it is, the periodic copy of index 2.
@inline ulo(::Nothing, u, I, l, i, g, per) = @inbounds u[l+(i-i32)*g.nc]
@inline uhi(::Nothing, u, I, l, i, g, per) = @inbounds u[l+g.s[i]+(i-i32)*g.nc]
@inline ulo(f::XBC{B,T}, u, I, l, i, g, per) where {B,T} =
    ((I[i] == Int32(2)) & !per[i]) ? f.bc(i, loc(i,I,T), f.t) : @inbounds(u[l+(i-i32)*g.nc])
@inline function uhi(f::XBC{B,T}, u, I, l, i, g, per) where {B,T}
    ci = (i-i32)*g.nc; s = g.s[i]; ni = g.n[i]
    I[i] < ni - i32 && return @inbounds u[l+s+ci]
    per[i] && return @inbounds u[l-(ni-Int32(3))*s+ci]
    (f.exit & (i == i32)) && return @inbounds u[l+s+ci]
    return f.bc(i, loc(i, setj(I, i, ni), T), f.t)
end
@inline uall(u, l, g::Grid{3}) = (@inbounds(u[l]), @inbounds(u[l+g.nc]), @inbounds(u[l+2g.nc]))
@inline uall(u, l, g::Grid{2}) = (@inbounds(u[l]), @inbounds(u[l+g.nc]))
# `xf`: an XBC to apply the x faces of BC! here (and read the face values BC! will leave),
# or nothing when BC! was applied before
function k_resid_src!(lo, hi, ps, pm, r, x, u, V, op::Op{D}, idt, useV::Bool, xf) where D
    I, ok = cellidx(lo, hi)
    s = 0f0
    if ok
        g = op.g; per = op.per
        l = lin(g, I)
        px = Pre(AV(x), op, I, l)
        v = 0f0
        if getiD(op, l) != 0
            du = 0f0; dv = 0f0
            @unroll for i in i32:Int32(D)
                ci = (i-i32)*g.nc
                du += uhi(xf, u, I, l, i, g, per) - ulo(xf, u, I, l, i, g, per)
                useV && (@inbounds dv += V[l+g.s[i]+ci] - V[l+ci])
            end
            z = useV ? du - (1 + (isreg(op,l) ? op.Dreg : @inbounds(op.Dg[l]))/(2D))*dv : du
            v = z*idt - mult_f(op, I, l, px)
        end
        @inbounds r[l] = v
        s = v
        xf === nothing || xfaces!(xf, u, I, l, g, uall(u, l, g))
    end
    block_partial!(ps, pm, s, 0f0)
    return nothing
end
# projected u at face (l,i): u - Δt·L·(x[l] - x[l-δᵢ]); `r` is isreg(op, l)
@inline function projv(u, x, op::Op, l, i, r::Bool, dt)
    g = op.g; ci = (i-i32)*g.nc
    L = r ? op.Lreg : @inbounds(op.L[l+ci])
    return @inbounds(u[l+ci]) - dt*L*(@inbounds(x[l])-@inbounds(x[l-g.s[i]]))
end
function k_proj!(lo, hi, u, x, op::Op{D}, dt, xf) where D
    I, ok = cellidx(lo, hi); ok || return
    g = op.g
    l = lin(g, I)
    r = isreg(op, l)
    v = projall(u, x, op, l, r, dt)
    @unroll for i in i32:Int32(D)
        @inbounds u[l+(i-i32)*g.nc] = v[i]
    end
    xfaces!(xf, u, I, l, g, v)
    return nothing
end
# Projection out of place (uold → unew) and the CFL flux Σᵢ max(0,u⁺ᵢ) + max(0,-u⁻ᵢ) of each cell
# (block max in pm), with the face values BC!(u, bc, t; saveexit=exit) leaves after the
# projection: bc values on non-periodic boundary faces, the old exit face, the periodic copy.
function k_proj_cfl!(lo, hi, ps, pm, unew, uold, x, op::Op{D}, dt::T, bc::B, t::T, exit::Bool, xf) where {D,T,B}
    I, ok = cellidx(lo, hi)
    m = 0f0
    if ok
        g = op.g; per = op.per
        l = lin(g, I)
        r = isreg(op, l)
        vs = projall(uold, x, op, l, r, dt)
        @unroll for i in i32:Int32(D)
            @inbounds unew[l+(i-i32)*g.nc] = vs[i]
        end
        xfaces!(xf, unew, I, l, g, vs)
        @unroll for i in i32:Int32(D)
            ci = (i-i32)*g.nc; s = g.s[i]; Ii = I[i]; ni = g.n[i]
            v = vs[i]
            lw = ((Ii == Int32(2)) & !per[i]) ? bc(i, loc(i, I, T), t) : v
            up = if Ii < ni - i32
                projv(uold, x, op, l+s, i, isreg(op, l+s), dt)
            elseif per[i]
                l2 = l - (ni-Int32(3))*s
                projv(uold, x, op, l2, i, isreg(op, l2), dt)
            elseif exit & (i == i32)
                @inbounds uold[l+s+ci]
            else
                bc(i, loc(i, setj(I, i, ni), T), t)
            end
            m += max(0f0, up) + max(0f0, -lw)
        end
    end
    block_partial!(ps, pm, 0f0, m)
    return nothing
end
# (WaterLily ≥ 1.9.1: domain BCs at t and the convective exit (predictor) are applied here,
# and the step is Δt/w with w=1 in the predictor and w=2 in the corrector)
# `cfl=true`: the CFL flux maximum of the final velocity goes to b.red slot 5 (see `cfl!`),
# from the projection kernel, which then writes to `a.f` (swapped into `a.u`)
function WaterLily.mom_project!(a::TFlow{D,T}, b::TPoisson, w::Int, t; cfl=false) where {D,T}
    # x faces of BC! (non-periodic x) in the residual and projection kernels, the rest of BC! after them
    xf = a.per[1] ? nothing : XBC(a.bc, T(t), a.exitBC)
    xf === nothing && BC!(a, a.u, t)
    w == 1 && a.exitBC && exitBC!(a, a.u, a.u⁰, a.Δt[end], t)
    dt = a.Δt[end]/T(w); g = a.grid; lv = b.levels[1]
    red_launch(k_resid_src!, inside_lo(g), inside_hi(g), b.red, 1, lv.r, lv.x, a.u, a.V, lv.op, inv(dt), !a.Vzero, xf)
    xf === nothing || BC!(a, a.u, t; jfirst=2)
    solve!(b, dt)
    if cfl
        red_launch(k_proj_cfl!, inside_lo(g), inside_hi(g), b.red, 5, a.f, a.u, b.x, lv.op, dt, a.bc, T(t), a.exitBC, xf)
        copy_exit!(a, a.f, a.u)
        a.u, a.f = a.f, a.u
    else
        launch(k_proj!, inside_lo(g), inside_hi(g), a.u, b.x, lv.op, dt, xf)
    end
    BC!(a, a.u, t; jfirst = xf === nothing ? 1 : 2)
end
"WaterLily.CFL of the velocity projected by `mom_project!(...; cfl=true)`."
cfl!(a::TFlow, b::TPoisson; Δt_max=10) = (fetch!(b.red); min(Δt_max, inv(getmax(b.red, 5) + 5a.ν)))
