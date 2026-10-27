# Biot-Savart domain boundary conditions (BiotSavartBCs.jl algorithm) on CUDA.
import BiotSavartBCs

"""
    TBiotPoisson

Pressure solver with Biot-Savart velocity boundary conditions, same algorithm as
`BiotSavartBCs.BiotSavartPoisson`: multigrid V-cycles with frozen domain BCs, then a
Biot-Savart update of the boundary velocity from the domain vorticity, repeated until
the residual converges.

- Level-1 vorticity lives in `flow.f` (scratch during projection) and is only computed in
  the boundary layer that the level-1 interactions read.
- Level-2 vorticity is summed directly from the curl of `u` (no full level-1 field).
- FMM interactions run one kernel per level over that level's target faces.
"""
struct TBiotPoisson{T,S,V,D,P<:TPoisson,W,G,K} <: AbstractPoisson{T,S,V}
    pois :: P
    x :: S
    L :: V
    z :: S
    ω :: W                          # vorticity levels 2:NL
    g :: G                          # NTuple of level grids (level 1 first)
    tar :: Vector{CuVector{Int32}}  # target (face) linear indices per level
    tadj :: CuVector{Int32}         # level-1 targets: the adjacent cell, negated for a lower face
    p :: S                          # pressure accumulator
    n :: Vector{Int16}
    perdir :: NTuple
    fmm :: Bool
    red :: RedBuf
    B :: CuVector{Int32}              # cells next to the domain boundary
    own :: Vector{CuVector{Float32}}  # per level, each target's own sum from the last FMM
    near :: Vector{CuVector{Int32}}   # per level, positions in `tar` of the targets that see the body box
    nnear :: Vector{Int}
    boxes :: Vector{Any}              # per level (lo,hi) body box (empty box: lo > hi)
    bst :: CuVector{Int32}            # [box lo, box hi, near count per level]
    images :: NTuple{K,Int32}         # symmetry planes (faces ±j) with image vortices
end
function TBiotPoisson(flow::TFlow{D,T}; nonbiotfaces=(), fmm=true, symmetry=()) where {D,T}
    # argument checks as BiotSavartBCs.BiotSavartPoisson
    flow.exitBC && throw(ArgumentError("exitBC=true is ignored when using Biot-Savart BCs"))
    isempty(flow.perdir) || throw(ArgumentError("TurboLily.BiotSimulation does not support periodic directions yet; use BiotSavartBCs.BiotSimulation"))
    fmm || throw(ArgumentError("fmm=false (the direct tree sum) is not implemented in TurboLily; use BiotSavartBCs.BiotSimulation"))
    all(f -> f isa Integer && 0 < abs(f) <= D, symmetry) && allunique(abs.(symmetry)) ||
        throw(ArgumentError("symmetry faces must be valid with at most one per direction"))
    pois = TPoisson(flow)
    ml = BiotSavartBCs.MLArray(flow.f)
    length(ml) >= 2 || throw(ArgumentError("grid $(size(flow.p) .- 2) has a single Biot-Savart level; TurboLily needs at least two (use BiotSavartBCs.BiotSimulation)"))
    ω = Tuple(ml[2:end])
    g = map(a->Grid(Base.front(size(a))), Tuple(ml))
    tars = BiotSavartBCs.collect_targets(ml, (nonbiotfaces..., symmetry...))   # no targets on these faces
    tar = [CuVector{Int32}(Int32.(LinearIndices(size(ml[l]))[tars[l]])) for l in eachindex(ml)]
    tadj = CuVector{Int32}(map(Array(tar[1])) do li
        i, l, lower = face_of(li, g[1])
        lower ? -(l + g[1].s[i]) : l - g[1].s[i]
    end)
    p = copy(flow.p)
    # cells next to the domain boundary (the only cells whose residual a Biot update changes)
    n = size(flow.p)
    B = Int32[LinearIndices(n)[I] for I in CartesianIndices(map(m->2:m-1, n)) if any(d->I[d]==2 || I[d]==n[d]-1, 1:D)]
    nl = length(tar)
    b = TBiotPoisson{T,typeof(flow.p),typeof(flow.μ₀),D,typeof(pois),typeof(ω),typeof(g),length(symmetry)}(
        pois, pois.x, pois.L, pois.z, ω, g, tar, tadj, p, pois.n, flow.perdir, fmm, pois.red,
        CuVector{Int32}(B), [CuVector{Float32}(undef, length(t)) for t in tar],
        [CuVector{Int32}(undef, length(t)) for t in tar], zeros(Int, nl), Any[],
        CuVector{Int32}(undef, 6 + nl), Int32.(Tuple(symmetry)))
    body_box!(b)
    return b
end
WaterLily.update!(b::TBiotPoisson) = (WaterLily.update!(b.pois); body_box!(b))
Base.getproperty(b::TBiotPoisson, s::Symbol) = s === :levels ? getfield(b, :pois).levels : getfield(b, s)

# Body box (BiotSavartBCs.transition_box): the cells with 0 < L < 1 on some face, grown by one
# cell. Outside it L is 0 or 1 on every face a curl stencil reads, so curl(L∇p) ≡ 0 there and
# ω of u - Δt·L∇p does not change when p changes.
@inline function warp_box!(bst, off, pred::Bool, I::NTuple{D,Int32}) where D
    @unroll for d in i32:Int32(D)
        lo = pred ? I[d] : typemax(Int32); hi = pred ? I[d] : typemin(Int32)
        o = Int32(16)
        while o > 0
            lo = min(lo, shfl_down_sync(0xffffffff, lo, o)); hi = max(hi, shfl_down_sync(0xffffffff, hi, o))
            o >>= 1
        end
        if flat_tid() % Int32(32) == 0 && lo != typemax(Int32)
            CUDA.atomic_min!(pointer(bst, off + d), lo)
            CUDA.atomic_max!(pointer(bst, off + 3 + d), hi)
        end
    end
    return nothing
end
function k_tbox!(lo, hi, L, g::Grid{D}, bst) where D
    I, ok = cellidx(lo, hi)
    tr = false
    if ok
        l = lin(g, I)
        @unroll for a in i32:Int32(D)
            @inbounds v = L[l + (a-i32)*g.nc]
            tr |= (v > 0) & (v < 1)
        end
    end
    warp_box!(bst, 0, tr, I)
    return nothing
end
# Whether the sources that target T sums on its level overlap the box blo:bhi.
@inline function sees(T::NTuple{D,Int32}, g::Grid{D}, top, bottom, blo, bhi) where D
    c = closeR(Val(D))
    olo = outer_lo(T, g, c, bottom, top); ohi = outer_hi(T, g, c, bottom, top)
    ok = true
    @unroll for d in i32:Int32(D)
        ok &= max(olo[d], blo[d]) <= min(ohi[d], bhi[d])
    end
    return ok
end
# Appends to `list` the positions in `tar` of the targets whose sum (or the sum of one of their
# images) reads a source in the box; the other targets keep their own sums in an update.
function k_near!(n, tar, g::Grid{D}, top::Bool, bottom::Bool, blo, bhi, imgs::NTuple{K,Int32}, list, cnt, slot) where {D,K}
    k = tid1()
    pred = false
    if k <= n
        li = @inbounds tar[k]
        i = (li - i32) ÷ g.nc + i32
        T = coords(g, li - (i-i32)*g.nc)
        pred = sees(T, g, top, bottom, blo, bhi)
        @unroll for m in i32:Int32(2^K - 1)
            Tm = T
            @unroll for b in i32:Int32(K)
                ((m >> (b - i32)) & i32 == i32) && (Tm = first(reflect(Tm, imgs[b], i, g)))
            end
            pred |= sees(Tm, g, top, bottom, blo, bhi)
        end
    end
    wappend!(list, cnt, slot, pred, k)
    return nothing
end
function body_box!(b::TBiotPoisson{T, S, V, D}) where {T, S, V, D}
    g = b.g; nl = length(g)
    copyto!(b.bst, vcat(fill(typemax(Int32), 3), fill(typemin(Int32), 3), zeros(Int32, nl)))
    launch(k_tbox!, ntuple(_->1, D), Int.(g[1].n), b.L, g[1], b.bst)
    st = hostcopy!(Vector{Int32}(undef, length(b.bst)), b.bst)
    box = st[1] > st[4] ? (ntuple(_->2, D), ntuple(_->1, D)) : (ntuple(d->Int(st[d])-1, D), ntuple(d->Int(st[3+d])+1, D))
    empty!(b.boxes); push!(b.boxes, box)
    for l in 2:nl
        lo, hi = b.boxes[end]
        push!(b.boxes, lo[1] > hi[1] ? (lo, hi) : (map(i->(i+2)÷2, lo), map(i->(i+2)÷2, hi)))
    end
    for l in 1:nl
        lo, hi = b.boxes[l]
        launch1(k_near!, length(b.tar[l]), b.tar[l], g[l], l == 1, l == nl, Int32.(lo), Int32.(hi), b.images, b.near[l], b.bst, Int32(6 + l))
    end
    hostcopy!(st, b.bst)
    b.nnear .= st[7:6+nl]
    return b
end

# velocity accessors: plain u, or the projected u - Δt·L∇p (without writing it)
struct UA{A}; u::A; nc::Int32; end
@inline (f::UA)(l, a) = @inbounds f.u[l + (a-i32)*f.nc]
struct UC{A,P,O,T}; u::A; p::P; op::O; dt::T; end
Adapt.@adapt_structure UA
Adapt.@adapt_structure UC
@inline function (f::UC)(l, a)
    op = f.op; ca = (a-i32)*op.g.nc
    L = isreg(op, l) ? op.Lreg : @inbounds(op.L[l+ca])
    @inbounds return f.u[l+ca] - f.dt*L*(f.p[l] - f.p[l-op.g.s[a]])
end
# centred curl at cell I (BiotSavartBCs.centered_curl): 3D all components, 2D ω₃ in slot 1
@inline function dcross(i, j, l, U::F, g::Grid) where F   # ∂uᵢ/∂xⱼ at the cell centre, i ≠ j
    sj, si = g.s[j], g.s[i]
    return (U(l+sj, i) + U(l+sj+si, i) - U(l-sj, i) - U(l-sj+si, i))*0.25f0
end
@inline function curl_c(i, l, U::F, g::Grid{3}) where F
    j = i % Int32(3) + i32; k = (i + i32) % Int32(3) + i32
    return dcross(k, j, l, U, g) - dcross(j, k, l, U, g)
end
@inline curl_c(i, l, U::F, g::Grid{2}) where F = i == i32 ? dcross(Int32(2), i32, l, U, g) - dcross(i32, Int32(2), l, U, g) : 0f0
@inline curl_c(i, l, u::AbstractArray, g::Grid{3}) = curl_c(i, l, UA(u, g.nc), g)
@inline curl_c(i, l, u::AbstractArray, g::Grid{2}) = curl_c(i, l, UA(u, g.nc), g)

# level-1 vorticity in the boundary layer read by the level-1 interactions
function k_curl_layer!(lo, hi, ω, u, g::Grid{D}, w) where D
    I, ok = cellidx(lo, hi); ok || return
    near = false
    @unroll for d in i32:Int32(D)
        near |= (I[d] <= w) | (I[d] >= g.n[d] - w + i32)
    end
    near || return
    l = lin(g, I)
    @unroll for i in i32:Int32(D)
        @inbounds ω[l + (i-i32)*g.nc] = curl_c(i, l, u, g)
    end
    return nothing
end
# level-2 vorticity = Σ over the 2ᵈ fine cells of the level-1 curl (zero outside buff=2)
function k_curl_restrict!(lo, hi, ω2, g2::Grid{D}, u, g::Grid{D}) where D
    I, ok = cellidx(lo, hi); ok || return
    l2 = lin(g2, I)
    J0 = ntuple(d-> Int32(2)*I[d] - Int32(2), Val(D))
    @unroll for i in i32:Int32(D)
        s = 0f0
        if D == 3
            for dz in 0:1, dy in 0:1, dx in 0:1
                J = (J0[1]+Int32(dx), J0[2]+Int32(dy), J0[D]+Int32(dz))
                inb = true
                @unroll for d in i32:Int32(D)
                    inb &= (J[d] >= Int32(3)) & (J[d] <= g.n[d] - Int32(2))
                end
                inb && (s += curl_c(i, lin(g, J), u, g))
            end
        else
            for dy in 0:1, dx in 0:1
                J = (J0[1]+Int32(dx), J0[2]+Int32(dy))
                inb = (J[1] >= Int32(3)) & (J[1] <= g.n[1] - Int32(2)) & (J[2] >= Int32(3)) & (J[2] <= g.n[2] - Int32(2))
                inb && (s += curl_c(i, lin(g, J), u, g))
            end
        end
        @inbounds ω2[l2 + (i-i32)*g2.nc] = s
    end
    return nothing
end
# vector restriction a[I,i] = Σ b[up(I),i] (BiotSavartBCs.restrict!)
function k_vrestrict!(lo, hi, a, ga::Grid{D}, b, gb::Grid{D}) where D
    I, ok = cellidx(lo, hi); ok || return
    la = lin(ga, I)
    lb = lin(gb, ntuple(d-> Int32(2)*I[d] - Int32(2), Val(D)))
    @unroll for i in i32:Int32(D)
        s = 0f0
        cb = (i-i32)*gb.nc
        if D == 3
            for dz in 0:1, dy in 0:1, dx in 0:1
                @inbounds s += b[lb + cb + dx + dy*gb.s[2] + dz*gb.s[D]]
            end
        else
            for dy in 0:1, dx in 0:1
                @inbounds s += b[lb + cb + dx + dy*gb.s[2]]
            end
        end
        @inbounds a[la + (i-i32)*ga.nc] = s
    end
    return nothing
end
layer_width(::Val{3}) = 6
layer_width(::Val{2}) = 10
function fill_ω!(b::TBiotPoisson{T,S,V,D}, ω1, u) where {T,S,V,D}
    g = b.g
    launch(k_curl_layer!, ntuple(_->3,D), ntuple(d->Int(g[1].n[d])-2,D), ω1, u, g[1], Int32(layer_width(Val(D))))
    launch(k_curl_restrict!, inside_lo(g[2]), inside_hi(g[2]), b.ω[1], g[2], u, g[1])
end

# ---------------------------------------------------------------- FMM interactions
@inline closeR(::Val{3}) = Int32(2)
@inline closeR(::Val{2}) = Int32(4)
# Biot-Savart kernel weighted(x-S, S, i, ω)
@inline function bs_weight(r1, r2, r3, ω, ls, g::Grid{3}, i)
    @inbounds w1 = ω[ls]; @inbounds w2 = ω[ls + g.nc]; @inbounds w3 = ω[ls + 2g.nc]
    ir = CUDA.rsqrt(r1*r1 + r2*r2 + r3*r3)
    c = i == 1 ? w2*r3 - w3*r2 : i == 2 ? w3*r1 - w1*r3 : w1*r2 - w2*r1
    return c*(ir*ir*ir)
end
@inline function bs_weight(r1, r2, r3, ω, ls, g::Grid{2}, i)
    @inbounds w = ω[ls]
    return (i == 1 ? -w*r2 : w*r1)*@fastmath(inv(r1*r1 + r2*r2))
end
bs_scale(::Val{3}) = 1f0/(4f0*Float32(π))
bs_scale(::Val{2}) = 1f0/(2f0*Float32(π))
@inline function outer_lo(T::NTuple{D,Int32}, g::Grid{D}, c, bottom, top) where D
    return ntuple(Val(D)) do d
        dT = fld(T[d] + Int32(2), Int32(2))   # (image targets can have negative indices)
        v = bottom ? Int32(2) : Int32(2)*max(dT - c, Int32(2)) - Int32(2)
        top ? max(v, Int32(3)) : v
    end
end
@inline function outer_hi(T::NTuple{D,Int32}, g::Grid{D}, c, bottom, top) where D
    return ntuple(Val(D)) do d
        dT = fld(T[d] + Int32(2), Int32(2))   # (image targets can have negative indices)
        v = bottom ? g.n[d] - i32 : Int32(2)*min(dT + c, (g.n[d] + i32) ÷ Int32(2)) - i32
        top ? min(v, g.n[d] - Int32(2)) : v
    end
end
# Sum over the source box olo:ohi minus the hole ilo:ihi (no hole on the top level). Each
# x-line is split around the hole so the inner loop has no branch.
# Lane p of P handles the outer rows p, p+P, …
@inline function xline(::Val{i}, x::NTuple{3}, ω, g::Grid{3}, ls, a, b, r2, r3) where i
    j = i % 3 + 1; k = (i + 1) % 3 + 1
    cj = Int32(j-1)*g.nc; ck = Int32(k-1)*g.nc
    s = 0f0
    for s1 in a:b
        r1 = x[1] - s1
        r = (r1, r2, r3)
        ir = CUDA.rsqrt(r1*r1 + r2*r2 + r3*r3)
        @inbounds s += (ω[ls+s1+cj]*r[k] - ω[ls+s1+ck]*r[j])*(ir*ir*ir)
    end
    return s
end
@inline function xline(::Val{i}, x::NTuple{2}, ω, g::Grid{2}, ls, a, b, r2, r3) where i
    s = 0f0
    for s1 in a:b
        r1 = x[1] - s1
        @inbounds s += ω[ls+s1]*(i == 1 ? r2 : r1)*@fastmath(inv(r1*r1 + r2*r2))
    end
    return i == 1 ? -s : s
end
@inline function boxsum(vi::Val, x::NTuple{D}, ω, g::Grid{D}, olo, ohi, ilo, ihi, top, p, P) where D
    s = 0f0
    q = (D == 3 ? olo[3] : olo[2]) + p
    qend = D == 3 ? ohi[3] : ohi[2]
    while q <= qend
        for s2 in (D == 3 ? olo[2] : q):(D == 3 ? ohi[2] : q)
            s3 = D == 3 ? q : i32
            ls = (s2 - i32)*g.s[2] + (D == 3 ? (s3 - i32)*g.s[D] : Int32(0))
            r2 = x[2] - s2; r3 = D == 3 ? x[D] - s3 : 0f0
            hole = !top & (ilo[2] <= s2 <= ihi[2]) & (D == 2 || (ilo[D] <= s3 <= ihi[D]))
            if hole
                s += xline(vi, x, ω, g, ls, olo[1], min(ohi[1], ilo[1]-i32), r2, r3)
                s += xline(vi, x, ω, g, ls, max(olo[1], ihi[1]+i32), ohi[1], r2, r3)
            else
                s += xline(vi, x, ω, g, ls, olo[1], ohi[1], r2, r3)
            end
        end
        q += P
    end
    return s
end
# Interaction sum at one target of one level (BiotSavartBCs.interaction).
# P consecutive lanes share a target, splitting the outermost source loop; P ∈ {1,2,4,…,32}.
# Unscaled interaction sum at target cell T (component i), possibly an image outside the domain.
@inline function target_sum(T::NTuple{D,Int32}, i, ω, g::Grid{D}, top, bottom, p, P) where D
    c = closeR(Val(D))
    olo = outer_lo(T, g, c, bottom, top)
    ohi = outer_hi(T, g, c, bottom, top)
    ilo = ntuple(d-> max(T[d] - c, Int32(2)), Val(D))
    ihi = ntuple(d-> min(T[d] + c, g.n[d] - i32), Val(D))
    x = ntuple(d-> Float32(T[d]) + (d == i ? (T[d] == i32 ? 0.5f0 : -0.5f0) : 0f0), Val(D))
    # compile-time component: only the two vorticity components that enter are loaded
    return i == i32 ? boxsum(Val(1), x, ω, g, olo, ohi, ilo, ihi, top, p, P) :
           i == 2i32 ? boxsum(Val(2), x, ω, g, olo, ohi, ilo, ihi, top, p, P) :
                       boxsum(Val(3), x, ω, g, olo, ohi, ilo, ihi, top, p, P)
end
# Reflect target T across domain face `face` (BiotSavartBCs.image): ±j is the upper/lower face
# normal to j. The normal velocity flips sign across the plane, the tangential ones do not.
# Across the lower face, a normal-velocity target (a face, not a cell centre) maps to 4-T.
@inline function reflect(T::NTuple{D,Int32}, face::Int32, i, g::Grid{D}) where D
    j = abs(face)
    lo = Int32(3) + ((i == j) & (T[j] > i32) ? i32 : Int32(0))
    T2 = ntuple(Val(D)) do d
        d == j ? (face > 0 ? Int32(2)*g.n[d] - T[d] - i32 : lo - T[d]) : T[d]
    end
    return T2, i == j ? -1f0 : 1f0
end
# Interaction sum at one target of one level (BiotSavartBCs.interaction), plus the image
# targets of every non-empty subset of the symmetry planes `imgs` (BiotSavartBCs.symmetry).
# P consecutive lanes share a target, splitting the outermost source loop; P ∈ {1,2,4,…,32}.
# The targets are tar[idx[k]] (all of tar when idx is nothing); each also stores its sum in own.
function k_interact!(n, tar, idx, own, ω, g::Grid{D}, top::Bool, bottom::Bool, P::Int32, imgs::NTuple{K,Int32}) where {D,K}
    t = tid1() - i32
    k = t ÷ P + i32; p = t % P
    valid = k <= n
    kk = valid ? k : n
    j = idx === nothing ? kk : @inbounds(idx[kk])
    li = @inbounds tar[j]
    i = (li - i32) ÷ g.nc + i32
    l = li - (i-i32)*g.nc
    T = coords(g, l)
    val = 0f0
    if valid
        val = target_sum(T, i, ω, g, top, bottom, p, P)
        @unroll for m in i32:Int32(2^K - 1)   # empty unless images are requested
            Tm = T; sgn = 1f0
            @unroll for b in i32:Int32(K)
                if (m >> (b - i32)) & i32 == i32
                    Tm, sb = reflect(Tm, imgs[b], i, g)
                    sgn *= sb
                end
            end
            val += sgn*target_sum(Tm, i, ω, g, top, bottom, p, P)
        end
    end
    off = P ÷ Int32(2)
    while off > 0
        val += shfl_down_sync(0xffffffff, val, off)
        off >>= 1
    end
    if valid && p == 0
        v = val*bs_scale(Val(D))
        @inbounds ω[li] = v
        @inbounds own[j] = v
    end
    return nothing
end
# a[target] += bilinear projection of the coarse targets (BiotSavartBCs.project)
@inline function pstep(I, j, gc::Grid)
    Ij = I[j]
    return Ij % Int32(2) == i32 ? (Ij ÷ Int32(2) == gc.n[j] - Int32(2) ? Int32(0) : i32) :
                                   (Ij ÷ Int32(2) == i32 ? Int32(0) : -i32)
end
function k_project!(n, tar, a, g::Grid{D}, b, gc::Grid{D}) where D
    k = tid1(); k <= n || return
    li = @inbounds tar[k]
    i = (li - i32) ÷ g.nc + i32
    l = li - (i-i32)*g.nc
    I = coords(g, l)
    Id = ntuple(d-> (I[d] + Int32(2)) ÷ Int32(2), Val(D))
    lc = lin(gc, Id) + (i-i32)*gc.nc
    if D == 3
        j = i % Int32(3) + i32; kk = (i + i32) % Int32(3) + i32
        dj = pstep(I, j, gc)*gc.s[j]; dk = pstep(I, kk, gc)*gc.s[kk]
        @inbounds v = 0.015625f0*(9f0*b[lc] + 3f0*b[lc+dj] + 3f0*b[lc+dk] + b[lc+dj+dk])
    else
        j = i % Int32(2) + i32
        dj = pstep(I, j, gc)*gc.s[j]
        @inbounds v = 0.125f0*(3f0*b[lc] + b[lc+dj])
    end
    @inbounds a[li] += v
    return nothing
end
"""
    fmm!(b, ω1; update=false)

Biot-Savart interactions on all levels (level-1 vorticity `ω1`, coarser levels `b.ω`) and the
coarse-to-fine projection; the level-1 target slots of `ω1` then hold the induced boundary
velocity. Every target stores its own sum in `b.own`. With `update`, only the targets that
see the body box are summed (BiotSavartBCs.fmmBC! with `idx`); the others keep their own sums.
"""
function fmm!(b::TBiotPoisson{T,S,V,D}, ω1; update=false) where {T,S,V,D}
    ωs = (ω1, b.ω...)
    nl = length(ωs)
    for l in 1:nl
        update && launch1(k_settar!, length(b.tar[l]), b.tar[l], ωs[l], b.own[l])
        n = update ? b.nnear[l] : length(b.tar[l])
        n == 0 && continue
        # lanes per target (measured on the benchmark cases): about 32k lanes per level, 1 to 16
        # per target, and at least one per image sum, which keeps the sources per warp small
        P = clamp(prevpow(2, max(1, 32768 ÷ n)), min(2^length(b.images), 16), 16)
        @cuda threads=256 blocks=cld(n*P, 256) k_interact!(Int32(n), b.tar[l], update ? b.near[l] : nothing, b.own[l], ωs[l], b.g[l], l == 1, l == nl, Int32(P), b.images)
    end
    for l in nl-1:-1:1
        launch1(k_project!, length(b.tar[l]), b.tar[l], ωs[l], b.g[l], ωs[l+1], b.g[l+1])
    end
end
# a[tar[k]] = s[k]
function k_settar!(n, tar, a, s)
    k = tid1(); k <= n || return
    @inbounds a[tar[k]] = s[k]
    return nothing
end


# ---------------------------------------------------------------- domain boundary updates
# u at the boundary face of target (I,i): index 2 (low) or n (high)
@inline function face_of(li, g::Grid)
    i = (li - i32) ÷ g.nc + i32
    l = li - (i-i32)*g.nc
    lower = (l - i32) ÷ g.s[i] % g.n[i] == 0   # I[i]==1
    return i, l, lower
end
# The same from the precomputed adjacent cell `a` (negative for a lower face), without
# divisions: the component, the u index of the boundary face, the adjacent cell, lower.
@inline function face_adj(li, a, g::Grid)
    i = li > Int32(2)*g.nc ? Int32(3) : li > g.nc ? Int32(2) : i32
    lower = a < 0; la = abs(a)
    l = li - (i-i32)*g.nc
    return i, (lower ? la : l) + (i-i32)*g.nc, la, lower
end
# u = U + uᵥ; with `resid`, also r[adjacent] += Δuₙ·(lower ? -1 : 1)
function k_biotbc!(n, tar, tadj, u, uv, r, U, g::Grid, resid::Bool)
    k = tid1(); k <= n || return
    li = @inbounds tar[k]
    i, lu, la, lower = face_adj(li, @inbounds(tadj[k]), g)
    un = @inbounds U[i] + uv[li]
    if resid
        un0 = @inbounds u[lu]
        CUDA.@atomic r[la] += (un - un0)*(lower ? -1f0 : 1f0)
    end
    @inbounds u[lu] = un
    return nothing
end
function k_norms!(lo, hi, ps, pm, r, g::Grid)
    I, ok = cellidx(lo, hi)
    v = ok ? abs(@inbounds(r[lin(g, I)])) : 0f0
    block_partial!(ps, pm, v, v)
    return nothing
end
"Σ|r| and max|r| over the whole level-1 residual."
function norms!(b::TBiotPoisson, r)
    g = b.g[1]
    red_launch(k_norms!, ntuple(_->1, length(g.n)), Int.(g.n), b.red, 4, r, g)
    fetch!(b.red)
    return getsum(b.red, 4), getmax(b.red, 4)
end

# Incompressible & irrotational ghosts (BiotSavartBCs.pflowBC!), one pass per normal direction i
@inline edgev(I, j, g::Grid, v) = (Int32(2) < I[j]) & (I[j] < g.n[j]) ? v : 0f0
function k_pflow_t!(lo, hi, i, u, g::Grid{D}) where D
    I, ok = faceidx(lo, hi, i); ok || return   # I[i] == 1
    si = g.s[i]; ci = (i-i32)*g.nc
    @unroll for j in i32:Int32(D)
        j == i && continue
        # I[j] ∈ 2:n, others 2:n-1
        inr = I[j] >= Int32(2)
        @unroll for d in i32:Int32(D)
            (d != i && d != j) && (inr &= (I[d] >= Int32(2)) & (I[d] <= g.n[d] - i32))
        end
        inr || continue
        cj = (j-i32)*g.nc; sj = g.s[j]
        l1 = lin(g, I); ln = l1 + (g.n[i]-i32)*si
        # low: u[I,j] = u[I+δi,j] - edge(uᵢ[I+δi]-uᵢ[I+δi-δj])
        @inbounds u[l1+cj] = u[l1+si+cj] - edgev(I, j, g, u[l1+si+ci] - u[l1+si-sj+ci])
        # high: u[I,j] = u[I-δi,j] + edge(uᵢ[I]-uᵢ[I-δj])
        @inbounds u[ln+cj] = u[ln-si+cj] + edgev(I, j, g, u[ln+ci] - u[ln-sj+ci])
    end
    return nothing
end
function k_pflow_n!(lo, hi, i, u, g::Grid{D}) where D
    I, ok = faceidx(lo, hi, i); ok || return   # I[i] == 1, others 2:n-1
    l = lin(g, I)
    dv = 0f0
    @unroll for k in i32:Int32(D)
        ck = (k-i32)*g.nc
        @inbounds dv += u[l+g.s[k]+ck] - u[l+ck]
    end
    @inbounds u[l+(i-i32)*g.nc] += dv
    return nothing
end
function pflowBC!(u, g::Grid{D}) where D
    for i in 1:D
        lo = ntuple(_->1, D); hi = ntuple(d-> d==i ? 1 : Int(g.n[d]), D)
        launch_face(k_pflow_t!, lo, hi, i, u, g)
        lo = ntuple(d-> d==i ? 1 : 2, D); hi = ntuple(d-> d==i ? 1 : Int(g.n[d])-1, D)
        launch_face(k_pflow_n!, lo, hi, i, u, g)
    end
end

# ---------------------------------------------------------------- vorticity in the body box
clip(lo, hi, a, b) = (map(max, lo, a), map(min, hi, b))
"""
    body_ω!(b, ω1, u, p, dt; full=false)

Recompute ω of `u - Δt·L∇p` in the body box on every level (BiotSavartBCs.body_ω!): level 1 in
the layer, level 2 from the fine curls, levels ≥3 restricted. With `full`, levels ≥3 are
restricted everywhere, after `fill_ω!` has written levels 1 and 2.
"""
function body_ω!(b::TBiotPoisson{T,S,V,D}, ω1, u, p, dt; full=false) where {T,S,V,D}
    g = b.g; U = UC(u, p, b.levels[1].op, dt)
    lo, hi = clip(b.boxes[1]..., ntuple(_->3, D), ntuple(d->Int(g[1].n[d])-2, D))
    launch(k_curl_layer!, lo, hi, ω1, U, g[1], Int32(layer_width(Val(D))))
    lo, hi = clip(b.boxes[2]..., inside_lo(g[2]), inside_hi(g[2]))
    launch(k_curl_restrict!, lo, hi, b.ω[1], g[2], U, g[1])
    for l in 3:length(g)
        lo, hi = full ? (inside_lo(g[l]), inside_hi(g[l])) : clip(b.boxes[l]..., inside_lo(g[l]), inside_hi(g[l]))
        launch(k_vrestrict!, lo, hi, b.ω[l-1], g[l], b.ω[l-2], g[l-1])
    end
end
# Σ|r| and max|r| over a cell list (block partials)
function k_listnorms!(n, ps, pm, list, r)
    k = tid1()
    v = k <= n ? abs(@inbounds(r[list[k]])) : 0f0
    block_partial!(ps, pm, v, v)
    return nothing
end
function list_norms!(b::TBiotPoisson, list, r, slot)
    n = length(list); red = b.red
    nb = cld(n, 256); checkcap(red, nb)
    @cuda threads=256 blocks=nb k_listnorms!(Int32(n), red.ps, red.pm, list, r)
    finish!(red, nb, red.s, red.m, slot)
end
# boundary velocity u = U + uᵥ; r[adjacent] += Δuₙ·sgn·scale; partial Σ Δuₙ·sgn (net flux change)
function k_biotbc2!(n, ps, pm, tar, tadj, u, uv, r, U, g::Grid, scale)
    k = tid1()
    d = 0f0
    if k <= n
        li = @inbounds tar[k]
        i, lu, la, lower = face_adj(li, @inbounds(tadj[k]), g)
        un = @inbounds U[i] + uv[li]
        d = (un - @inbounds(u[lu]))*(lower ? -1f0 : 1f0)
        CUDA.@atomic r[la] += d*scale
        @inbounds u[lu] = un
    end
    block_partial!(ps, pm, d, 0f0)
    return nothing
end
# global flux fix with fix = sums[slot]·fscale/n: u[face] += fix·(lower ? 1 : -1), r[adjacent] -= fix·rscale
function k_fixresid2!(n, tar, tadj, u, r, sums, slot::Int32, fscale, rscale, g::Grid)
    k = tid1(); k <= n || return
    fix = Float32(@inbounds(sums[slot]))*fscale / Float32(n)
    li = @inbounds tar[k]
    i, lu, la, lower = face_adj(li, @inbounds(tadj[k]), g)
    @inbounds u[lu] += fix*(lower ? 1f0 : -1f0)
    CUDA.@atomic r[la] -= fix*rscale
    return nothing
end

# ---------------------------------------------------------------- projection
bctuple(f::Function, t::T, N) where T = ntuple(i->f(i, zero(SVector{N,T}), t), N)
bctuple(f::Tuple, t, N) = f
"""
    mom_project!(a::TFlow, b::TBiotPoisson, w::Int, t; tol=2e-3, itmx=32)

BiotSavartBCs projection, rearranged to avoid full-grid passes per Biot iteration:
the pressure `p` is solved for directly and kept cumulative, `u` is only projected once at
the end, the vorticity of the projected field is recomputed only in the body box, and the
global flux fix and the residual norms are updated from the boundary cells.
Stopping criterion as BiotSavartBCs#13 / `WaterLily.solver!`: `max|r| < tol`, `Σ|r|/N < tol/10`.
The step is `Δt/w` (`w=1` predictor, `w=2` corrector), as `WaterLily.mom_project!`.
"""
function WaterLily.mom_project!(a::TFlow{D,T}, b::TBiotPoisson, w::Int, t; tol=2e-3, itmx=32) where {D,T}
    # (no BC! here, as BiotSavartBCs: the Biot-Savart BCs and pflowBC! set the boundary)
    dt = a.Δt[end]/T(w); idt = inv(dt)
    U = T.(bctuple(a.uBC, T(t), D))
    top = b.levels[1]; op = top.op; g = op.g; red = b.red
    tar1 = b.tar[1]; nt = length(tar1)
    fill!(b.pois.ωd, 1)
    # stopping criterion (BiotSavartBCs#13, as WaterLily.solver!): max|r| < tol and Σ|r|/N < tol/10,
    # in units of Δt·r since r here is the residual of the equation for p
    r₁tol = WaterLily.l1n_tol(b.pois.ml.levels[1], tol)/dt; r∞tol = tol/dt
    # vorticity of u - Δt·L∇p, Biot-Savart boundary velocity
    fill_ω!(b, a.f, a.u)
    body_ω!(b, a.f, a.u, a.p, dt; full=true)
    fmm!(b, a.f)
    launch1(k_biotbc!, nt, tar1, b.tadj, a.u, a.f, top.r, U, g, false)
    # residual r = (div(u) - δv·div(V))/Δt - A p (Σr in slot 1), global flux fix, norms
    red_launch(k_resid_src!, inside_lo(g), inside_hi(g), red, 1, top.r, a.p, a.u, a.V, op, idt, !a.Vzero, nothing)
    launch1(k_fixresid2!, nt, tar1, b.tadj, a.u, top.r, red.s, Int32(1), dt, idt, g)
    r₁, r∞ = norms!(b, top.r)
    nᵖ, nᵇ = 0, 0
    @log ", $nᵖ, $(r∞*dt), $(r₁*dt), $nᵇ\n"
    while nᵖ < itmx
        # V-cycles with fixed BCs until Σ|r| drops >10x
        rtol = max(r₁tol, 0.1r₁)
        while nᵖ < itmx
            iteration!(b.pois; exclb=true, adapt=false)
            fetch!(red); r₁ = getsum(red, 3); nᵖ += 1
            r₁ < rtol && break
        end
        r₁mg, r∞in = getsum(red, 3), getmax(red, 3)   # max over cells away from the boundary
        # Biot-Savart update of the boundary velocity with the current p
        body_ω!(b, a.f, a.u, a.p, dt)
        fmm!(b, a.f; update=true)
        list_norms!(b, b.B, top.r, 5)
        nb = cld(nt, 256); checkcap(red, nb)
        @cuda threads=256 blocks=nb k_biotbc2!(Int32(nt), red.ps, red.pm, tar1, b.tadj, a.u, a.f, top.r, U, g, idt)
        finish!(red, nb, red.s, red.m, 6)
        launch1(k_fixresid2!, nt, tar1, b.tadj, a.u, top.r, red.s, Int32(6), 1f0, idt, g)
        list_norms!(b, b.B, top.r, 7)
        fetch!(red)
        r₁ = r₁mg - getsum(red, 5) + getsum(red, 7); r∞ = max(r∞in, getmax(red, 7)); nᵇ += 1
        @log ", $nᵖ, $(r∞*dt), $(r₁*dt), $nᵇ\n"
        (r₁ < r₁tol && r∞ < r∞tol) && break
    end
    push!(b.n, nᵖ)
    launch(k_proj!, inside_lo(g), inside_hi(g), a.u, a.p, op, dt, nothing)
    pflowBC!(a.u, g)
end
"""
    TurboLily.BiotSimulation(dims, uBC, L; nonbiotfaces=(), fmm=true, symmetry=(), kwargs...)

Same signature as `BiotSavartBCs.BiotSimulation`; returns a `WaterLily.Simulation` with
the CUDA flow and Biot-Savart pressure solver.

- `nonbiotfaces`: domain faces without Biot-Savart BCs (e.g. `(-2,)` for the lower y face).
- `symmetry`: domain faces that are symmetry planes (e.g. `(-2,)`). Like `nonbiotfaces` they
  get no Biot-Savart BCs, but the images of the vorticity across them are included.
  At most one face per direction.

Not supported (`ArgumentError`): `fmm=false`, periodic directions (`perdir`), and grids
too small for a second Biot-Savart level.
"""
function BiotSimulation(args...; nonbiotfaces=(), fmm=true, symmetry=(), mem=CuArray, kwargs...)
    WaterLily.Simulation(args...; mem=CuArray,
        flow_ctor=(dims,uBC;kw...)->TFlow(dims,uBC; bdimfaces=(nonbiotfaces..., symmetry...), kw...),
        pois_ctor=flow->TBiotPoisson(flow; nonbiotfaces, fmm, symmetry), kwargs...)
end
