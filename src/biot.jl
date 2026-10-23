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
struct TBiotPoisson{T,S,V,D,P<:TPoisson,W,G,W1,K} <: AbstractPoisson{T,S,V}
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
    lists :: Vector{CuVector{Int32}}  # [A1, A2, B]: affected layer cells, affected level-2 cells, cells next to the boundary
    dω :: W1                          # Δω on all levels (level 1 first) for incremental FMM updates
    boxes :: Vector{Any}              # per level (lo,hi) bounding box of Δω
    mask :: CuArray{UInt8,D}          # affected-cell mask on level 1
    bst :: CuVector{Int32}            # [nA1, box1 lo/hi, nA2, box2 lo/hi]
    nA :: Vector{Int}                 # host copies of the A1, A2 counts
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
    dω = map(a->(z = similar(a); fill!(z, 0); z), Tuple(ml))
    b = TBiotPoisson{T,typeof(flow.p),typeof(flow.μ₀),D,typeof(pois),typeof(ω),typeof(g),typeof(dω),length(symmetry)}(
        pois, pois.x, pois.L, pois.z, ω, g, tar, tadj, p, pois.n, flow.perdir, fmm, pois.red,
        [CuVector{Int32}(undef,256), CuVector{Int32}(undef,256), CuVector{Int32}(B)],
        dω, Any[], CUDA.zeros(UInt8, size(flow.p)), CUDA.zeros(Int32, 16), [0, 0],
        Int32.(Tuple(symmetry)))
    affected_lists!(b)
    return b
end
WaterLily.update!(b::TBiotPoisson) = (WaterLily.update!(b.pois); affected_lists!(b))
Base.getproperty(b::TBiotPoisson, s::Symbol) = s === :levels ? getfield(b, :pois).levels : getfield(b, s)

# Cells whose centred-curl stencil reads a face with L≠1: there the curl of Δt·L∇p is not
# zero, so ω must be recomputed when p changes. Elsewhere curl(L∇p) ≡ 0 exactly.
# Writes the mask m (0x01 affected, 0x03 affected and in the level-1 layer) and appends
# the layer cells to A1 (count bst[1], box bst[2:7]); level-2 list A2 uses bst[8:14].
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
function k_affected!(lo, hi, m, L, g::Grid{D}, w, A1, bst) where D
    I, ok = cellidx(lo, hi)
    aff = false; near = false
    l = ok ? lin(g, I) : i32
    if ok
        @unroll for a in i32:Int32(D)
            ca = (a-i32)*g.nc
            @unroll for b in i32:Int32(D)
                if a != b
                    sb, sa = g.s[b], g.s[a]
                    @inbounds aff |= (L[l+sb+ca] != 1) | (L[l-sb+ca] != 1) | (L[l+sb+sa+ca] != 1) | (L[l-sb+sa+ca] != 1)
                end
            end
        end
        @unroll for d in i32:Int32(D)
            near |= (I[d] <= w) | (I[d] >= g.n[d] - w + i32)
        end
        @inbounds m[l] = aff ? (near ? 0x03 : 0x01) : 0x00
    end
    wappend!(A1, bst, 1, aff & near, l)
    warp_box!(bst, 1, aff & near, I)
    return nothing
end
function k_affected2!(lo, hi, m, g::Grid{D}, g2::Grid{D}, A2, bst) where D
    I, ok = cellidx(lo, hi)
    aff = false
    if ok
        J0 = lin(g, ntuple(d-> Int32(2)*I[d] - Int32(2), Val(D)))
        if D == 3
            for dz in 0:1, dy in 0:1, dx in 0:1
                @inbounds aff |= m[J0 + dx + dy*g.s[2] + dz*g.s[D]] != 0
            end
        else
            for dy in 0:1, dx in 0:1
                @inbounds aff |= m[J0 + dx + dy*g.s[2]] != 0
            end
        end
    end
    wappend!(A2, bst, 8, aff, ok ? lin(g2, I) : i32)
    warp_box!(bst, 8, aff, I)
    return nothing
end
function k_clear_delta!(n, list, dω, g::Grid{D}) where {D}
    k = tid1(); k <= n || return
    l = @inbounds list[k]
    @unroll for i in i32:Int32(D)
        @inbounds dω[l + (i - i32) * g.nc] = zero(eltype(dω))
    end
    return nothing
end
function affected_lists!(b::TBiotPoisson{T, S, V, D}) where {T, S, V, D}
    g = b.g; L = b.L
    # Cells leaving the lists must contribute zero to subsequent Δω restrictions.
    for l in 1:2
        launch1(k_clear_delta!, b.nA[l], b.lists[l], b.dω[l], g[l])
    end
    while true
        st0 = vcat(Int32[0], fill(typemax(Int32),3), fill(typemin(Int32),3), Int32[0], fill(typemax(Int32),3), fill(typemin(Int32),3), Int32[0, 0])
        copyto!(b.bst, st0)
        launch(k_affected!, ntuple(_->3,D), ntuple(d->Int(g[1].n[d])-2,D), b.mask, L, g[1], Int32(layer_width(Val(D))), b.lists[1], b.bst)
        launch(k_affected2!, inside_lo(g[2]), inside_hi(g[2]), b.mask, g[1], g[2], b.lists[2], b.bst)
        st = hostcopy!(Vector{Int32}(undef, length(b.bst)), b.bst)
        b.nA[1], b.nA[2] = st[1], st[8]
        if b.nA[1] > length(b.lists[1]) || b.nA[2] > length(b.lists[2])
            b.lists[1] = CuVector{Int32}(undef, 2b.nA[1] + 256)
            b.lists[2] = CuVector{Int32}(undef, 2b.nA[2] + 256)
            continue
        end
        # bounding boxes of Δω per level (empty box: lo > hi)
        box(o) = st[o+1] > st[o+4] ? (ntuple(_->2,D), ntuple(_->1,D)) : (Tuple(st[o+1:o+D]), Tuple(st[o+4:o+3+D]))
        boxes = Any[box(1), box(8)]
        for l in 3:length(g)
            lo, hi = boxes[end]
            push!(boxes, lo[1] > hi[1] ? (lo, hi) : (map(i->(i+2)÷2, lo), map(i->(i+2)÷2, hi)))
        end
        empty!(b.boxes); append!(b.boxes, boxes)
        return b
    end
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
@inline function target_sum(T::NTuple{D,Int32}, i, ω, g::Grid{D}, top, bottom, blo, bhi, p, P) where D
    c = closeR(Val(D))
    olo0 = outer_lo(T, g, c, bottom, top)
    ohi0 = outer_hi(T, g, c, bottom, top)
    olo = ntuple(d-> max(olo0[d], blo[d]), Val(D))
    ohi = ntuple(d-> min(ohi0[d], bhi[d]), Val(D))
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
function k_interact!(n, tar, ω, g::Grid{D}, top::Bool, bottom::Bool, P::Int32, blo, bhi, imgs::NTuple{K,Int32}) where {D,K}
    t = tid1() - i32
    k = t ÷ P + i32; p = t % P
    valid = k <= n
    kk = valid ? k : n
    li = @inbounds tar[kk]
    i = (li - i32) ÷ g.nc + i32
    l = li - (i-i32)*g.nc
    T = coords(g, l)
    val = 0f0
    if valid
        val = target_sum(T, i, ω, g, top, bottom, blo, bhi, p, P)
        @unroll for m in i32:Int32(2^K - 1)   # empty unless images are requested
            Tm = T; sgn = 1f0
            @unroll for b in i32:Int32(K)
                if (m >> (b - i32)) & i32 == i32
                    Tm, sb = reflect(Tm, imgs[b], i, g)
                    sgn *= sb
                end
            end
            val += sgn*target_sum(Tm, i, ω, g, top, bottom, blo, bhi, p, P)
        end
    end
    off = P ÷ Int32(2)
    while off > 0
        val += shfl_down_sync(0xffffffff, val, off)
        off >>= 1
    end
    (valid && p == 0) && (@inbounds ω[li] = val*bs_scale(Val(D)))
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
    fmm!(b, ωs, boxes=nothing)

Biot-Savart interactions on all levels of `ωs` (level-1 first, sources restricted to
`boxes[l]` when given) and the coarse-to-fine projection; level-1 target slots of `ωs[1]`
then hold the induced boundary velocity.
"""
function fmm!(b::TBiotPoisson{T,S,V,D}, ωs, boxes=nothing) where {T,S,V,D}
    nl = length(ωs)
    for l in 1:nl
        n = length(b.tar[l])
        P = clamp(prevpow(2, max(1, 65536 ÷ n)), 1, 32)
        blo, bhi = boxes === nothing ? (ntuple(_->1,D), Int.(b.g[l].n)) : boxes[l]
        @cuda threads=256 blocks=cld(n*P, 256) k_interact!(Int32(n), b.tar[l], ωs[l], b.g[l], l == 1, l == nl, Int32(P), Int32.(blo), Int32.(bhi), b.images)
    end
    for l in nl-1:-1:1
        launch1(k_project!, length(b.tar[l]), b.tar[l], ωs[l], b.g[l], ωs[l+1], b.g[l+1])
    end
end
fmm!(b::TBiotPoisson, ω1::AbstractArray) = fmm!(b, (ω1, b.ω...))

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

# ---------------------------------------------------------------- incremental vorticity
# recompute ω on the affected cells from u - Δt·L∇p
# component i of the curl, with i a compile-time constant (static tuple indexing)
@inline curl_ci(i, l, U::F, g::Grid{3}) where F = i == i32 ? curl_c(i32, l, U, g) : i == Int32(2) ? curl_c(Int32(2), l, U, g) : curl_c(Int32(3), l, U, g)
@inline curl_ci(i, l, U::F, g::Grid{2}) where F = i == i32 ? curl_c(i32, l, U, g) : curl_c(Int32(2), l, U, g)
# one thread per (listed cell, component)
function k_curl_corr1!(nt, n, list, ω, dω, U::F, g::Grid{D}) where {D,F}
    t = tid1() - i32; t < nt || return
    k = t ÷ Int32(D) + i32; i = t % Int32(D) + i32
    l = @inbounds list[k]
    li = l + (i-i32)*g.nc
    v = curl_ci(i, l, U, g)
    @inbounds dω[li] = v - ω[li]
    @inbounds ω[li] = v
    return nothing
end
# Level-2 ω of a listed coarse cell: sum of the curls of its 2ᵈ fine cells inside 3:n-2.
# One thread per (cell, component, fine cell); the first lane of each group of 2ᵈ adds the
# group's curls in the order (dz, dy, dx) of a nested loop.
function k_curl_corr2!(nt, n, list, ω2, dω2, g2::Grid{D}, U::F, g::Grid{D}) where {D,F}
    NS = Int32(2^D)
    t = tid1() - i32
    grp = t ÷ NS; sub = t % NS
    k = grp ÷ Int32(D) + i32; i = grp % Int32(D) + i32
    valid = t < nt
    l2 = @inbounds list[valid ? k : i32]
    I = coords(g2, l2)
    J = ntuple(d-> Int32(2)*I[d] - Int32(2) + ((sub >> (d-1)) & i32), Val(D))
    inb = valid
    @unroll for d in i32:Int32(D)
        inb &= (J[d] >= Int32(3)) & (J[d] <= g.n[d] - Int32(2))
    end
    v = inb ? curl_ci(i, lin(g, J), U, g) : 0f0
    base = (threadIdx().x - i32) & ~(NS - i32)          # first lane of the group in the block
    lane0 = base % Int32(32) + i32
    s = 0f0
    for q in Int32(0):NS-i32
        vq = shfl_sync(0xffffffff, v, lane0 + q)
        bq = shfl_sync(0xffffffff, Int32(inb), lane0 + q)
        bq != 0 && (s += vq)
    end
    if valid & (sub == 0)
        li2 = l2 + (i-i32)*g2.nc
        @inbounds dω2[li2] = s - ω2[li2]
        @inbounds ω2[li2] = s
    end
    return nothing
end
"""
    correct_ω!(b, ω1, u, p, dt, incremental)

Recompute ω of `u - Δt·L∇p` on the affected cells (levels 1 and 2), storing the change in
`b.dω`. Levels ≥3 are then restricted from ω (full update) or from Δω (incremental).
"""
function correct_ω!(b::TBiotPoisson, ω1, u, p, dt, incremental::Bool)
    g = b.g; op = b.levels[1].op
    U = UC(u, p, op, dt)
    D = length(g[1].n)
    launch1(k_curl_corr1!, D*b.nA[1], b.nA[1], b.lists[1], ω1, b.dω[1], U, g[1])
    launch1(k_curl_corr2!, 2^D*D*b.nA[2], b.nA[2], b.lists[2], b.ω[1], b.dω[2], g[2], U, g[1])
    src = incremental ? b.dω : (ω1, b.ω...)
    for l in 3:length(g)
        launch(k_vrestrict!, inside_lo(g[l]), inside_hi(g[l]), src[l], g[l], src[l-1], g[l-1])
    end
end
# a[target] += d[target] on level 1 (incremental boundary velocity)
function k_addtar!(n, tar, a, d)
    k = tid1(); k <= n || return
    li = @inbounds tar[k]
    @inbounds a[li] += d[li]
    return nothing
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
the end, the vorticity of the projected field is recomputed only where `curl(L∇p)≠0`,
and the global flux fix and the residual norms are updated from the boundary cells.
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
    correct_ω!(b, a.f, a.u, a.p, dt, false)
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
        correct_ω!(b, a.f, a.u, a.p, dt, true)
        fmm!(b, b.dω, b.boxes)
        launch1(k_addtar!, nt, tar1, a.f, b.dω[1])
        list_norms!(b, b.lists[3], top.r, 5)
        nb = cld(nt, 256); checkcap(red, nb)
        @cuda threads=256 blocks=nb k_biotbc2!(Int32(nt), red.ps, red.pm, tar1, b.tadj, a.u, a.f, top.r, U, g, idt)
        finish!(red, nb, red.s, red.m, 6)
        launch1(k_fixresid2!, nt, tar1, b.tadj, a.u, top.r, red.s, Int32(6), 1f0, idt, g)
        list_norms!(b, b.lists[3], top.r, 7)
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
