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
struct TBiotPoisson{T,S,V,D,P<:TPoisson,W,G} <: AbstractPoisson{T,S,V}
    pois :: P
    x :: S
    L :: V
    z :: S
    ω :: W                          # vorticity levels 2:NL
    g :: G                          # NTuple of level grids (level 1 first)
    tar :: Vector{CuVector{Int32}}  # target (face) linear indices per level
    p :: S                          # pressure accumulator
    n :: Vector{Int16}
    perdir :: NTuple
    fmm :: Bool
    red :: RedBuf
end
function TBiotPoisson(flow::TFlow{D,T}; nonbiotfaces=(), fmm=true) where {D,T}
    isempty(flow.perdir) || @warn "BiotSavart BCs ignore perdir" maxlog=1
    pois = TPoisson(flow)
    ml = BiotSavartBCs.MLArray(flow.f)
    ω = Tuple(ml[2:end])
    g = map(a->Grid(Base.front(size(a))), Tuple(ml))
    tars = BiotSavartBCs.collect_targets(ml, nonbiotfaces)
    tar = [CuVector{Int32}(Int32.(LinearIndices(size(ml[l]))[tars[l]])) for l in eachindex(ml)]
    p = copy(flow.p)
    TBiotPoisson{T,typeof(flow.p),typeof(flow.μ₀),D,typeof(pois),typeof(ω),typeof(g)}(
        pois, pois.x, pois.L, pois.z, ω, g, tar, p, pois.n, flow.perdir, fmm, pois.red)
end
WaterLily.update!(b::TBiotPoisson) = WaterLily.update!(b.pois)
Base.getproperty(b::TBiotPoisson, s::Symbol) = s === :levels ? getfield(b, :pois).levels : getfield(b, s)

# ---------------------------------------------------------------- vorticity
# centred curl at cell I (BiotSavartBCs.centered_curl): 3D all components, 2D ω₃ in slot 1
@inline function dcross(i, j, l, u, g::Grid)   # ∂uᵢ/∂xⱼ at the cell centre, i ≠ j
    ci = (i-i32)*g.nc; sj, si = g.s[j], g.s[i]
    @inbounds return (u[l+sj+ci] + u[l+sj+si+ci] - u[l-sj+ci] - u[l-sj+si+ci])*0.25f0
end
@inline function curl_c(i, l, u, g::Grid{3})
    j = i % Int32(3) + i32; k = (i + i32) % Int32(3) + i32
    return dcross(k, j, l, u, g) - dcross(j, k, l, u, g)
end
@inline curl_c(i, l, u, g::Grid{2}) = i == i32 ? dcross(Int32(2), i32, l, u, g) - dcross(i32, Int32(2), l, u, g) : 0f0

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
    for l in 3:length(g)
        launch(k_vrestrict!, inside_lo(g[l]), inside_hi(g[l]), b.ω[l-1], g[l], b.ω[l-2], g[l-1])
    end
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
        dT = (T[d] + Int32(2)) ÷ Int32(2)
        v = bottom ? Int32(2) : Int32(2)*max(dT - c, Int32(2)) - Int32(2)
        top ? max(v, Int32(3)) : v
    end
end
@inline function outer_hi(T::NTuple{D,Int32}, g::Grid{D}, c, bottom, top) where D
    return ntuple(Val(D)) do d
        dT = (T[d] + Int32(2)) ÷ Int32(2)
        v = bottom ? g.n[d] - i32 : Int32(2)*min(dT + c, (g.n[d] + i32) ÷ Int32(2)) - i32
        top ? min(v, g.n[d] - Int32(2)) : v
    end
end
# Interaction sum at one target of one level (BiotSavartBCs.interaction).
# P consecutive lanes share a target, splitting the outermost source loop; P ∈ {1,2,4,…,32}.
function k_interact!(n, tar, ω, g::Grid{D}, top::Bool, bottom::Bool, P::Int32) where D
    t = tid1() - i32
    k = t ÷ P + i32; p = t % P
    valid = k <= n
    kk = valid ? k : n
    li = @inbounds tar[kk]
    i = (li - i32) ÷ g.nc + i32
    l = li - (i-i32)*g.nc
    T = coords(g, l)
    c = closeR(Val(D))
    olo = outer_lo(T, g, c, bottom, top)
    ohi = outer_hi(T, g, c, bottom, top)
    ilo = ntuple(d-> max(T[d] - c, Int32(2)), Val(D))
    ihi = ntuple(d-> min(T[d] + c, g.n[d] - i32), Val(D))
    x = ntuple(d-> Float32(T[d]) + (d == i ? (T[d] == i32 ? 0.5f0 : -0.5f0) : 0f0), Val(D))
    val = 0f0
    if valid
        if D == 3
            s3 = olo[3] + p
            while s3 <= ohi[3]
                for s2 in olo[2]:ohi[2], s1 in olo[1]:ohi[1]
                    if !top
                        (ilo[1] <= s1 <= ihi[1]) & (ilo[2] <= s2 <= ihi[2]) & (ilo[3] <= s3 <= ihi[3]) && continue
                    end
                    ls = s1 + (s2 - i32)*g.s[2] + (s3 - i32)*g.s[3]
                    val += bs_weight(x[1]-s1, x[2]-s2, x[D]-s3, ω, ls, g, i)
                end
                s3 += P
            end
        else
            s2 = olo[2] + p
            while s2 <= ohi[2]
                for s1 in olo[1]:ohi[1]
                    if !top
                        (ilo[1] <= s1 <= ihi[1]) & (ilo[2] <= s2 <= ihi[2]) && continue
                    end
                    ls = s1 + (s2 - i32)*g.s[2]
                    val += bs_weight(x[1]-s1, x[2]-s2, 0f0, ω, ls, g, i)
                end
                s2 += P
            end
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
function fmm!(b::TBiotPoisson, ω1)
    ωs = (ω1, b.ω...)
    nl = length(ωs)
    for l in 1:nl
        n = length(b.tar[l])
        P = clamp(prevpow(2, max(1, 65536 ÷ n)), 1, 32)
        @cuda threads=256 blocks=cld(n*P, 256) k_interact!(Int32(n), b.tar[l], ωs[l], b.g[l], l == 1, l == nl, Int32(P))
    end
    for l in nl-1:-1:1
        launch1(k_project!, length(b.tar[l]), b.tar[l], ωs[l], b.g[l], ωs[l+1], b.g[l+1])
    end
end

# ---------------------------------------------------------------- domain boundary updates
# u at the boundary face of target (I,i): index 2 (low) or n (high)
@inline function face_of(li, g::Grid)
    i = (li - i32) ÷ g.nc + i32
    l = li - (i-i32)*g.nc
    lower = (l - i32) ÷ g.s[i] % g.n[i] == 0   # I[i]==1
    return i, l, lower
end
# u = U + uᵥ; with `resid`, also r[adjacent] += Δuₙ·(lower ? -1 : 1)
function k_biotbc!(n, tar, u, uv, r, U, g::Grid, resid::Bool)
    k = tid1(); k <= n || return
    li = @inbounds tar[k]
    i, l, lower = face_of(li, g)
    lu = (lower ? l + g.s[i] : l) + (i-i32)*g.nc
    un = @inbounds U[i] + uv[li]
    if resid
        un0 = @inbounds u[lu]
        CUDA.@atomic r[lower ? l + g.s[i] : l - g.s[i]] += (un - un0)*(lower ? -1f0 : 1f0)
    end
    @inbounds u[lu] = un
    return nothing
end
# global flux fix: u[face] += fix·(lower ? 1 : -1), r[adjacent] -= fix, fix = Σr/ntar
function k_fixresid!(n, tar, u, r, sums, g::Grid)
    k = tid1(); k <= n || return
    fix = Float32(@inbounds(sums[1])) / Float32(n)
    li = @inbounds tar[k]
    i, l, lower = face_of(li, g)
    lu = (lower ? l + g.s[i] : l) + (i-i32)*g.nc
    @inbounds u[lu] += fix*(lower ? 1f0 : -1f0)
    CUDA.@atomic r[lower ? l + g.s[i] : l - g.s[i]] -= fix
    return nothing
end
function k_sum!(lo, hi, ps, pm, r, g::Grid)
    I, ok = cellidx(lo, hi)
    s = ok ? @inbounds(r[lin(g, I)]) : 0f0
    block_partial!(ps, pm, s, 0f0)
    return nothing
end
function fix_resid!(b::TBiotPoisson, u, r)
    g = b.g[1]
    red_launch(k_sum!, ntuple(_->1, length(g.n)), Int.(g.n), b.red, 1, r, g)
    launch1(k_fixresid!, length(b.tar[1]), b.tar[1], u, r, b.red.s, g)
end
function k_sumsq!(lo, hi, ps, pm, r, g::Grid)
    I, ok = cellidx(lo, hi)
    v = ok ? @inbounds(r[lin(g, I)]) : 0f0
    block_partial!(ps, pm, v*v, abs(v))
    return nothing
end
function L₂!(b::TBiotPoisson, r)
    g = b.g[1]
    red_launch(k_sumsq!, ntuple(_->1, length(g.n)), Int.(g.n), b.red, 4, r, g)
    fetch!(b.red)
    return getsum(b.red, 4), getmax(b.red, 4)
end
# r = iD==0 ? 0 : div(u) (interior), 0 on ghosts
function k_divres!(lo, hi, r, u, op::Op{D}) where D
    I, ok = cellidx(lo, hi); ok || return
    g = op.g; l = lin(g, I)
    interior = true
    @unroll for d in i32:Int32(D)
        interior &= (I[d] > i32) & (I[d] < g.n[d])
    end
    v = 0f0
    if interior && getiD(op, l) != 0
        @unroll for i in i32:Int32(D)
            ci = (i-i32)*g.nc
            @inbounds v += u[l+g.s[i]+ci] - u[l+ci]
        end
    end
    @inbounds r[l] = v
    return nothing
end
# u -= L∂x ; pacc += x   (BiotSavartBCs.project_update!, x is then logically zero)
function k_projupd!(lo, hi, u, x, pacc, op::Op{D}) where D
    I, ok = cellidx(lo, hi); ok || return
    g = op.g
    l = lin(g, I)
    rg = isreg(op, l)
    @unroll for i in i32:Int32(D)
        ci = (i-i32)*g.nc
        L = rg ? op.Lreg : @inbounds(op.L[l+ci])
        @inbounds u[l+ci] -= L*(x[l]-x[l-g.s[i]])
    end
    @inbounds pacc[l] += x[l]
    return nothing
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

# ---------------------------------------------------------------- projection
bctuple(f::Function, t::T, N) where T = ntuple(i->f(i, zero(SVector{N,T}), t), N)
bctuple(f::Tuple, t, N) = f
function biot_update!(b::TBiotPoisson, a::TFlow, U, r)
    fill_ω!(b, a.f, a.u)
    b.fmm ? fmm!(b, a.f) : error("TurboLily: fmm=false (tree sum) is not implemented")
    launch1(k_biotbc!, length(b.tar[1]), b.tar[1], a.u, a.f, r, U, b.g[1], r !== nothing)
end
function WaterLily.mom_project!(a::TFlow{D,T}, b::TBiotPoisson, w, t; tol=1e-4, itmx=32) where {D,T}
    dt = T(w)*a.Δt[end]; a.p .*= dt
    U = T.(bctuple(a.uBC, T(t), D))
    top = b.levels[1]; op = top.op; g = op.g
    fill!(b.p, 0); fill!(b.pois.ωd, 1)
    launch(k_projupd!, inside_lo(g), inside_hi(g), a.u, a.p, b.p, op)
    fill_ω!(b, a.f, a.u); fmm!(b, a.f)
    launch1(k_biotbc!, length(b.tar[1]), b.tar[1], a.u, a.f, top.r, U, g, false)
    launch(k_divres!, (1,1,1)[1:D], Int.(g.n), top.r, a.u, op)
    fix_resid!(b, a.u, top.r)
    r₂, r∞ = L₂!(b, top.r)
    nᵖ, nᵇ = 0, 0; xzero = true
    @log ", $nᵖ, $r∞, $r₂, $nᵇ\n"
    while nᵖ < itmx
        rtol = max(tol, 0.1r₂)
        while nᵖ < itmx
            iteration!(b.pois; assign=xzero, sq=true, adapt=false); xzero = false
            fetch!(b.red); r₂ = getsum(b.red, 3); nᵖ += 1
            r₂ < rtol && break
        end
        launch(k_projupd!, inside_lo(g), inside_hi(g), a.u, top.x, b.p, op); xzero = true
        fill_ω!(b, a.f, a.u); fmm!(b, a.f)
        launch1(k_biotbc!, length(b.tar[1]), b.tar[1], a.u, a.f, top.r, U, g, true)
        fix_resid!(b, a.u, top.r)
        r₂, r∞ = L₂!(b, top.r); nᵇ += 1
        @log ", $nᵖ, $r∞, $r₂, $nᵇ\n"
        r₂ < tol && break
    end
    push!(b.n, nᵖ)
    pflowBC!(a.u, g)
    a.p .= b.p ./ dt
end

"""
    TurboLily.BiotSimulation(dims, uBC, L; nonbiotfaces=(), fmm=true, kwargs...)

Same signature as `BiotSavartBCs.BiotSimulation`; returns a `WaterLily.Simulation` with
the CUDA flow and Biot-Savart pressure solver.
"""
function BiotSimulation(args...; nonbiotfaces=(), fmm=true, mem=CuArray, kwargs...)
    WaterLily.Simulation(args...; mem=CuArray,
        flow_ctor=(dims,uBC;kw...)->TFlow(dims,uBC;kw...),
        pois_ctor=flow->TBiotPoisson(flow; nonbiotfaces, fmm), kwargs...)
end
