# Fused body measurement and device-side updates of everything derived from μ₀, μ₁, V.

"""
    wappend!(list, cnt, slot, pred, val)

Warp-aggregated append of `val` (for lanes with `pred`) to `list`, counting in `cnt[slot]`.
All 32 lanes of the warp must call it. Entries past `length(list)` are dropped (the count
still grows, so the caller can detect the overflow and retry with a larger list).
"""
@inline function wappend!(list, cnt, slot, pred::Bool, val::Int32)
    mask = vote_ballot_sync(0xffffffff, pred)
    mask == 0 && return nothing
    lane = (flat_tid() % Int32(32))
    leader = Int32(trailing_zeros(mask))
    rank = Int32(count_ones(mask & ((UInt32(1) << lane) - UInt32(1))))
    base = Int32(0)
    if lane == leader
        base = CUDA.atomic_add!(pointer(cnt, slot), Int32(count_ones(mask)))
    end
    base = shfl_sync(0xffffffff, base, leader + i32)
    k = base + rank + i32
    (pred && k <= length(list)) && (@inbounds list[k] = val)
    return nothing
end

# WaterLily.measure! in two passes. The first evaluates the distance in every cell and updates
# only cells away from the body that were near it, or that changed side since the last measure
# (a body that moved further than the band in between); it lists the cells near the body. The
# second measures the listed cells (normal and body velocity), so the costly `measure` code
# does not set the register use of the pass over all cells.
# `near` holds 0x01 for a cell near the body, 0x00 / 0x02 for a cell away from it outside / inside.
# Both append the BDIM band faces (class 1: μ₀≠1, μ₁=0; class 2: μ₁≠0); `stat` counts
# [band, shell, V≠0, near cells].
function k_measure!(lo, hi, μ₀, μ₁, V, σ, near_prev, body::B, t::T, d²::T, g::Grid{D},
                    stat, band, nearl, bdimlow::NTuple{D,Bool}) where {B,T,D}
    I, ok = cellidx(lo, hi)
    l = ok ? lin(g, I) : i32
    near = false
    d = zero(T)
    if ok
        d = WaterLily.sdf(body, loc(0, I, T), t; fastd²=d²)
        @inbounds σ[l] = d
        near = d^2 < d²
    end
    wappend!(nearl, stat, 4, near, l)
    prev = ok ? @inbounds(near_prev[l]) : 0x00
    @unroll for i in i32:Int32(D)
        li = l + (i-i32)*g.nc
        cls = 0
        if ok & !near
            m0 = d < 0 ? zero(T) : one(T)
            if prev != (d < 0 ? 0x02 : 0x00)    # was near, or changed side
                @inbounds V[li] = zero(T)
                @inbounds μ₀[li] = m0
                @unroll for j in i32:Int32(D)
                    @inbounds μ₁[li + (j-i32)*D*g.nc] = zero(T)
                end
            end
            cls = m0 != 1 ? 1 : 0
            I[i] == 2i32 && (cls = bdimlow[i] ? max(cls, 1) : 0)
        end
        wappend!(band, stat, 1, cls == 1, li)
    end
    ok && (@inbounds near_prev[l] = near ? 0x01 : d < 0 ? 0x02 : 0x00)
    return nothing
end
function k_measure_near!(cap, nearl, μ₀, μ₁, V, σ, body::B, t::T, ϵ::T, d²::T, g::Grid{D},
                         stat, band, shell, bdimlow::NTuple{D,Bool}) where {B,T,D}
    k = tid1()
    ok = k <= min(@inbounds(stat[4]), cap)
    l = ok ? @inbounds(nearl[k]) : i32
    I = coords(g, l)
    d = ok ? @inbounds(σ[l]) : zero(T)
    vnz = false
    @unroll for i in i32:Int32(D)
        li = l + (i-i32)*g.nc
        cls = 0
        if ok
            dᵢ, nᵢ, Vᵢ = WaterLily.measure(body, loc(i, I, T), t; fastd²=d²)
            dᵢ = abs(dᵢ) ≤ T(0.5) ? dᵢ : copysign(dᵢ, d)
            @inbounds V[li] = Vᵢ[i]
            vnz |= Vᵢ[i] != 0
            m0 = WaterLily.μ₀(dᵢ, ϵ)
            @inbounds μ₀[li] = m0
            m1 = WaterLily.μ₁(dᵢ, ϵ)
            sh = false
            @unroll for j in i32:Int32(D)
                v = m1*nᵢ[j]
                @inbounds μ₁[li + (j-i32)*D*g.nc] = v
                sh |= v != 0
            end
            cls = sh ? 2 : m0 != 1 ? 1 : 0
            # lower boundary face: BC! overwrites u there, except where `bdimlow` (an interior
            # face in a periodic direction, or a non-Biot-Savart face of a Biot-Savart
            # simulation, where μ₀ is zeroed after this kernel by vecBC0!)
            I[i] == 2i32 && (cls = bdimlow[i] ? max(cls, 1) : 0)
        end
        wappend!(band, stat, 1, cls == 1, li)
        wappend!(shell, stat, 2, cls == 2, li)
    end
    vnz && (@inbounds stat[3] = Int32(1))
    return nothing
end

# zero boundary values for vector fields (WaterLily.BC!(a, zeros, saveexit, perdir))
function vecBC0!(a, g::Grid{D}, per, saveexit) where D
    z = ConstBC{D,eltype(a)}(ntuple(_->zero(eltype(a)), D))
    for j in 1:D
        lo = ntuple(_->1, D); hi = ntuple(d-> d==j ? 1 : Int(g.n[d]), D)
        launch_face(k_bc!, lo, hi, j, a, g, z, zero(eltype(a)), saveexit, per[j])
    end
end

function WaterLily.measure!(a::TFlow{D,T}, body::AbstractBody; t=zero(T), ϵ=1) where {D,T}
    g = a.grid
    while true
        fill!(a.mstat, 0)
        d² = T((2+ϵ)^2)
        launch(k_measure!, inside_lo(g), inside_hi(g), a.μ₀, a.μ₁, a.V, a.σ, a.near, body, T(t),
               d², g, a.mstat, a.band, a.nearl, a.bdimlow)
        # one thread per entry of the near list; threads past the count (on the device) return
        launch1(k_measure_near!, length(a.nearl), a.nearl, a.μ₀, a.μ₁, a.V, a.σ, body, T(t), T(ϵ),
                d², g, a.mstat, a.band, a.shell, a.bdimlow)
        vecBC0!(a.μ₀, g, a.per, false)
        vecBC0!(a.V, g, a.per, a.exitBC)
        st = hostcopy!(Vector{Int32}(undef, length(a.mstat)), a.mstat)
        a.nband, a.nshell = st[1], st[2]
        a.Vzero = st[3] == 0
        (a.nband <= length(a.band) && a.nshell <= length(a.shell) && st[4] <= length(a.nearl)) && break
        # lists too small: grow and measure again (every cell near the body is rewritten,
        # so a repeated measure gives the same result)
        a.band = CuVector{Int32}(undef, 2a.nband + 1024)
        a.shell = CuVector{Int32}(undef, 2a.nshell + 1024)
        a.nearl = CuVector{Int32}(undef, 2st[4] + 1024)
        a.near .= 0x01
    end
    return nothing
end
WaterLily.measure!(a::TFlow, ::NoBody; kwargs...) = nothing

# ---------------------------------------------------------------- Poisson level update
# D = -Σ(L[I,j]+L[I+δj,j]), iD = D==0 ? 0 : 1/D, reg = all faces == Lreg; partial count(iD≠0)
function k_diag!(lo, hi, ps, pm, L, Dg, iD, reg, g::Grid{D}, Lreg) where D
    I, ok = cellidx(lo, hi)
    c = 0f0
    if ok
        l = lin(g, I)
        s = zero(eltype(Dg)); rg = true
        @unroll for j in i32:Int32(D)
            cj = (j-i32)*g.nc
            @inbounds a = L[l+cj]; @inbounds b = L[l+g.s[j]+cj]
            s -= a + b
            rg &= (a == Lreg) & (b == Lreg)
        end
        @inbounds Dg[l] = s
        @inbounds iD[l] = iszero(s) ? s : inv(s)
        @inbounds reg[l] = rg ? 0x01 : 0x00
        c = iszero(s) ? 0f0 : 1f0
    end
    block_partial!(ps, pm, c, 0f0)
    return nothing
end
# coarse L from fine L (WaterLily.restrictL with the semi-coarsening mask c)
function k_restrictL!(lo, hi, a, ga::Grid{D}, b, gb::Grid{D}, c) where D
    I, ok = cellidx(lo, hi); ok || return
    la = lin(ga, I)
    @unroll for i in i32:Int32(D)
        J0 = ntuple(d-> d == i ? (c[d] ? Int32(2)*I[d]-Int32(2) : I[d]) : (c[d] ? Int32(2)*I[d]-Int32(2) : I[d]), Val(D))
        e = ntuple(d-> (d != i && c[d]) ? i32 : Int32(0), Val(D))
        lb = lin(gb, J0) + (i-i32)*gb.nc
        s = zero(eltype(a))
        if D == 3
            for dz in 0:e[3], dy in 0:e[2], dx in 0:e[1]
                @inbounds s += b[lb + dx + dy*gb.s[2] + dz*gb.s[3]]
            end
        else
            for dy in 0:e[2], dx in 0:e[1]
                @inbounds s += b[lb + dx + dy*gb.s[2]]
            end
        end
        @inbounds a[la + (i-i32)*ga.nc] = c[i] ? s/2 : s
    end
    return nothing
end
function WaterLily.update!(p::TPoisson{T}) where T
    for l in eachindex(p.levels)
        lv = p.levels[l]; op = lv.op; g = op.g
        if l > 1
            f = p.levels[l-1]
            launch(k_restrictL!, inside_lo(g), inside_hi(g), op.L, g, f.op.L, f.op.g, f.c)
            # Without periodic directions BC! changes nothing that is read: the boundary
            # faces at index 2 are restricted from zero fine faces, those at n are never
            # written, and no stencil reads the tangential ghosts of L.
            any(j-> p.per[j], 1:length(g.n)) && vecBC0!(op.L, g, p.per, false)
        end
        checkcap(p.red, nblocks(inside_lo(g), inside_hi(g)))
        nb = launch(k_diag!, inside_lo(g), inside_hi(g), p.red.ps, p.red.pm, op.L, op.Dg, op.iD, op.reg, g, op.Lreg)
        set_deep!(op)
        l == 1 && finish!(p.red, nb, p.nsd, p.red.m, 1)
    end
    return p
end
