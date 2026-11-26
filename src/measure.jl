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

# Initialize `mstat`: counts 1:4 = 0, dirty box 5:10 and transition box 11:16 = [max..max, min..min].
function k_mstat_init!(stat)
    @inbounds for k in 1:4; stat[k] = Int32(0); end
    @inbounds for k in 5:7; stat[k] = typemax(Int32); end
    @inbounds for k in 8:10; stat[k] = typemin(Int32); end
    @inbounds for k in 11:13; stat[k] = typemax(Int32); end
    @inbounds for k in 14:16; stat[k] = typemin(Int32); end
    return nothing
end

# Pure signed-distance evaluation without ForwardDiff: WaterLily.sdf falls back to
# `measure(body, ...)[1]` for `SetBody`, which would pull ForwardDiff and local memory
# into the full-grid `k_measure!` kernel.
@inline fast_sdf(body::WaterLily.AutoBody, x, t) = @inline body.sdf((@inline body.map(x, t)), t)
@inline fast_sdf(body::WaterLily.SetBody{typeof(-)}, x, t) = -fast_sdf(body.a, x, t)
@inline fast_sdf(body::WaterLily.SetBody, x, t) = body.op(fast_sdf(body.a, x, t), fast_sdf(body.b, x, t))
@inline fast_sdf(body::WaterLily.NoBody, x, t) = convert(eltype(x), Inf)
@inline fast_sdf(body::AbstractBody, x, t) = WaterLily.sdf(body, x, t; fastd²=zero(eltype(x)))

# WaterLily.measure! in two passes. The first evaluates the distance in every cell and updates
# only cells away from the body that were near it, or that changed side since the last measure
# (a body that moved further than the band in between); it lists the cells near the body. The
# second measures the listed cells (normal and body velocity), so the costly `measure` code
# does not set the register use of the pass over all cells.
# `near` holds 0x01 for a cell near the body, 0x00 / 0x02 for a cell away from it outside / inside.
# Both append the BDIM band faces (class 1: μ₀≠1, μ₁=0; class 2: μ₁≠0); `stat` holds
# [band, shell, V≠0, near, dirty lo(3), dirty hi(3), transition lo(3), transition hi(3)].
function k_measure!(lo, hi, μ₀, μ₁, V, σ, near_prev, body::B, t::T, d²::T, g::Grid{D},
                    stat, band, nearl, bdimlow::NTuple{D,Bool}) where {B,T,D}
    I, ok = cellidx(lo, hi)
    l = ok ? lin(g, I) : i32
    near = false
    d = zero(T)
    if ok
        d = fast_sdf(body, loc(0, I, T), t)
        @inbounds σ[l] = d
        near = d^2 < d²
    end
    wappend!(nearl, stat, 4, near, l)
    prev = ok ? @inbounds(near_prev[l]) : 0x00
    changed = ok & (near | (prev != (d < 0 ? 0x02 : 0x00)))
    warp_box!(stat, 4, changed, I)
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
    d = zero(T)
    if ok
        if B <: WaterLily.SetBody
            # Refine with WaterLily's gradient-normalized SetBody sdf on the small near list
            # so scaled SetBody components match WaterLily.measure! exactly.
            d = WaterLily.sdf(body, loc(0, I, T), t; fastd²=d²)
            @inbounds σ[l] = d
        else
            d = @inbounds σ[l]
        end
    end
    inband = ok & (d^2 < d²)
    vnz = false
    tr = false
    @unroll for i in i32:Int32(D)
        li = l + (i-i32)*g.nc
        cls = 0
        if inband
            dᵢ, nᵢ, Vᵢ = WaterLily.measure(body, loc(i, I, T), t; fastd²=d²)
            dᵢ = abs(dᵢ) ≤ T(0.5) ? dᵢ : copysign(dᵢ, d)
            @inbounds V[li] = Vᵢ[i]
            vnz |= Vᵢ[i] != 0
            m0 = WaterLily.μ₀(dᵢ, ϵ)
            @inbounds μ₀[li] = m0
            tr |= (m0 > 0) & (m0 < 1) & (I[i] > 2i32)
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
        elseif ok
            m0 = d < 0 ? zero(T) : one(T)
            @inbounds V[li] = zero(T)
            @inbounds μ₀[li] = m0
            @unroll for j in i32:Int32(D)
                @inbounds μ₁[li + (j-i32)*D*g.nc] = zero(T)
            end
            cls = m0 != 1 ? 1 : 0
            I[i] == 2i32 && (cls = bdimlow[i] ? max(cls, 1) : 0)
        end
        wappend!(band, stat, 1, cls == 1, li)
        wappend!(shell, stat, 2, cls == 2, li)
    end
    warp_box!(stat, 10, inband & tr, I)
    vnz && (@inbounds stat[3] = Int32(1))
    return nothing
end

# zero boundary values for vector fields (WaterLily.BC!(a, zeros, saveexit, perdir)).
# When `dlo`/`dhi` are given (a previous measure already zeroed the full boundaries), only
# directions whose boundary layer (`2` or `n-1`) intersects `dlo:dhi` are updated, over the
# dirty box expanded by 1 in the tangential directions (to include adjacent edges/corners).
function vecBC0!(a, g::Grid{D}, per, saveexit, dlo=nothing, dhi=nothing) where D
    z = ConstBC{D,eltype(a)}(ntuple(_->zero(eltype(a)), D))
    for j in 1:D
        if dlo !== nothing
            (dlo[1] > dhi[1]) && return nothing
            (dlo[j] > 2 && dhi[j] < Int(g.n[j]) - 1) && continue
            lo = ntuple(d -> d == j ? 1 : max(1, dlo[d] - 1), D)
            hi = ntuple(d -> d == j ? 1 : min(Int(g.n[d]), dhi[d] + 1), D)
        else
            lo = ntuple(_->1, D); hi = ntuple(d-> d==j ? 1 : Int(g.n[d]), D)
        end
        launch_face(k_bc!, lo, hi, j, a, g, z, zero(eltype(a)), saveexit, per[j])
    end
    return nothing
end

function WaterLily.measure!(a::TFlow{D,T}, body::AbstractBody; t=zero(T), ϵ=1) where {D,T}
    g = a.grid
    was_valid = a.dirty.valid
    while true
        @cuda k_mstat_init!(a.mstat)
        d² = T((2+ϵ)^2)
        launch(k_measure!, inside_lo(g), inside_hi(g), a.μ₀, a.μ₁, a.V, a.σ, a.near, body, T(t),
               d², g, a.mstat, a.band, a.nearl, a.bdimlow)
        # one thread per entry of the near list; threads past the count (on the device) return
        launch1(k_measure_near!, length(a.nearl), a.nearl, a.μ₀, a.μ₁, a.V, a.σ, body, T(t), T(ϵ),
                d², g, a.mstat, a.band, a.shell, a.bdimlow)
        st = hostcopy!(a.mstat_h, a.mstat)
        a.nband, a.nshell = st[1], st[2]
        a.Vzero = st[3] == 0
        if a.nband <= length(a.band) && a.nshell <= length(a.shell) && st[4] <= length(a.nearl)
            dlo = ntuple(d -> Int(st[4 + d]), D)
            dhi = ntuple(d -> Int(st[7 + d]), D)
            if was_valid
                vecBC0!(a.μ₀, g, a.per, false, dlo, dhi)
                vecBC0!(a.V, g, a.per, a.exitBC, dlo, dhi)
            else
                vecBC0!(a.μ₀, g, a.per, false)
                vecBC0!(a.V, g, a.per, a.exitBC)
            end
            if dlo[1] <= dhi[1]
                if a.dirty.valid && a.dirty.lo[1] <= a.dirty.hi[1]
                    a.dirty.lo = min.(a.dirty.lo, dlo)
                    a.dirty.hi = max.(a.dirty.hi, dhi)
                else
                    a.dirty.lo = dlo; a.dirty.hi = dhi
                end
            elseif !a.dirty.valid
                a.dirty.lo = ntuple(_ -> 2, D); a.dirty.hi = ntuple(_ -> 1, D)
            end
            a.dirty.valid = true
            a.dirty.tlo = st[11] > st[14] ? ntuple(_ -> 2, D) : ntuple(d -> Int(st[10 + d]), D)
            a.dirty.thi = st[11] > st[14] ? ntuple(_ -> 1, D) : ntuple(d -> Int(st[13 + d]), D)
            a.dirty.tvalid = true
            break
        end
        # lists too small: grow and measure again (every cell near the body is rewritten,
        # so a repeated measure gives the same result)
        a.band = CuVector{Int32}(undef, 2a.nband + 1024)
        a.shell = CuVector{Int32}(undef, 2a.nshell + 1024)
        a.nearl = CuVector{Int32}(undef, 2st[4] + 1024)
        a.near .= 0x01
        was_valid = false
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
# Same as `k_diag!`, but accumulates the change in solved-row count `Δ(iD≠0)` over a dirty box.
function k_diag_delta!(lo, hi, ps, pm, L, Dg, iD, reg, g::Grid{D}, Lreg) where D
    I, ok = cellidx(lo, hi)
    dc = 0f0
    if ok
        l = lin(g, I)
        old_nz = iszero(@inbounds(iD[l])) ? 0f0 : 1f0
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
        dc = (iszero(s) ? 0f0 : 1f0) - old_nz
    end
    block_partial!(ps, pm, dc, 0f0)
    return nothing
end
# Coarse-level diagonal update (no solved-row count needed).
function k_diag_coarse!(lo, hi, L, Dg, iD, reg, g::Grid{D}, Lreg) where D
    I, ok = cellidx(lo, hi); ok || return
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
    return nothing
end
function k_add_nsd!(nsd, sums, slot::Int32)
    @inbounds nsd[1] += sums[slot]
    return nothing
end
# coarse L from fine L (WaterLily.restrictL with the semi-coarsening mask c)
function k_restrictL!(lo, hi, a, ga::Grid{D}, b, gb::Grid{D}, c) where D
    I, ok = cellidx(lo, hi); ok || return
    la = lin(ga, I)
    @unroll for i in i32:Int32(D)
        # the fine faces under coarse face i: the first fine cell in every coarsened direction,
        # and both fine cells (e=1) in the coarsened directions tangential to the face
        J0 = ntuple(d-> c[d] ? Int32(2)*I[d]-Int32(2) : I[d], Val(D))
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
clip_box(lo, hi, a, b) = (map(max, lo, a), map(min, hi, b))
function update_box(lo, hi, g::Grid{D}, per, padlo::Int, padhi::Int) where D
    ilo, ihi = inside_lo(g), inside_hi(g)
    blo = ntuple(d -> (per[d] && (lo[d] - padlo < ilo[d] || hi[d] + padhi > ihi[d])) ? ilo[d] : max(ilo[d], lo[d] - padlo), D)
    bhi = ntuple(d -> (per[d] && (lo[d] - padlo < ilo[d] || hi[d] + padhi > ihi[d])) ? ihi[d] : min(ihi[d], hi[d] + padhi), D)
    return blo, bhi
end
function WaterLily.update!(p::TPoisson{T}) where T
    db = p.dirty
    if db !== nothing && db.valid
        dlo, dhi = db.lo, db.hi
        db.valid = false
        dlo[1] > dhi[1] && return p
        for l in eachindex(p.levels)
            lv = p.levels[l]; op = lv.op; g = op.g
            if l > 1
                f = p.levels[l-1]
                dlo = ntuple(d -> f.c[d] ? (dlo[d] + 2) ÷ 2 : dlo[d], length(dlo))
                dhi = ntuple(d -> f.c[d] ? (dhi[d] + 2) ÷ 2 : dhi[d], length(dhi))
                dlo, dhi = clip_box(dlo, dhi, inside_lo(g), inside_hi(g))
                launch(k_restrictL!, dlo, dhi, op.L, g, f.op.L, f.op.g, f.c)
                any(j -> p.per[j], 1:length(g.n)) && vecBC0!(op.L, g, p.per, false)
            end
            diag_lo, diag_hi = update_box(dlo, dhi, g, p.per, 1, 0)
            if l == 1
                checkcap(p.red, nblocks(diag_lo, diag_hi))
                nb = launch(k_diag_delta!, diag_lo, diag_hi, p.red.ps, p.red.pm, op.L, op.Dg, op.iD, op.reg, g, op.Lreg)
                if nb > 0
                    finish!(p.red, nb, p.red.s, p.red.m, SLOT_RSUM)
                    @cuda k_add_nsd!(p.nsd, p.red.s, Int32(SLOT_RSUM))
                end
            else
                launch(k_diag_coarse!, diag_lo, diag_hi, op.L, op.Dg, op.iD, op.reg, g, op.Lreg)
            end
            deep_lo, deep_hi = update_box(dlo, dhi, g, p.per, 2, 1)
            set_deep!(op, deep_lo, deep_hi)
        end
        return p
    end
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
        if l == 1
            checkcap(p.red, nblocks(inside_lo(g), inside_hi(g)))
            nb = launch(k_diag!, inside_lo(g), inside_hi(g), p.red.ps, p.red.pm, op.L, op.Dg, op.iD, op.reg, g, op.Lreg)
            finish!(p.red, nb, p.nsd, p.red.m, 1)
        else
            launch(k_diag_coarse!, inside_lo(g), inside_hi(g), op.L, op.Dg, op.iD, op.reg, g, op.Lreg)
        end
        set_deep!(op)
    end
    return p
end
