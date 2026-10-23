using WaterLily: AbstractFlow, AbstractPoisson, AbstractBody, NoBody

"Constant boundary velocity as an isbits `(i,x,t)->U[i]` functor."
struct ConstBC{D,T}
    U::NTuple{D,T}
end
@inline (c::ConstBC)(i, x, t) = @inbounds c.U[i]

"Acceleration `g(i,x,t)+∂ₜU(i,x,t)` as isbits functors (WaterLily.accelerate! semantics)."
struct AccU{F}; U::F; end
@inline (a::AccU)(i, x, t) = WaterLily.derivative(τ->a.U(i,x,τ), t)
struct AccGU{G,F}; g::G; U::F; end
@inline (a::AccGU)(i, x, t) = a.g(i,x,t) + WaterLily.derivative(τ->a.U(i,x,τ), t)
make_acc(::Nothing, ::Tuple) = nothing
make_acc(g::Function, ::Tuple) = g
make_acc(::Nothing, U::Function) = AccU(U)
make_acc(g::Function, U::Function) = AccGU(g, U)

"""
    TFlow

GPU flow state with the same fields as `WaterLily.Flow`, so every WaterLily
function that reads `flow.u`, `flow.p`, `flow.μ₀`, … keeps working. The time step
is a fused CUDA implementation of `WaterLily.mom_step!`.

`u` and `u⁰` (and `f` as the corrector output) are swapped instead of copied;
always read `flow.u` afresh rather than holding on to the array.
"""
mutable struct TFlow{D,T,Sf<:AbstractArray{T},Vf<:AbstractArray{T},Tf<:AbstractArray{T},Lf,Bf,Af,Pf<:Per} <: AbstractFlow{D,T}
    u :: Vf
    u⁰:: Vf
    f :: Vf
    p :: Sf
    σ :: Sf
    V :: Vf
    μ₀:: Vf
    μ₁:: Tf
    uBC :: Union{NTuple{D,Number},Function}
    Δt:: Vector{T}
    ν :: T
    g :: Union{Function,Nothing}
    exitBC :: Bool
    perdir :: NTuple
    λ :: Lf
    # --- TurboLily state
    bc :: Bf                    # isbits boundary function
    acc :: Af                   # isbits acceleration or nothing
    grid :: Grid{D}
    per :: Pf                   # periodic directions
    bdimlow :: NTuple{D,Bool}   # lower boundary faces normal to j get the BDIM update (no BC! overwrites them)
    band :: CuVector{Int32}     # linear (I,i) indices of faces with μ₀≠1, μ₁=0 (u = V+μ₀f̃)
    shell :: CuVector{Int32}    # linear (I,i) indices of faces with μ₁≠0 (full BDIM update)
    nband :: Int
    nshell :: Int
    Vzero :: Bool               # true when V ≡ 0 (skip div(V) source)
    red :: RedBuf
    aux :: Union{Nothing,Vf}    # extra vector buffer for the udf path (allocated on first use)
    near :: CuArray{UInt8,D}    # per cell at the last measure!: 0x01 near the body, else 0x00 outside / 0x02 inside
    mstat :: CuVector{Int32}    # [nband, nshell, V≠0, near cells] from the last measure!
    nearl :: CuVector{Int32}    # linear indices of the cells near the body (measure! scratch)
end

"""
    TFlow(N, uBC; bdimfaces=(), kwargs...)

`bdimfaces`: lower domain faces (`-j`) whose normal velocity is not reset by `BC!` after the
momentum update (non-Biot-Savart faces of a Biot-Savart simulation), so they get the BDIM
update like any other face. Faces in periodic directions always do.
"""
function TFlow(N::NTuple{D}, uBC; T=Float32, mem=CuArray, bdimfaces=(), kw...) where D
    T === Float32 || throw(ArgumentError("TurboLily supports T=Float32 only (got T=$T)"))
    f = WaterLily.Flow(N, uBC; T, mem=CuArray, kw...)
    # `f` is swapped into `u` by the corrector: start it with the ghost values of `u`, as
    # cells that no BC writes (domain corners and edges) keep their initial values
    copyto!(f.f, f.u)
    bc = f.uBC isa Function ? f.uBC : ConstBC{D,T}(T.(f.uBC))
    acc = make_acc(f.g, f.uBC)
    per = Per(ntuple(j->j ∈ f.perdir, D))
    bdimlow = ntuple(j->per[j] || -j ∈ bdimfaces, D)
    a = TFlow{D,T,typeof(f.p),typeof(f.u),typeof(f.μ₁),typeof(f.λ),typeof(bc),typeof(acc),typeof(per)}(
        f.u, f.u⁰, f.f, f.p, f.σ, f.V, f.μ₀, f.μ₁, f.uBC, f.Δt, f.ν, f.g, f.exitBC, f.perdir, f.λ,
        bc, acc, Grid(f.p), per, bdimlow, CuVector{Int32}(undef,1024), CuVector{Int32}(undef,1024), 0, 0, true,
        RedBuf(Grid(f.p)), nothing, (n = similar(f.p, UInt8); fill!(n, 0x01); n), CUDA.zeros(Int32, 4),
        CuVector{Int32}(undef, 1024))
    return a
end

"""
    setproperties(flow::TFlow, patch::NamedTuple)

`WaterLily.setproperties(flow; uBC)` as for `WaterLily.Flow`: a new flow sharing all arrays,
with the kernel-side boundary function and acceleration rebuilt from the new values.
A tuple `uBC` is converted to the flow's eltype.
"""
function WaterLily.setproperties(a::TFlow{D,T}, patch::NamedTuple) where {D,T}
    names = fieldnames(typeof(a))
    all(n -> n in names, keys(patch)) || throw(ArgumentError("TFlow has no field(s) $(setdiff(keys(patch), names))"))
    nt = merge(NamedTuple{names}(map(n -> getfield(a, n), names)), patch)
    uBC = nt.uBC isa Tuple ? T.(nt.uBC) : nt.uBC
    bc = uBC isa Function ? uBC : ConstBC{D,T}(uBC)
    acc = make_acc(nt.g, uBC)
    nt = merge(nt, (; uBC, bc, acc))
    return TFlow{D,T,typeof(nt.p),typeof(nt.u),typeof(nt.μ₁),typeof(nt.λ),typeof(bc),typeof(acc),typeof(nt.per)}(values(nt)...)
end

# ---------------------------------------------------------------- convection-diffusion
# Flux Φⱼ of momentum component i through the lower j-face of cell I (WaterLily.conv_diff! semantics)
# Interior face (3 ≤ Iⱼ ≤ nⱼ-1): the upwind stencil is selected first so λ is evaluated once.
# Both far values are loaded before the selection, so no load waits on the sign of uf.
@inline function fluxi(j, i, l, u, g::AGrid, λ::F, ν) where F
    nc, sj = g.nc, g.s[j]
    ci = (i-i32)*nc
    @inbounds uf = (u[l + (j-i32)*nc] + u[l - g.s[i] + (j-i32)*nc]) * 0.5f0 # ϕ(i,CI(I,j),u)
    @inbounds c = u[l+ci]; @inbounds b = u[l-sj+ci]
    @inbounds fm = u[l+ci-2sj]; @inbounds fp = u[l+ci+sj]
    up = uf > 0
    return uf*λ(up ? fm : fp, up ? b : c, up ? c : b) - ν*(c-b)
end
# Any face, including the domain-boundary faces Iⱼ=2 and Iⱼ=nⱼ
@inline function flux(j, i, I::NTuple{D,Int32}, l, u, g::Grid{D}, λ::F, ν, per) where {D,F}
    Ij, nj = I[j], g.n[j]
    ((Ij != 2i32) & (Ij != nj)) && return fluxi(j, i, l, u, g, λ, ν)
    nc, sj = g.nc, g.s[j]
    ci = (i-i32)*nc
    @inbounds uf = (u[l + (j-i32)*nc] + u[l - g.s[i] + (j-i32)*nc]) * 0.5f0
    if per[j] && Ij == nj   # upper periodic face ≡ lower face at index 2
        l = l - (nj-Int32(2))*sj; Ij = Int32(2)
    end
    @inbounds c = u[l+ci]; @inbounds b = u[l-sj+ci]
    if Ij == 2i32
        if per[j]
            @inbounds up = u[l + (nj-Int32(4))*sj + ci]
            return (uf > 0 ? uf*λ(up, b, c) : uf*λ(@inbounds(u[l+sj+ci]), c, b)) - ν*(c-b)
        else
            return (uf > 0 ? uf*(c+b)*0.5f0 : uf*λ(@inbounds(u[l+sj+ci]), c, b)) - ν*(c-b)
        end
    else # Ij == nj
        return (uf < 0 ? uf*(c+b)*0.5f0 : uf*λ(@inbounds(u[l-2sj+ci]), b, c)) - ν*(c-b)
    end
end
# Any face with 2 ≤ Iⱼ ≤ nⱼ, as `flux` but with selects instead of branches, so that a kernel
# over the full j range keeps the code and register use of `fluxi`. Equal to `fluxi` on
# interior faces. Unselected values may come from outside the row; they are never used.
@inline function fluxb(j, i, Ij, l, u, g::AGrid, λ::F, ν, per) where F
    nc, sj, nj = g.nc, g.s[j], g.n[j]
    ci = (i-i32)*nc
    @inbounds uf = (u[l + (j-i32)*nc] + u[l - g.s[i] + (j-i32)*nc]) * 0.5f0
    wrap = per[j] & (Ij == nj)          # upper periodic face ≡ lower face at index 2
    lb = wrap ? l - (nj-Int32(2))*sj : l
    lo = Ij == Int32(2) || wrap
    @inbounds c = u[lb+ci]; @inbounds b = u[lb-sj+ci]
    @inbounds fm = u[lb + ci + ((per[j] & lo) ? (nj-Int32(4))*sj : -2sj)]
    @inbounds fp = u[lb+ci+sj]
    up = uf > 0
    t = uf*λ(up ? fm : fp, up ? b : c, up ? c : b)
    central = !per[j] & ((lo & up) | ((Ij == nj) & (uf < 0)))
    return (central ? uf*(c+b)*0.5f0 : t) - ν*(c-b)
end
@inline function iscore(I::NTuple{D,Int32}, g::Grid{D}) where D
    c = true
    @unroll for j in i32:Int32(D)
        c &= (I[j] >= Int32(3)) & (I[j] <= g.n[j] - Int32(2))
    end
    return c
end
# rᵢ(I) = Σⱼ Φⱼ(I) - Φⱼ(I+δⱼ), zero on the lower ghosts, partial on the upper ghosts
@inline function rconv(i, I::NTuple{D,Int32}, l, u, g::Grid{D}, λ::F, ν::T, per) where {D,F,T}
    s = zero(T)
    lowghost = false
    @unroll for j in i32:Int32(D)
        lowghost |= I[j] == i32
    end
    lowghost && return s
    if iscore(I, g)
        @unroll for j in i32:Int32(D)
            s += fluxi(j, i, l, u, g, λ, ν)
            s -= fluxi(j, i, l+g.s[j], u, g, λ, ν)
        end
        return s
    end
    @unroll for j in i32:Int32(D)
        if I[j] <= g.n[j]-i32
            s += flux(j, i, I, l, u, g, λ, ν, per)
            s -= flux(j, i, shift(I,j,1), l+g.s[j], u, g, λ, ν, per)
        end
    end
    s
end
@inline accel(::Nothing, i, I, t, ::Type{T}) where T = zero(T)
@inline accel(acc::A, i, I, t, ::Type{T}) where {A,T} = acc(i, loc(i,I,T), t)

# Interior update with μ₀=1, μ₁=0: out = u⁰+Δt·r(uin)   (predictor)
#                                   out = ½(uin + u⁰+Δt·r(uin)) (corrector)
function k_mom!(lo, hi, out, uin, u0, g::Grid{D}, dt::T, ν::T, λ::F, per, acc::A, t::T, corr::Bool) where {D,T,F,A}
    I, ok = cellidx(lo, hi); ok || return
    mom_cell!(I, out, uin, u0, g, dt, ν, λ, per, acc, t, corr)
    return nothing
end
# same on a one-cell-thick slab normal to j
function k_momf!(lo, hi, j, out, uin, u0, g::Grid{D}, dt::T, ν::T, λ::F, per, acc::A, t::T, corr::Bool) where {D,T,F,A}
    I, ok = faceidx(lo, hi, j); ok || return
    mom_cell!(I, out, uin, u0, g, dt, ν, λ, per, acc, t, corr)
    return nothing
end
@inline function mom_cell!(I, out, uin, u0, g::Grid{D}, dt::T, ν::T, λ::F, per, acc::A, t::T, corr::Bool) where {D,T,F,A}
    l = lin(g, I)
    @unroll for i in i32:Int32(D)
        li = l + (i-1)*g.nc
        r = rconv(i, I, l, uin, g, λ, ν, per) + accel(acc, i, I, t, T)
        @fastmath v = @inbounds(u0[li]) + dt*r
        @inbounds out[li] = corr ? T(0.5)*(uin[li] + v) : v
    end
    nothing
end
# core cells (all faces interior) + the one-cell shell next to the domain boundary
function core_and_shell(g::Grid{D}, fullx=false) where D
    lo, hi = inside_lo(g), inside_hi(g)
    clo, chi = lo .+ 1, hi .- 1
    if fullx
        clo = (lo[1], Base.tail(clo)...); chi = (hi[1], Base.tail(chi)...)
    end
    slabs = Tuple{NTuple{D,Int},NTuple{D,Int}}[]
    for j in (fullx ? 2 : 1):D   # slab j: Iⱼ ∈ {2, n-1}, dims < j restricted to the core, dims > j full
        l1 = ntuple(d-> d < j ? clo[d] : lo[d], D); h1 = ntuple(d-> d < j ? chi[d] : d == j ? lo[d] : hi[d], D)
        l2 = ntuple(d-> d < j ? clo[d] : d == j ? hi[d] : lo[d], D); h2 = ntuple(d-> d < j ? chi[d] : hi[d], D)
        push!(slabs, (l1,h1), (l2,h2))
    end
    return (clo, chi), slabs
end
# per-cell update with y/z faces interior and the full x range; the x faces use the
# branch-free `fluxb`, which handles the domain-boundary faces x=2 (lower) and x=n (upper)
function k_momc!(lo, hi, out, uin, u0, g::AGrid{D}, dt::T, ν::T, λ::F, per, acc::A, t::T, corr::Bool) where {D,T,F,A}
    I, ok = cellidx(lo, hi); ok || return
    l = lin(g, I)
    @unroll for i in i32:Int32(D)
        r = zero(T)
        r += fluxb(1, i, I[1], l, uin, g, λ, ν, per)
        r -= fluxb(1, i, I[1]+i32, l+i32, uin, g, λ, ν, per)
        @unroll for j in Int32(2):Int32(D)
            r += fluxi(j, i, l, uin, g, λ, ν)
            r -= fluxi(j, i, l+g.s[j], uin, g, λ, ν)
        end
        r += accel(acc, i, I, t, T)
        li = l + (i-i32)*g.nc
        @fastmath v = @inbounds(u0[li]) + dt*r
        @inbounds out[li] = corr ? T(0.5)*(uin[li] + v) : v
    end
    return nothing
end
launch_core!(a::TFlow{D,T}, lo, hi, out, uin, t, corr) where {D,T} =
    launch(k_momc!, lo, hi, out, uin, a.u⁰, SGrid(a.grid), a.Δt[end], a.ν, a.λ, a.per, a.acc, T(t), corr)
# all boundary slabs in one launch: thread t takes cell t-start[b] of slab b (x fastest)
function k_momslabs!(nt, lo::NTuple{K}, ext::NTuple{K}, start::NTuple{K,Int32}, out, uin, u0, g::Grid{D},
                     dt::T, ν::T, λ::F, per, acc::A, t::T, corr::Bool) where {K,D,T,F,A}
    k = tid1() - i32; k < nt || return
    I = ntuple(_->i32, Val(D))
    @unroll for b in 1:K
        (k >= start[b]) && (I = unravel(k - start[b], lo[b], ext[b]))
    end
    mom_cell!(I, out, uin, u0, g, dt, ν, λ, per, acc, t, corr)
    return nothing
end
function launch_mom!(a::TFlow{D,T}, out, uin, t, corr) where {D,T}
    fullx = true
    (clo, chi), slabs = core_and_shell(a.grid, fullx)
    launch_core!(a, clo, chi, out, uin, t, corr)
    lo = Tuple(Int32.(l) for (l, h) in slabs); ext = Tuple(Int32.(h .- l .+ 1) for (l, h) in slabs)
    len = map(e-> Int(prod(e)), ext)
    start = Tuple(Int32.(cumsum((0, len...))[1:end-1]))
    launch1(k_momslabs!, sum(len), lo, ext, start, out, uin, a.u⁰, a.grid, a.Δt[end], a.ν, a.λ, a.per, a.acc, T(t), corr)
end

# Full BDIM update on the listed faces; `shell=false` lists have μ₁=0.
@inline function ftilde(i, I, l, uin, u0, V, g::Grid, dt::T, ν, λ, per, acc, t) where T
    li = l + (i-i32)*g.nc
    r = rconv(i, I, l, uin, g, λ, ν, per) + accel(acc, i, I, t, T)
    @fastmath @inbounds u0[li] + dt*r - V[li]
end
function k_band!(n, band, out, uin, u0, V, μ₀, μ₁, g::Grid{D}, dt::T, ν::T, λ::F, per, acc::A, t::T, corr::Bool, shell::Bool) where {D,T,F,A}
    k = tid1(); k <= n || return
    li = @inbounds band[k]
    iv = (li - i32) ÷ g.nc + i32
    # compile-time component index so that tuple indexing by i stays in registers
    args = (li, out, uin, u0, V, μ₀, μ₁, g, dt, ν, λ, per, acc, t, corr, shell)
    if iv == 1
        band_face!(Val(Int32(1)), args...)
    elseif iv == 2
        band_face!(Val(Int32(2)), args...)
    elseif D == 3
        band_face!(Val(Int32(3)), args...)
    end
    return nothing
end
@inline function band_face!(::Val{i}, li, out, uin, u0, V, μ₀, μ₁, g::Grid{D}, dt::T, ν, λ, per, acc, t, corr, shell) where {i,D,T}
    l = li - (i-i32)*g.nc
    μ0 = @inbounds μ₀[li]
    if !shell && μ0 == 0        # u = V exactly (μ₀ = μ₁ = 0)
        v = @inbounds V[li]
    else
        I = coords(g, l)
        f0 = ftilde(i, I, l, uin, u0, V, g, dt, ν, λ, per, acc, t)
        m = zero(T)
        if shell
            @unroll for j in i32:Int32(D)
                μ = @inbounds μ₁[li + (j-i32)*D*g.nc]
                if μ != 0
                    fp = ftilde(i, shift(I,j,1), l+g.s[j], uin, u0, V, g, dt, ν, λ, per, acc, t)
                    fm = ftilde(i, shift(I,j,-1), l-g.s[j], uin, u0, V, g, dt, ν, λ, per, acc, t)
                    m += μ*(fp-fm)
                end
            end
        end
        @fastmath v = m*T(0.5) + @inbounds(V[li]) + μ0*f0
    end
    @inbounds out[li] = corr ? T(0.5)*(uin[li] + v) : v
    return nothing
end
# Shell faces (μ₁≠0) with a group of 8 lanes per face: lane q evaluates f̃ at the face (q=0)
# or at its ±δⱼ neighbour (q=2j-1, 2j) when μ₁ⱼ≠0, all lanes running the same code; the
# group's first lane combines them by shuffles in the order of band_face!.
function k_shell!(nt, n, list, out, uin, u0, V, μ₀, μ₁, g::Grid{D}, dt::T, ν::T, λ::F, per, acc::A, t::T, corr::Bool) where {D,T,F,A}
    t0 = tid1() - i32
    k = t0 ÷ Int32(8) + i32; q = t0 % Int32(8)
    valid = t0 < nt
    li = @inbounds list[valid ? k : i32]
    iv = (li - i32) ÷ g.nc + i32
    args = (q, valid, li, out, uin, u0, V, μ₀, μ₁, g, dt, ν, λ, per, acc, t, corr)
    if iv == 1
        shell_face!(Val(Int32(1)), args...)
    elseif iv == 2
        shell_face!(Val(Int32(2)), args...)
    elseif D == 3
        shell_face!(Val(Int32(3)), args...)
    end
    return nothing
end
@inline function shell_face!(::Val{i}, q, valid, li, out, uin, u0, V, μ₀, μ₁, g::Grid{D}, dt::T, ν, λ, per, acc, t, corr) where {i,D,T}
    l = li - (i-i32)*g.nc
    I = coords(g, l)
    j = (q + i32) >> 1                                    # neighbour direction (0: the face)
    sg = q == 0 ? Int32(0) : isodd(q) ? i32 : -i32
    sj = j == i32 ? g.s[1] : j == Int32(2) ? g.s[2] : g.s[D]
    μ = (valid & (q > 0) & (j <= Int32(D))) ? @inbounds(μ₁[li + (j-i32)*D*g.nc]) : zero(T)
    Ip = ntuple(d-> I[d] + (d == j ? sg : Int32(0)), Val(D))
    lp = l + sg*sj
    need = valid & ((q == 0) | (μ != 0))
    f = need ? ftilde(i, Ip, lp, uin, u0, V, g, dt, ν, λ, per, acc, t) : zero(T)
    lane0 = ((threadIdx().x - i32) & ~Int32(7)) % Int32(32) + i32
    f0 = shfl_sync(0xffffffff, f, lane0)
    m = zero(T)
    @unroll for jj in i32:Int32(D)
        fp = shfl_sync(0xffffffff, f, lane0 + Int32(2)*jj - i32)
        fm = shfl_sync(0xffffffff, f, lane0 + Int32(2)*jj)
        if valid & (q == 0)
            μj = @inbounds μ₁[li + (jj-i32)*D*g.nc]
            μj != 0 && (m += μj*(fp-fm))
        end
    end
    if valid & (q == 0)
        μ0 = @inbounds μ₀[li]
        @fastmath v = m*T(0.5) + @inbounds(V[li]) + μ0*f0
        @inbounds out[li] = corr ? T(0.5)*(uin[li] + v) : v
    end
    return nothing
end
function mom_update!(a::TFlow{D,T}, out, uin, t, corr) where {D,T}
    g, dt = a.grid, a.Δt[end]
    launch_mom!(a, out, uin, t, corr)
    launch1(k_band!, a.nband, a.band, out, uin, a.u⁰, a.V, a.μ₀, a.μ₁, g, dt, a.ν, a.λ, a.per, a.acc, T(t), corr, false)
    launch1(k_shell!, 8a.nshell, a.nshell, a.shell, out, uin, a.u⁰, a.V, a.μ₀, a.μ₁, g, dt, a.ν, a.λ, a.per, a.acc, T(t), corr)
end

# ---------------------------------------------------------------- boundary conditions
# One launch per direction j (order matters on edges), all components per thread.
function k_bc!(lo, hi, j, a, g::Grid{D}, bc::B, t::T, saveexit::Bool, perj::Bool) where {D,B,T}
    I, ok = faceidx(lo, hi, j); ok || return
    nj = g.n[j]
    l1 = lin(g, I)                 # I[j]==1
    sj = g.s[j]
    ln = l1 + (nj-i32)*sj
    @unroll for i in i32:Int32(D)
        ci = (i-1)*g.nc
        if perj
            @inbounds a[l1+ci] = a[l1+(nj-Int32(2))*sj+ci]
            @inbounds a[ln+ci] = a[l1+sj+ci]
        elseif i == j
            @inbounds a[l1+ci] = bc(i, loc(i,I,T), t)
            @inbounds a[l1+sj+ci] = bc(i, loc(i,setj(I,j,2),T), t)
            (!saveexit || i > 1) && (@inbounds a[ln+ci] = bc(i, loc(i,setj(I,j,nj),T), t))
        else
            @inbounds a[l1+ci] = bc(i, loc(i,I,T), t) + a[l1+sj+ci] - bc(i, loc(i,setj(I,j,2),T), t)
            @inbounds a[ln+ci] = bc(i, loc(i,setj(I,j,nj),T), t) + a[ln-sj+ci] - bc(i, loc(i,setj(I,j,nj-1),T), t)
        end
    end
    nothing
end
function BC!(a::TFlow{D,T}, u, t; jfirst=1) where {D,T}
    g = a.grid
    for j in jfirst:D
        lo = ntuple(_->1, D); hi = ntuple(d-> d==j ? 1 : Int(g.n[d]), D)
        launch_face(k_bc!, lo, hi, j, u, g, a.bc, T(t), a.exitBC, a.per[j])
    end
end

# The exit-plane ghost of u₁ is not written by BC! when exitBC=true, so it must follow
# the array swaps: dst[n₁,:,:,1] = src[n₁,:,:,1]
function k_copyexit!(lo, hi, j, dst, src, g::Grid)
    I, ok = faceidx(lo, hi, j); ok || return
    l = lin(g, I)
    @inbounds dst[l] = src[l]
    return nothing
end
function copy_exit!(a::TFlow{D}, dst, src) where D
    a.exitBC || return
    g = a.grid
    lo = ntuple(d-> d==1 ? Int(g.n[1]) : 1, D); hi = ntuple(d->Int(g.n[d]), D)
    launch_face(k_copyexit!, lo, hi, 1, dst, src, g)
end

# Convective exit (WaterLily.exitBC!), in three launches over the plane (block partial sums,
# reduced on the device), so that all SMs share the strided plane accesses; sums in a.red
# slots 1 (inlet) and 2 (exit).
# (the inlet u₁ is the value BC! sets there, so it is evaluated rather than read)
function k_exitsum!(lo, hi, j, ps, pm, bc::B, t::T) where {B,T}
    I, ok = faceidx(lo, hi, j)
    v = ok ? bc(i32, loc(i32, I, T), t) : 0f0
    block_partial!(ps, pm, v, 0f0)
    return nothing
end
function k_exitupd!(lo, hi, j, ps, pm, u, u0, g::Grid, dt::T, sums, np) where T
    I, ok = faceidx(lo, hi, j)
    U = Float32(@inbounds(sums[1]))/np
    v = 0f0
    if ok
        l = lin(g, I)
        v = @inbounds u0[l] - U*dt*(u0[l]-u0[l-i32])
        @inbounds u[l] = v
    end
    block_partial!(ps, pm, v, 0f0)
    return nothing
end
function k_exitfix!(lo, hi, j, u, g::Grid, sums, np)
    I, ok = faceidx(lo, hi, j); ok || return
    U = Float32(@inbounds(sums[1]))/np
    fix = Float32(@inbounds(sums[2]))/np - U
    @inbounds u[lin(g, I)] -= fix
    return nothing
end
function exitBC!(a::TFlow{D,T}, u, u0, dt, t) where {D,T}
    g = a.grid; b = a.red
    plane(x) = (ntuple(d-> d == 1 ? x : 2, D), ntuple(d-> d == 1 ? x : Int(g.n[d])-1, D))
    np = Float32(prod(Int.(g.n[2:D]) .- 2))
    nb = launch_face(k_exitsum!, plane(2)..., 1, b.ps, b.pm, a.bc, T(t))
    finish!(b, nb, b.s, b.m, 1)
    lo, hi = plane(Int(g.n[1]))
    nb = launch_face(k_exitupd!, lo, hi, 1, b.ps, b.pm, u, u0, g, T(dt), b.s, np)
    finish!(b, nb, b.s, b.m, 2)
    launch_face(k_exitfix!, lo, hi, 1, u, g, b.s, np)
    return nothing
end

# ---------------------------------------------------------------- CFL
function k_cfl!(lo, hi, ps, pm, u, g::Grid{D}) where D
    I, ok = cellidx(lo, hi)
    m = 0f0
    if ok
        l = lin(g, I)
        @unroll for i in i32:Int32(D)
            ci = (i-i32)*g.nc
            @inbounds m += max(0f0, u[l+g.s[i]+ci]) + max(0f0, -u[l+ci])
        end
    end
    block_partial!(ps, pm, 0f0, m)
    return nothing
end
function WaterLily.CFL(a::TFlow; Δt_max=10)
    g = a.grid; b = a.red
    red_launch(k_cfl!, inside_lo(g), inside_hi(g), b, 1, a.u, g)
    fetch!(b)
    min(Δt_max, inv(getmax(b,1) + 5a.ν))
end

# ---------------------------------------------------------------- time step
using WaterLily: @log
function WaterLily.mom_step!(a::TFlow{D,T}, b::AbstractPoisson; udf=nothing, kwargs...) where {D,T}
    isnothing(udf) || return mom_step_udf!(a, b; udf, kwargs...)
    a.u, a.u⁰ = a.u⁰, a.u                     # u⁰ ← u (no copy)
    copy_exit!(a, a.u, a.u⁰)
    t₁ = sum(a.Δt); t₀ = t₁ - a.Δt[end]
    # predictor
    @log "p"
    mom_update!(a, a.u, a.u⁰, t₀, false)
    WaterLily.mom_project!(a, b, 1, t₁)
    # corrector
    @log "c"
    mom_update!(a, a.f, a.u, t₁, true)
    copy_exit!(a, a.f, a.u)
    a.u, a.f = a.f, a.u
    if b isa TPoisson
        WaterLily.mom_project!(a, b, 2, t₁; cfl=true)
        push!(a.Δt, cfl!(a, b))
    else
        WaterLily.mom_project!(a, b, 2, t₁)
        push!(a.Δt, WaterLily.CFL(a))
    end
end

# ---------------------------------------------------------------- udf path
# The user function modifies `flow.f` between the convective RHS and the BDIM update
# (WaterLily.mom_predict!/mom_correct! semantics), so r is materialised in `f`.
function k_rhs!(lo, hi, f, u, g::Grid{D}, λ::F, ν::T, per) where {D,T,F}
    I, ok = cellidx(lo, hi); ok || return
    l = lin(g, I)
    @unroll for i in i32:Int32(D)
        @inbounds f[l + (i-i32)*g.nc] = rconv(i, I, l, u, g, λ, ν, per)
    end
    return nothing
end
# f̃ = u⁰ + Δt(f + acc) - V from the stored RHS
@inline ftilde_f(i, I, l, f, u0, V, g::Grid, dt::T, acc, t) where T =
    @fastmath @inbounds u0[l+(i-i32)*g.nc] + dt*(f[l+(i-i32)*g.nc] + accel(acc, i, I, t, T)) - V[l+(i-i32)*g.nc]
function k_bdim_plain!(lo, hi, out, uin, f, u0, V, g::Grid{D}, dt::T, acc::A, t::T, corr::Bool) where {D,T,A}
    I, ok = cellidx(lo, hi); ok || return
    l = lin(g, I)
    @unroll for i in i32:Int32(D)
        li = l + (i-i32)*g.nc
        v = ftilde_f(i, I, l, f, u0, V, g, dt, acc, t) + @inbounds(V[li])
        @inbounds out[li] = corr ? T(0.5)*(uin[li] + v) : v
    end
    return nothing
end
function k_bdim_list!(n, list, out, uin, f, u0, V, μ₀, μ₁, g::Grid{D}, dt::T, acc::A, t::T, corr::Bool) where {D,T,A}
    k = tid1(); k <= n || return
    li = @inbounds list[k]
    iv = (li - i32) ÷ g.nc + i32
    args = (li, out, uin, f, u0, V, μ₀, μ₁, g, dt, acc, t, corr)
    if iv == 1
        bdim_face!(Val(Int32(1)), args...)
    elseif iv == 2
        bdim_face!(Val(Int32(2)), args...)
    elseif D == 3
        bdim_face!(Val(Int32(3)), args...)
    end
    return nothing
end
@inline function bdim_face!(::Val{i}, li, out, uin, f, u0, V, μ₀, μ₁, g::Grid{D}, dt::T, acc, t, corr) where {i,D,T}
    l = li - (i-i32)*g.nc
    I = coords(g, l)
    f0 = ftilde_f(i, I, l, f, u0, V, g, dt, acc, t)
    m = zero(T)
    @unroll for j in i32:Int32(D)
        μ = @inbounds μ₁[li + (j-i32)*D*g.nc]
        if μ != 0
            m += μ*(ftilde_f(i, shift(I,j,1), l+g.s[j], f, u0, V, g, dt, acc, t) -
                    ftilde_f(i, shift(I,j,-1), l-g.s[j], f, u0, V, g, dt, acc, t))
        end
    end
    @fastmath v = m*T(0.5) + @inbounds(V[li]) + @inbounds(μ₀[li])*f0
    @inbounds out[li] = corr ? T(0.5)*(uin[li] + v) : v
    return nothing
end
function udf_update!(a::TFlow{D,T}, out, uin, t, corr, udf; kwargs...) where {D,T}
    g, dt = a.grid, a.Δt[end]
    launch(k_rhs!, ntuple(_->1,D), Int.(g.n), a.f, uin, g, a.λ, a.ν, a.per)
    WaterLily.udf!(a, udf, uin, t; kwargs...)
    launch(k_bdim_plain!, inside_lo(g), inside_hi(g), out, uin, a.f, a.u⁰, a.V, g, dt, a.acc, T(t), corr)
    for (n, list) in ((a.nband, a.band), (a.nshell, a.shell))
        launch1(k_bdim_list!, n, list, out, uin, a.f, a.u⁰, a.V, a.μ₀, a.μ₁, g, dt, a.acc, T(t), corr)
    end
end
function k_zero_u!(lo, hi, u, g::Grid{D}) where D
    I, ok = cellidx(lo, hi); ok || return
    l = lin(g, I)
    @unroll for i in i32:Int32(D)
        @inbounds u[l + (i-i32)*g.nc] = zero(eltype(u))
    end
    return nothing
end
function mom_step_udf!(a::TFlow{D,T}, b::AbstractPoisson; udf, kwargs...) where {D,T}
    a.aux === nothing && (a.aux = similar(a.u))
    a.u, a.u⁰ = a.u⁰, a.u
    copy_exit!(a, a.u, a.u⁰)
    t₁ = sum(a.Δt); t₀ = t₁ - a.Δt[end]
    # the predictor udf sees flow.u with a zero interior, as in WaterLily.mom_step!
    launch(k_zero_u!, inside_lo(a.grid), inside_hi(a.grid), a.u, a.grid)
    @log "p"
    udf_update!(a, a.u, a.u⁰, t₀, false, udf; kwargs...)
    WaterLily.mom_project!(a, b, 1, t₁)
    @log "c"
    udf_update!(a, a.aux, a.u, t₁, true, udf; kwargs...)
    copy_exit!(a, a.aux, a.u)
    a.u, a.aux = a.aux, a.u
    WaterLily.mom_project!(a, b, 2, t₁)
    push!(a.Δt, WaterLily.CFL(a))
end
