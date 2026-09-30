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
mutable struct TFlow{D,T,Sf<:AbstractArray{T},Vf<:AbstractArray{T},Tf<:AbstractArray{T},Lf,Bf,Af} <: AbstractFlow{D,T}
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
    per :: NTuple{D,Bool}
    band :: CuVector{Int32}     # linear (I,i) indices of faces needing the full BDIM update
    Vzero :: Bool               # true when V ≡ 0 (skip div(V) source)
    red :: RedBuf
end

function TFlow(N::NTuple{D}, uBC; T=Float32, mem=CuArray, kw...) where D
    f = WaterLily.Flow(N, uBC; T, mem=CuArray, kw...)
    bc = f.uBC isa Function ? f.uBC : ConstBC{D,T}(T.(f.uBC))
    acc = make_acc(f.g, f.uBC)
    per = ntuple(j->j ∈ f.perdir, D)
    a = TFlow{D,T,typeof(f.p),typeof(f.u),typeof(f.μ₁),typeof(f.λ),typeof(bc),typeof(acc)}(
        f.u, f.u⁰, f.f, f.p, f.σ, f.V, f.μ₀, f.μ₁, f.uBC, f.Δt, f.ν, f.g, f.exitBC, f.perdir, f.λ,
        bc, acc, Grid(f.p), per, CuVector{Int32}(undef,0), true, RedBuf(Grid(f.p)))
    update_band!(a)
    a
end

# ---------------------------------------------------------------- measure
function WaterLily.measure!(a::TFlow, body::AbstractBody; t=zero(eltype(a.p)), ϵ=1)
    invoke(WaterLily.measure!, Tuple{AbstractFlow,AbstractBody}, a, body; t, ϵ)
    update_band!(a)
end
WaterLily.measure!(a::TFlow, ::NoBody; kwargs...) = nothing

function k_bandmask!(lo, hi, m, μ₀, μ₁, g::Grid{D}) where D
    I, ok = cellidx(lo, hi); ok || return
    l = lin(g, I)
    @unroll for i in i32:Int32(D)
        li = l + (i-1)*g.nc
        b = @inbounds μ₀[li] != 1
        @unroll for j in i32:Int32(D)
            b |= @inbounds μ₁[li + (j-1)*D*g.nc] != 0
        end
        @inbounds m[li] = b
    end
    nothing
end
function update_band!(a::TFlow{D}) where D
    g = a.grid
    m = CUDA.zeros(Bool, size(a.u))
    launch(k_bandmask!, inside_lo(g), inside_hi(g), m, a.μ₀, a.μ₁, g)
    a.band = CuVector{Int32}(findall(vec(m)))
    a.Vzero = !any(!iszero, a.V)
    CUDA.unsafe_free!(m)
    nothing
end

# ---------------------------------------------------------------- convection-diffusion
# Flux Φⱼ of momentum component i through the lower j-face of cell I (WaterLily.conv_diff! semantics)
@inline function flux(j, i, I::NTuple{D,Int32}, l, u, g::Grid{D}, λ::F, ν, per) where {D,F}
    nc, sj = g.nc, g.s[j]
    ci = (i-1)*nc
    @inbounds uf = (u[l + (j-1)*nc] + u[l - g.s[i] + (j-1)*nc]) * 0.5f0 # ϕ(i,CI(I,j),u)
    Ij, nj = I[j], g.n[j]
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
    elseif Ij == nj
        return (uf < 0 ? uf*(c+b)*0.5f0 : uf*λ(@inbounds(u[l-2sj+ci]), b, c)) - ν*(c-b)
    else
        return (uf > 0 ? uf*λ(@inbounds(u[l-2sj+ci]), b, c) : uf*λ(@inbounds(u[l+sj+ci]), c, b)) - ν*(c-b)
    end
end
# rᵢ(I) = Σⱼ Φⱼ(I) - Φⱼ(I+δⱼ), zero on the lower ghosts, partial on the upper ghosts
@inline function rconv(i, I::NTuple{D,Int32}, l, u, g::Grid{D}, λ::F, ν::T, per) where {D,F,T}
    s = zero(T)
    lowghost = false
    @unroll for j in i32:Int32(D)
        lowghost |= I[j] == i32
    end
    lowghost && return s
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
    l = lin(g, I)
    @unroll for i in i32:Int32(D)
        li = l + (i-1)*g.nc
        r = rconv(i, I, l, uin, g, λ, ν, per) + accel(acc, i, I, t, T)
        @fastmath v = @inbounds(u0[li]) + dt*r
        @inbounds out[li] = corr ? T(0.5)*(uin[li] + v) : v
    end
    nothing
end
# Full BDIM update on the band faces
@inline function ftilde(i, I, l, uin, u0, V, g::Grid, dt::T, ν, λ, per, acc, t) where T
    li = l + (i-1)*g.nc
    r = rconv(i, I, l, uin, g, λ, ν, per) + accel(acc, i, I, t, T)
    @fastmath @inbounds u0[li] + dt*r - V[li]
end
function k_band!(n, band, out, uin, u0, V, μ₀, μ₁, g::Grid{D}, dt::T, ν::T, λ::F, per, acc::A, t::T, corr::Bool) where {D,T,F,A}
    k = tid1(); k <= n || return
    li = @inbounds band[k]
    i = (li - i32) ÷ g.nc + i32
    l = li - (i-i32)*g.nc
    I = coords(g, l)
    f0 = ftilde(i, I, l, uin, u0, V, g, dt, ν, λ, per, acc, t)
    m = zero(T)
    @unroll for j in i32:Int32(D)
        μ = @inbounds μ₁[li + (j-1)*D*g.nc]
        if μ != 0
            fp = ftilde(i, shift(I,j,1), l+g.s[j], uin, u0, V, g, dt, ν, λ, per, acc, t)
            fm = ftilde(i, shift(I,j,-1), l-g.s[j], uin, u0, V, g, dt, ν, λ, per, acc, t)
            m += μ*(fp-fm)
        end
    end
    @fastmath v = m*T(0.5) + @inbounds(V[li]) + @inbounds(μ₀[li])*f0
    @inbounds out[li] = corr ? T(0.5)*(uin[li] + v) : v
    nothing
end
function mom_update!(a::TFlow{D,T}, out, uin, t, corr) where {D,T}
    g, dt = a.grid, a.Δt[end]
    launch(k_mom!, inside_lo(g), inside_hi(g), out, uin, a.u⁰, g, dt, a.ν, a.λ, a.per, a.acc, T(t), corr)
    launch1(k_band!, length(a.band), a.band, out, uin, a.u⁰, a.V, a.μ₀, a.μ₁, g, dt, a.ν, a.λ, a.per, a.acc, T(t), corr)
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
function BC!(a::TFlow{D,T}, u, t) where {D,T}
    g = a.grid
    for j in 1:D
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

# Convective exit (single block, three phases): WaterLily.exitBC!
function k_exit!(u, u0, g::Grid{D}, dt::T) where {D,T}
    # plane of cells with I[1] fixed, others 2:n-1
    np = Int32(1)
    @unroll for d in i32:Int32(D)
        d > 1 && (np *= g.n[d]-Int32(2))
    end
    sh = CuStaticSharedArray(Float32, 32)
    function planeI(k)   # k ∈ 0:np-1 → offset with I[1]=1
        o = Int32(0); r = k
        @unroll for d in i32:Int32(D)
            if d > 1
                m = g.n[d]-Int32(2)
                o += (r % m + i32)*g.s[d]; r ÷= m
            end
        end
        o + i32
    end
    function bsum(v)
        v, _ = warp_red(v, 0f0)
        lane = (threadIdx().x - i32) % Int32(32) + i32
        wid = (threadIdx().x - i32) ÷ Int32(32) + i32
        lane == i32 && (@inbounds sh[wid] = v)
        sync_threads()
        v = lane <= blockDim().x ÷ Int32(32) ? @inbounds(sh[lane]) : 0f0
        v, _ = warp_red(v, 0f0)
        v = shfl_sync(0xffffffff, v, 1)
        sync_threads()
        v
    end
    n1 = g.n[1]
    s = 0f0; k = threadIdx().x - i32
    while k < np
        s += @inbounds u[planeI(k) + i32]          # I[1]=2
        k += blockDim().x
    end
    sh2 = CuStaticSharedArray(Float32, 1)
    s = bsum(s)
    threadIdx().x == i32 && (@inbounds sh2[1] = s)
    sync_threads()
    U = @inbounds(sh2[1]) / np
    sync_threads()
    s = 0f0; k = threadIdx().x - i32
    while k < np
        l = planeI(k) + n1 - i32
        v = @inbounds u0[l] - U*dt*(u0[l]-u0[l-i32])
        @inbounds u[l] = v
        s += v
        k += blockDim().x
    end
    s = bsum(s)
    threadIdx().x == i32 && (@inbounds sh2[1] = s)
    sync_threads()
    fix = @inbounds(sh2[1])/np - U
    k = threadIdx().x - i32
    while k < np
        l = planeI(k) + n1 - i32
        @inbounds u[l] -= fix
        k += blockDim().x
    end
    nothing
end
exitBC!(a::TFlow{D,T}, u, u0, dt) where {D,T} = (@cuda threads=1024 k_exit!(u, u0, a.grid, T(dt)); nothing)

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
    BC!(a, a.u, t₁)
    a.exitBC && exitBC!(a, a.u, a.u⁰, a.Δt[end])
    WaterLily.mom_project!(a, b, 1, t₁)
    # corrector
    @log "c"
    mom_update!(a, a.f, a.u, t₁, true)
    copy_exit!(a, a.f, a.u)
    a.u, a.f = a.f, a.u
    BC!(a, a.u, t₁)
    WaterLily.mom_project!(a, b, 0.5, t₁)
    push!(a.Δt, WaterLily.CFL(a))
end

# udf path: materialize r in `f` so the user function can modify it (WaterLily semantics)
function mom_step_udf!(a::TFlow{D,T}, b::AbstractPoisson; udf, kwargs...) where {D,T}
    error("TurboLily: udf is not supported yet")
end
