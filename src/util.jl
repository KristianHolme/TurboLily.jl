# Kernel-side grid geometry. All kernels index arrays linearly with Int32 offsets:
# the neighbour I±δⱼ of linear index `l` is `l±s[j]`, component `i` of a vector field
# lives at `l+(i-1)nc`.
const i32 = Int32(1)

struct Grid{D}
    n::NTuple{D,Int32}   # array size including ghosts
    s::NTuple{D,Int32}   # strides
    nc::Int32            # number of cells (= prod(n))
end
Grid(n::NTuple{D,<:Integer}) where D =
    Grid{D}(Int32.(n), ntuple(d->Int32(prod(n[1:d-1])), Val(D)), Int32(prod(n)))
Grid(a::AbstractArray) = Grid(size(a))

@inline lin(g::Grid{D}, I::NTuple{D,Int32}) where D = i32 + sum(ntuple(d->(I[d]-i32)*g.s[d], Val(D)))
@inline function coords(g::Grid{D}, l::Int32) where D
    l0 = l - i32
    ntuple(Val(D)) do d
        d == D ? l0 ÷ g.s[d] + i32 : (l0 ÷ g.s[d]) % g.n[d] + i32
    end
end
@inline shift(I::NTuple{D,Int32}, j, k) where D = ntuple(d-> d==j ? I[d]+Int32(k) : I[d], Val(D))
@inline setj(I::NTuple{D,Int32}, j, k) where D = ntuple(d-> d==j ? Int32(k) : I[d], Val(D))

# neighbour linear index with periodic wrap (ghost values in non-periodic
# directions are never needed by the Poisson operators because L=0 on the boundary faces)
@inline nbm(g::Grid, I, l, j, per) = (per[j] && I[j]==2i32) ? l + (g.n[j]-Int32(3))*g.s[j] : l - g.s[j]
@inline nbp(g::Grid, I, l, j, per) = (per[j] && I[j]==g.n[j]-i32) ? l - (g.n[j]-Int32(3))*g.s[j] : l + g.s[j]

# location of face i (i=0: centre) of cell I, matching WaterLily.loc
@inline loc(i, I::NTuple{D,Int32}, ::Type{T}) where {D,T} =
    SVector{D,T}(ntuple(d-> T(I[d]) - T(1.5) - (d==i ? T(0.5) : zero(T)), Val(D)))

"""
    @unroll for i in a:b ... end

Request full unrolling of a loop with compile-time constant bounds (LLVM loop metadata),
so tuple indexing by the loop variable becomes static.
"""
macro unroll(ex)
    (ex isa Expr && ex.head == :for) || error("@unroll expects a for loop")
    body = ex.args[2]
    ex.args[2] = Expr(:block, body, Expr(:loopinfo, (Symbol("llvm.loop.unroll.full"), 1)))
    return esc(ex)
end

@inline function tid3()
    x = (blockIdx().x-i32)*blockDim().x + threadIdx().x
    y = (blockIdx().y-i32)*blockDim().y + threadIdx().y
    z = (blockIdx().z-i32)*blockDim().z + threadIdx().z
    (x,y,z)
end
@inline function cellidx(lo::NTuple{D,Int32}, hi::NTuple{D,Int32}) where D
    t = tid3()
    I = ntuple(d-> lo[d] + t[d] - i32, Val(D))
    I, all(ntuple(d->I[d]<=hi[d], Val(D)))
end
@inline tid1() = (blockIdx().x-i32)*blockDim().x + threadIdx().x

threads_for(::Val{1}) = (256,1,1)
threads_for(::Val{2}) = (32,8,1)
threads_for(::Val{3}) = (32,4,2)

"""
    launch(k, lo, hi, args...)

Launch kernel `k(lo,hi,args...)` with one thread per cell of the box `lo:hi`.
"""
function launch(k::K, lo::NTuple{D,<:Integer}, hi::NTuple{D,<:Integer}, args...) where {K,D}
    ext = hi .- lo .+ 1
    any(ext .<= 0) && return 0
    th = threads_for(Val(D))
    ext3 = (ext..., ntuple(_->1, 3-D)...)
    bl = cld.(ext3, th)
    @cuda threads=th blocks=bl k(Int32.(lo), Int32.(hi), args...)
    return prod(bl)
end
"""
    launch_face(k, lo, hi, j, args...)

Launch over a box that is one cell thick in direction `j`; threads are laid out over
the other directions (the kernel uses `faceidx(lo,hi,j)`).
"""
function launch_face(k::K, lo::NTuple{D,<:Integer}, hi::NTuple{D,<:Integer}, j, args...) where {K,D}
    ext = hi .- lo .+ 1
    any(ext .<= 0) && return 0
    oth = ntuple(d-> d < j ? ext[d] : ext[d+1], D-1)
    th = D == 3 ? (32,8,1) : (256,1,1)
    ext3 = (oth..., ntuple(_->1, 3-(D-1))...)
    bl = cld.(ext3, th)
    @cuda threads=th blocks=bl k(Int32.(lo), Int32.(hi), Int32(j), args...)
    return prod(bl)
end
@inline function faceidx(lo::NTuple{D,Int32}, hi::NTuple{D,Int32}, j) where D
    t = tid3()
    I = ntuple(Val(D)) do d
        d == j ? lo[d] : d < j ? lo[d] + t[d] - i32 : lo[d] + t[d-1] - i32
    end
    return I, all(ntuple(d->I[d]<=hi[d], Val(D)))
end
function launch1(k::K, n::Integer, args...; threads=256) where K
    n <= 0 && return nothing
    @cuda threads=threads blocks=cld(n,threads) k(Int32(n), args...)
    nothing
end
inside_lo(::Grid{D}) where D = ntuple(_->2, D)
inside_hi(g::Grid{D}) where D = ntuple(d->Int(g.n[d])-1, D)

# ---------------------------------------------------------------- reductions
@inline function warp_red(s::Float32, m::Float32)
    off = Int32(16)
    while off > 0
        s += shfl_down_sync(0xffffffff, s, off)
        m = max(m, shfl_down_sync(0xffffffff, m, off))
        off >>= 1
    end
    return s, m
end
@inline function warp_red(s::Float64, m::Float32)
    off = Int32(16)
    while off > 0
        s += shfl_down_sync(0xffffffff, s, off)
        m = max(m, shfl_down_sync(0xffffffff, m, off))
        off >>= 1
    end
    return s, m
end
@inline flat_tid() = threadIdx().x - i32 + (threadIdx().y - i32)*blockDim().x +
                     (threadIdx().z - i32)*blockDim().x*blockDim().y
@inline flat_bid() = blockIdx().x + (blockIdx().y - i32)*gridDim().x +
                     (blockIdx().z - i32)*gridDim().x*gridDim().y
# (sum,max) over the block; every thread must call it. Returns the result on thread 0.
@inline function block_red(s::S, m::Float32) where S
    shs = CuStaticSharedArray(S, 32)
    shm = CuStaticSharedArray(Float32, 32)
    t = flat_tid()
    nt = blockDim().x*blockDim().y*blockDim().z
    s, m = warp_red(s, m)
    lane = t % Int32(32) + i32
    wid = t ÷ Int32(32) + i32
    nw = (nt + Int32(31)) ÷ Int32(32)
    lane == i32 && (@inbounds shs[wid] = s; @inbounds shm[wid] = m)
    sync_threads()
    if wid == i32
        s = lane <= nw ? @inbounds(shs[lane]) : zero(S)
        m = lane <= nw ? @inbounds(shm[lane]) : 0f0
        s, m = warp_red(s, m)
    end
    return s, m
end
"Write this block's (sum,max) partial to `ps/pm[block]`."
@inline function block_partial!(ps, pm, s::Float32, m::Float32)
    s, m = block_red(s, m)
    if flat_tid() == 0
        b = flat_bid()
        @inbounds ps[b] = s; @inbounds pm[b] = m
    end
    return nothing
end
function k_finish!(ps, pm, nb::Int32, outs, outm, k::Int32)
    s = 0.0; m = 0f0
    i = threadIdx().x
    while i <= nb
        @inbounds s += Float64(ps[i]); @inbounds m = max(m, pm[i])
        i += blockDim().x
    end
    s, m = block_red(s, m)
    if threadIdx().x == i32
        @inbounds outs[k] = s; @inbounds outm[k] = m
    end
    return nothing
end

"""
    RedBuf(nblocks)

Reduction scratch: per-block partials `ps/pm`, results `s` (Float64 sums) and `m`
(Float32 maxima) on the device, and host mirrors `hs/hm`.
"""
struct RedBuf
    s::CuVector{Float64}
    m::CuVector{Float32}
    ps::CuVector{Float32}
    pm::CuVector{Float32}
    hs::Vector{Float64}
    hm::Vector{Float32}
end
RedBuf(nb::Integer, n=8) = RedBuf(CUDA.zeros(Float64,n), CUDA.zeros(Float32,n),
    CUDA.zeros(Float32,nb), CUDA.zeros(Float32,nb), zeros(Float64,n), zeros(Float32,n))
RedBuf(g::Grid) = RedBuf(cld(Int(g.nc), 64) + 1)
function fetch!(b::RedBuf)
    copyto!(b.hs, b.s); copyto!(b.hm, b.m)
    return b
end
getsum(b::RedBuf, k) = b.hs[k]
getmax(b::RedBuf, k) = b.hm[k]

"""
    red_launch(k, lo, hi, b::RedBuf, slot, args...)

Launch `k(lo,hi,ps,pm,args...)` (which calls `block_partial!`) and reduce the partials
into `b.s[slot]`, `b.m[slot]`.
"""
function red_launch(k::K, lo, hi, b::RedBuf, slot, args...) where K
    nb = launch(k, lo, hi, b.ps, b.pm, args...)
    nb == 0 && return nothing
    @assert nb <= length(b.ps)
    @cuda threads=1024 k_finish!(b.ps, b.pm, Int32(nb), b.s, b.m, Int32(slot))
    return nothing
end
