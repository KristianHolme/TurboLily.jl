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

"""
    Per(t::NTuple{D,Bool})

Periodic flags as a type parameter: `per[j]` is a compile-time constant in kernels, so the
wrap code of non-periodic directions disappears.
"""
struct Per{P} end
Per(t::NTuple{D,Bool}) where D = Per{t}()
@inline Base.getindex(::Per{P}, j::Integer) where P = @inbounds P[j]
Base.Tuple(::Per{P}) where P = P

# (`sum` would widen Int32 to Int64)
@inline lin(g::Grid{3}, I::NTuple{3,Int32}) = i32 + (I[1]-i32) + (I[2]-i32)*g.s[2] + (I[3]-i32)*g.s[3]
@inline lin(g::Grid{2}, I::NTuple{2,Int32}) = i32 + (I[1]-i32) + (I[2]-i32)*g.s[2]
@inline function coords(g::Grid{3}, l::Int32)
    l0 = l - i32
    k = l0 ÷ g.s[3]; r = l0 - k*g.s[3]
    j = r ÷ g.s[2]; i = r - j*g.s[2]
    return (i + i32, j + i32, k + i32)
end
@inline function coords(g::Grid{2}, l::Int32)
    l0 = l - i32
    j = l0 ÷ g.s[2]
    return (l0 - j*g.s[2] + i32, j + i32)
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
    th, bl = launch_dims(lo, hi)
    prod(bl) == 0 && return 0
    @cuda threads=th blocks=bl k(Int32.(lo), Int32.(hi), args...)
    return prod(bl)
end
# threads and blocks of `launch` over the box `lo:hi` (zero blocks for an empty box)
function launch_dims(lo::NTuple{D,<:Integer}, hi::NTuple{D,<:Integer}) where D
    ext = hi .- lo .+ 1
    th = threads_for(Val(D))
    any(ext .<= 0) && return th, (0, 0, 0)
    ext3 = (ext..., ntuple(_->1, 3-D)...)
    return th, cld.(ext3, th)
end
nblocks(lo, hi) = prod(launch_dims(lo, hi)[2])
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
# Cell of this thread on a face normal to j: the thread's (x,y) map to the two other
# directions. Written per dimension with literal tuple indices: with `j` known only at run
# time, a generic `t[d-1]` (which would be t[0] for d=1) is not a static index and some
# Julia/GPUCompiler versions reject it.
@inline function faceidx(lo::NTuple{3,Int32}, hi::NTuple{3,Int32}, j)
    t = tid3()
    I = j == 1 ? (lo[1], lo[2] + t[1] - i32, lo[3] + t[2] - i32) :
        j == 2 ? (lo[1] + t[1] - i32, lo[2], lo[3] + t[2] - i32) :
                 (lo[1] + t[1] - i32, lo[2] + t[2] - i32, lo[3])
    return I, (I[1] <= hi[1]) & (I[2] <= hi[2]) & (I[3] <= hi[3])
end
@inline function faceidx(lo::NTuple{2,Int32}, hi::NTuple{2,Int32}, j)
    t = tid3()
    I = j == 1 ? (lo[1], lo[2] + t[1] - i32) : (lo[1] + t[1] - i32, lo[2])
    return I, (I[1] <= hi[1]) & (I[2] <= hi[2])
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

Reduction scratch: per-block partials `ps/pm`, results on the device in `s` (`n` sums,
then `n` maxima, so that one copy fetches both; `m` is the maxima half), and the host
mirror `hs`.
"""
struct RedBuf
    s::CuVector{Float64}
    m::CuVector{Float64}
    ps::CuVector{Float32}
    pm::CuVector{Float32}
    hs::Vector{Float64}
end
function RedBuf(nb::Integer, n=8)
    s = CUDA.zeros(Float64, 2n)
    RedBuf(s, view(s, n+1:2n), CUDA.zeros(Float32,nb), CUDA.zeros(Float32,nb), zeros(Float64,2n))
end
# Room for the block partials of any reduction on `g`: a `launch` over a box inside the
# full array, or a 1D launch of 256 threads over at most `nc` items.
RedBuf(g::Grid) = RedBuf(max(nblocks(ntuple(_->1, length(g.n)), Int.(g.n)), cld(Int(g.nc), 256)))
"""
    hostcopy!(dst::Array, src::CuArray)

`copyto!(dst, src)` for small reads on the critical path. CUDA.jl's copy waits for the
stream cooperatively (a worker thread wakes the task), which leaves the GPU idle for
50–200 µs before the next launch; this blocks the calling thread in the driver instead.
"""
function hostcopy!(dst::Array{T}, src::CuArray{T}) where T
    s = CUDA.stream()
    CUDA.synchronize(s; blocking=true)
    # the stream is idle: a copy to pageable memory completes before the call returns
    GC.@preserve dst src unsafe_copyto!(pointer(dst), pointer(src), length(src); stream=s, async=true)
    return dst
end
function fetch!(b::RedBuf)
    hostcopy!(b.hs, b.s)
    return b
end
getsum(b::RedBuf, k) = b.hs[k]
getmax(b::RedBuf, k) = Float32(b.hs[length(b.m)+k])

"""
    red_launch(k, lo, hi, b::RedBuf, slot, args...)

Launch `k(lo,hi,ps,pm,args...)` (which calls `block_partial!`) and reduce the partials
into `b.s[slot]`, `b.m[slot]`.
"""
function red_launch(k::K, lo, hi, b::RedBuf, slot, args...) where K
    checkcap(b, nblocks(lo, hi))
    nb = launch(k, lo, hi, b.ps, b.pm, args...)
    nb == 0 && return nothing
    @cuda threads=1024 k_finish!(b.ps, b.pm, Int32(nb), b.s, b.m, Int32(slot))
    return nothing
end
"Throw before a launch that would write `nb` block partials into `b`."
checkcap(b::RedBuf, nb) = nb <= length(b.ps) ||
    throw(ErrorException("reduction scratch too small: $nb blocks, room for $(length(b.ps))"))
