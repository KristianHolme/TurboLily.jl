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

"""
    SGrid{D,N}

`Grid` with the array size `N` as a type parameter: strides and component offsets are
compile-time constants, so neighbour offsets become immediates of the loads. Costs one
compilation per grid size; used by the momentum core only.
"""
struct SGrid{D,N} end
SGrid(g::Grid{D}) where D = SGrid{D,Tuple(Int.(g.n))}()
@inline function Base.getproperty(::SGrid{D,N}, s::Symbol) where {D,N}
    s === :n && return Int32.(N)
    s === :s && return ntuple(d-> Int32(prod(N[1:d-1])), Val(D))
    s === :nc && return Int32(prod(N))
    error("SGrid has no property $s")
end
const AGrid{D} = Union{Grid{D},SGrid{D}}

# (`sum` would widen Int32 to Int64)
@inline lin(g::SGrid{3}, I::NTuple{3,Int32}) = i32 + (I[1]-i32) + (I[2]-i32)*g.s[2] + (I[3]-i32)*g.s[3]
@inline lin(g::SGrid{2}, I::NTuple{2,Int32}) = i32 + (I[1]-i32) + (I[2]-i32)*g.s[2]
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
# block shape of `launch` for kernel `k` over a box of extent `ext`
kernel_threads(k, ::Val{D}, ext) where D = threads_for(Val(D))
# From this many cells in an x-y plane, the halo planes of a stencil kernel no longer stay in a
# 6 MB L2 between neighbouring blocks (see AGENTS.md); smaller boxes keep the default shape.
const BIG_PLANE = 40_000

"""
    launch(k, lo, hi, args...)

Launch kernel `k(lo,hi,args...)` with one thread per cell of the box `lo:hi`, in blocks of
`kernel_threads(k, Val(D), ext)`. A block of more than `SAFE_THREADS` threads may exceed what the
compiled kernel can run (its register use limits the block size, more so on some architectures):
then the shape is halved in its last dimension until it fits if `shrinkable(k)`, and it is an
error otherwise. `cap`: most blocks the caller has room for.
"""
function launch(k::K, lo::NTuple{D,<:Integer}, hi::NTuple{D,<:Integer}, args...;
                th=kernel_threads(k, Val(D), hi .- lo .+ 1), cap=typemax(Int)) where {K,D}
    th, bl = launch_dims(lo, hi, th)
    prod(bl) == 0 && return 0
    if prod(th) > SAFE_THREADS
        kern = @cuda launch=false k(Int32.(lo), Int32.(hi), args...)
        maxthr = CUDA.maxthreads(kern)
        if prod(th) > maxthr
            shrinkable(k) || error("$k needs blocks of $(prod(th)) threads, but its register use allows $maxthr")
            th, bl = launch_dims(lo, hi, fit_threads(th, maxthr))
        end
    end
    prod(bl) <= cap || throw(ArgumentError("$k needs $(prod(bl)) blocks, there is room for $cap"))
    @cuda threads=th blocks=bl k(Int32.(lo), Int32.(hi), args...)
    return prod(bl)
end
# A block of up to 256 threads always fits: the register file holds 64k registers per block and a
# thread uses at most 255. The query of the kernel's limit is only made above that, which keeps
# the host cost of the common launch what a plain `@cuda` has.
const SAFE_THREADS = 256
"""
    fit_threads(th, maxthr)

The block shape `th` with its last dimension above 1 halved (repeatedly) until it has at most
`maxthr` threads.
"""
function fit_threads(th::NTuple{N,Int}, maxthr::Integer) where N
    maxthr >= 1 || throw(ArgumentError("a block of at most $maxthr threads"))
    while prod(th) > maxthr
        j = something(findlast(>(1), th), 1)
        th = Base.setindex(th, th[j] ÷ 2, j)
    end
    return th
end
# Whether `launch` may shrink the block shape of `k`: true for kernels that index cells by
# `cellidx` and reduce with `block_partial!`, false where the shape is part of the algorithm
# (the Jacobi kernels' fused restriction needs whole 2ᵈ coarsening blocks).
shrinkable(k) = false
# threads and blocks of `launch` over the box `lo:hi` (zero blocks for an empty box)
function launch_dims(lo::NTuple{D,<:Integer}, hi::NTuple{D,<:Integer}, th=threads_for(Val(D))) where D
    ext = hi .- lo .+ 1
    any(ext .<= 0) && return th, (0, 0, 0)
    ext3 = (ext..., ntuple(_->1, 3-D)...)
    return th, cld.(ext3, th)
end
# the most blocks any block shape of `launch` gives over the box (room for the block partials)
nblocks(lo, hi) = maximum(th -> prod(launch_dims(lo, hi, th)[2]), shapes(Val(length(lo))))
shapes(::Val{D}) where D = (threads_for(Val(D)),)
shapes(::Val{3}) = ((32,4,2), (32,8,1), (32,1,8))
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
# (sum,max) of the block partials 1:nb, grid-strided; block b writes its result to outs/outm[k+b-1]
function k_finish!(ps, pm, nb::Int32, outs, outm, k::Int32)
    s = 0.0; m = 0f0
    i = (blockIdx().x - i32)*blockDim().x + threadIdx().x
    while i <= nb
        @inbounds s += Float64(ps[i]); @inbounds m = max(m, pm[i])
        i += gridDim().x*blockDim().x
    end
    s, m = block_red(s, m)
    if threadIdx().x == i32
        @inbounds outs[k + blockIdx().x - i32] = s; @inbounds outm[k + blockIdx().x - i32] = m
    end
    return nothing
end
# (sum,max) of Float64/Float32 values 1:nq in one block
function k_finish2!(qs, qm, nq::Int32, outs, outm, k::Int32)
    i = threadIdx().x
    s = i <= nq ? @inbounds(qs[i]) : 0.0; m = i <= nq ? @inbounds(qm[i]) : 0f0
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
    qs::CuVector{Float64}    # per-block results of the first level of `finish!`
    qm::CuVector{Float32}
end
const FINISH_BLOCKS = 64
function RedBuf(nb::Integer, n=8)
    s = CUDA.zeros(Float64, 2n)
    RedBuf(s, view(s, n+1:2n), CUDA.zeros(Float32,nb), CUDA.zeros(Float32,nb), zeros(Float64,2n),
        CUDA.zeros(Float64, FINISH_BLOCKS), CUDA.zeros(Float32, FINISH_BLOCKS))
end
"""
    finish!(b::RedBuf, nb, outs, outm, k)

Reduce the block partials `b.ps/pm[1:nb]` into `outs[k]` (Float64 sum) and `outm[k]` (max).
FP64 runs at 1/32 rate on GeForce cards, so many partials are first reduced by
`FINISH_BLOCKS` blocks, then by one.
"""
function finish!(b::RedBuf, nb, outs, outm, k)
    if nb <= 16384
        @cuda threads=1024 k_finish!(b.ps, b.pm, Int32(nb), outs, outm, Int32(k))
    else
        @cuda threads=256 blocks=FINISH_BLOCKS k_finish!(b.ps, b.pm, Int32(nb), b.qs, b.qm, i32)
        @cuda threads=FINISH_BLOCKS k_finish2!(b.qs, b.qm, Int32(FINISH_BLOCKS), outs, outm, Int32(k))
    end
    return nothing
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
    nb = launch(k, lo, hi, b.ps, b.pm, args...; cap=length(b.ps))
    nb == 0 && return nothing
    finish!(b, nb, b.s, b.m, slot)
    return nothing
end
"Throw before a launch that would write `nb` block partials into `b`."
checkcap(b::RedBuf, nb) = nb <= length(b.ps) ||
    throw(ErrorException("reduction scratch too small: $nb blocks, room for $(length(b.ps))"))
