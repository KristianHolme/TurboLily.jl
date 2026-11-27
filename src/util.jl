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
@inline lin(g::AGrid{3}, I::NTuple{3,Int32}) = i32 + (I[1]-i32) + (I[2]-i32)*g.s[2] + (I[3]-i32)*g.s[3]
@inline lin(g::AGrid{2}, I::NTuple{2,Int32}) = i32 + (I[1]-i32) + (I[2]-i32)*g.s[2]
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

# TurboLily's arrays are always CuArrays: reject a request for anything else
check_mem(mem) = mem === CuArray ||
    throw(ArgumentError("TurboLily supports mem=CuArray only (got mem=$mem); use WaterLily for other array types"))

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

"""
    DirtyBox{D}

Bounding box of cells modified by `measure!` (and the `0 < μ₀ < 1` transition box for
Biot-Savart), shared between `TFlow` and `TPoisson` so `update!` and `body_box!` touch only
the changed region after body motion.
"""
mutable struct DirtyBox{D}
    valid :: Bool
    lo :: NTuple{D,Int}
    hi :: NTuple{D,Int}
    tvalid :: Bool
    tlo :: NTuple{D,Int}
    thi :: NTuple{D,Int}
end
DirtyBox(::Val{D}) where D = DirtyBox{D}(false, ntuple(_->2, D), ntuple(_->1, D), false, ntuple(_->2, D), ntuple(_->1, D))

# Warp-reduced axis-aligned bounding box of cells with `pred`, atomically merged into
# `bst[off+1:off+3]` (lo) and `bst[off+4:off+6]` (hi). All 32 lanes of the warp must call it.
@inline function warp_box!(bst, off, pred::Bool, I::NTuple{D,Int32}) where D
    vote_ballot_sync(0xffffffff, pred) == 0 && return nothing
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

# ---------------------------------------------------------------- reductions
# (sum,max) over the warp, returned on lane 1; `s` is a Float32 or Float64 sum
@inline function warp_red(s::Union{Float32,Float64}, m::Float32)
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
"Write this block's (sum1, sum2, max) partials to `ps/ps2/pm[block]`."
@inline function block_partial2!(ps, ps2, pm, s1::Float32, s2::Float32, m::Float32)
    shs1 = CuStaticSharedArray(Float32, 32)
    shs2 = CuStaticSharedArray(Float32, 32)
    shm = CuStaticSharedArray(Float32, 32)
    t = flat_tid()
    nt = blockDim().x*blockDim().y*blockDim().z
    off = Int32(16)
    while off > 0
        s1 += shfl_down_sync(0xffffffff, s1, off)
        s2 += shfl_down_sync(0xffffffff, s2, off)
        m = max(m, shfl_down_sync(0xffffffff, m, off))
        off >>= 1
    end
    lane = t % Int32(32) + i32
    wid = t ÷ Int32(32) + i32
    nw = (nt + Int32(31)) ÷ Int32(32)
    if lane == i32
        @inbounds shs1[wid] = s1; @inbounds shs2[wid] = s2; @inbounds shm[wid] = m
    end
    sync_threads()
    if wid == i32
        s1 = lane <= nw ? @inbounds(shs1[lane]) : 0f0
        s2 = lane <= nw ? @inbounds(shs2[lane]) : 0f0
        m = lane <= nw ? @inbounds(shm[lane]) : 0f0
        off = Int32(16)
        while off > 0
            s1 += shfl_down_sync(0xffffffff, s1, off)
            s2 += shfl_down_sync(0xffffffff, s2, off)
            m = max(m, shfl_down_sync(0xffffffff, m, off))
            off >>= 1
        end
        if lane == i32
            b = flat_bid()
            @inbounds ps[b] = s1; @inbounds ps2[b] = s2; @inbounds pm[b] = m
        end
    end
    return nothing
end
@inline function warp_red_sm2(s::Union{Float32,Float64}, m1::Float32, m2::Float32)
    off = Int32(16)
    while off > 0
        s += shfl_down_sync(0xffffffff, s, off)
        m1 = max(m1, shfl_down_sync(0xffffffff, m1, off))
        m2 = max(m2, shfl_down_sync(0xffffffff, m2, off))
        off >>= 1
    end
    return s, m1, m2
end
@inline function block_red_sm2(s::S, m1::Float32, m2::Float32) where S
    shs = CuStaticSharedArray(S, 32)
    shm1 = CuStaticSharedArray(Float32, 32)
    shm2 = CuStaticSharedArray(Float32, 32)
    t = flat_tid()
    nt = blockDim().x*blockDim().y*blockDim().z
    s, m1, m2 = warp_red_sm2(s, m1, m2)
    lane = t % Int32(32) + i32
    wid = t ÷ Int32(32) + i32
    nw = (nt + Int32(31)) ÷ Int32(32)
    lane == i32 && (@inbounds shs[wid] = s; @inbounds shm1[wid] = m1; @inbounds shm2[wid] = m2)
    sync_threads()
    if wid == i32
        s = lane <= nw ? @inbounds(shs[lane]) : zero(S)
        m1 = lane <= nw ? @inbounds(shm1[lane]) : 0f0
        m2 = lane <= nw ? @inbounds(shm2[lane]) : 0f0
        s, m1, m2 = warp_red_sm2(s, m1, m2)
    end
    return s, m1, m2
end
@inline function warp_red2(s1::Float64, s2::Float64, m::Float32)
    off = Int32(16)
    while off > 0
        s1 += shfl_down_sync(0xffffffff, s1, off)
        s2 += shfl_down_sync(0xffffffff, s2, off)
        m = max(m, shfl_down_sync(0xffffffff, m, off))
        off >>= 1
    end
    return s1, s2, m
end
@inline function block_red2(s1::Float64, s2::Float64, m::Float32)
    shs1 = CuStaticSharedArray(Float64, 32)
    shs2 = CuStaticSharedArray(Float64, 32)
    shm = CuStaticSharedArray(Float32, 32)
    t = flat_tid()
    nt = blockDim().x*blockDim().y*blockDim().z
    s1, s2, m = warp_red2(s1, s2, m)
    lane = t % Int32(32) + i32
    wid = t ÷ Int32(32) + i32
    nw = (nt + Int32(31)) ÷ Int32(32)
    lane == i32 && (@inbounds shs1[wid] = s1; @inbounds shs2[wid] = s2; @inbounds shm[wid] = m)
    sync_threads()
    if wid == i32
        s1 = lane <= nw ? @inbounds(shs1[lane]) : 0.0
        s2 = lane <= nw ? @inbounds(shs2[lane]) : 0.0
        m = lane <= nw ? @inbounds(shm[lane]) : 0f0
        s1, s2, m = warp_red2(s1, s2, m)
    end
    return s1, s2, m
end
"Write this block's (sum, max1, max2) partials to `ps/pm/pm2[block]`."
@inline function block_partial_sm2!(ps, pm, pm2, s::Float32, m1::Float32, m2::Float32)
    s, m1, m2 = block_red_sm2(s, m1, m2)
    if flat_tid() == 0
        b = flat_bid()
        @inbounds ps[b] = s; @inbounds pm[b] = m1; @inbounds pm2[b] = m2
    end
    return nothing
end
@inline skip_fix(::Nothing) = false
@inline skip_fix(fixd) = abs(@inbounds(fixd[1])) <= @inbounds(fixd[2])
# (sum,max) of the block partials 1:nb, grid-strided; block b writes its result to outs/outm[k+b-1]
function k_finish!(ps, pm, nb::Int32, outs, outm, k::Int32, fixd=nothing)
    skip_fix(fixd) && return nothing
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
# (sum1, sum2, max) of the block partials 1:nb, grid-strided
function k_finish_2s!(ps, ps2, pm, nb::Int32, outs1, outs2, outm, k1::Int32, k2::Int32)
    s1 = 0.0; s2 = 0.0; m = 0f0
    i = (blockIdx().x - i32)*blockDim().x + threadIdx().x
    while i <= nb
        @inbounds s1 += Float64(ps[i]); @inbounds s2 += Float64(ps2[i]); @inbounds m = max(m, pm[i])
        i += gridDim().x*blockDim().x
    end
    s1, s2, m = block_red2(s1, s2, m)
    if threadIdx().x == i32
        b = blockIdx().x - i32
        @inbounds outs1[k1 + b] = s1; @inbounds outs2[k2 + b] = s2; @inbounds outm[k2 + b] = m
    end
    return nothing
end
# (sum,max1,max2) of the block partials 1:nb, grid-strided
function k_finish_sm2!(ps, pm, pm2, nb::Int32, outs, outm, outm2, k1::Int32, k2::Int32)
    s = 0.0; m1 = 0f0; m2 = 0f0
    i = (blockIdx().x - i32)*blockDim().x + threadIdx().x
    while i <= nb
        @inbounds s += Float64(ps[i]); @inbounds m1 = max(m1, pm[i]); @inbounds m2 = max(m2, pm2[i])
        i += gridDim().x*blockDim().x
    end
    s, m1, m2 = block_red_sm2(s, m1, m2)
    if threadIdx().x == i32
        b = blockIdx().x - i32
        @inbounds outs[k1 + b] = s; @inbounds outm[k1 + b] = m1; @inbounds outm2[k2 + b] = m2
    end
    return nothing
end
# (sum,max) of Float64/Float32 values 1:nq in one block
function k_finish2!(qs, qm, nq::Int32, outs, outm, k::Int32, fixd=nothing)
    skip_fix(fixd) && return nothing
    i = threadIdx().x
    s = i <= nq ? @inbounds(qs[i]) : 0.0; m = i <= nq ? @inbounds(qm[i]) : 0f0
    s, m = block_red(s, m)
    if threadIdx().x == i32
        @inbounds outs[k] = s; @inbounds outm[k] = m
    end
    return nothing
end
function k_finish2_2s!(qs, qs2, qm, nq::Int32, outs, outm, k1::Int32, k2::Int32)
    i = threadIdx().x
    s1 = i <= nq ? @inbounds(qs[i]) : 0.0; s2 = i <= nq ? @inbounds(qs2[i]) : 0.0; m = i <= nq ? @inbounds(qm[i]) : 0f0
    s1, s2, m = block_red2(s1, s2, m)
    if threadIdx().x == i32
        @inbounds outs[k1] = s1; @inbounds outs[k2] = s2; @inbounds outm[k1] = m; @inbounds outm[k2] = m
    end
    return nothing
end
function k_finish2_sm2!(qs, qm, qm2, nq::Int32, outs, outm, k1::Int32, k2::Int32)
    i = threadIdx().x
    s = i <= nq ? @inbounds(qs[i]) : 0.0; m1 = i <= nq ? @inbounds(qm[i]) : 0f0; m2 = i <= nq ? @inbounds(qm2[i]) : 0f0
    s, m1, m2 = block_red_sm2(s, m1, m2)
    if threadIdx().x == i32
        @inbounds outs[k1] = s; @inbounds outm[k1] = m1; @inbounds outm[k2] = m2
    end
    return nothing
end

"""
    RedBuf(nblocks)

Reduction scratch: per-block partials `ps/pm`, results on the device in `s` (`n` sums,
then `n` maxima, so that one copy fetches both; `m` is the maxima half), and the host
mirror `hs`.

A result lives in a slot `1:n`. Each file names the slots it uses on its buffer as `SLOT_*`
constants (a flow and a pressure solver have separate buffers); kernels that read a sum on
the device index `s` with the same constants.
"""
struct RedBuf
    s::CuVector{Float64}
    m::CuVector{Float64}
    ps::CuVector{Float32}
    ps2::CuVector{Float32}   # second per-block sum partial (e.g. Biot-Savart residual L1)
    pm::CuVector{Float32}
    hs::Vector{Float64}
    qs::CuVector{Float64}    # per-block results of the first level of `finish!`
    qs2::CuVector{Float64}
    qm::CuVector{Float32}
    qm2::CuVector{Float32}
end
const FINISH_BLOCKS = 64
function RedBuf(nb::Integer, n=8)
    s = CUDA.zeros(Float64, 2n)
    RedBuf(s, view(s, n+1:2n), CUDA.zeros(Float32,nb), CUDA.zeros(Float32,nb), CUDA.zeros(Float32,nb),
        zeros(Float64,2n), CUDA.zeros(Float64, FINISH_BLOCKS), CUDA.zeros(Float64, FINISH_BLOCKS),
        CUDA.zeros(Float32, FINISH_BLOCKS), CUDA.zeros(Float32, FINISH_BLOCKS))
end
"""
    finish!(b::RedBuf, nb, outs, outm, k; ps=b.ps)

Reduce the block partials `ps/b.pm[1:nb]` into `outs[k]` (Float64 sum) and `outm[k]` (max).
FP64 runs at 1/32 rate on GeForce cards, so many partials are first reduced by
`FINISH_BLOCKS` blocks, then by one.
"""
function finish!(b::RedBuf, nb, outs, outm, k; ps=b.ps, fixd=nothing)
    if nb <= 16384
        @cuda threads=1024 k_finish!(ps, b.pm, Int32(nb), outs, outm, Int32(k), fixd)
    else
        @cuda threads=256 blocks=FINISH_BLOCKS k_finish!(ps, b.pm, Int32(nb), b.qs, b.qm, i32, fixd)
        @cuda threads=FINISH_BLOCKS k_finish2!(b.qs, b.qm, Int32(FINISH_BLOCKS), outs, outm, Int32(k), fixd)
    end
    return nothing
end
"Reduce `(b.ps, b.pm) -> (outs[k1], outm[k1])` and `(b.ps2, b.pm) -> (outs[k2], outm[k2])` in one pass."
function finish2!(b::RedBuf, nb, outs, outm, k1, k2)
    if nb <= 16384
        @cuda threads=1024 k_finish_2s!(b.ps, b.ps2, b.pm, Int32(nb), outs, outs, outm, Int32(k1), Int32(k2))
    else
        @cuda threads=256 blocks=FINISH_BLOCKS k_finish_2s!(b.ps, b.ps2, b.pm, Int32(nb), b.qs, b.qs2, b.qm, i32, i32)
        @cuda threads=FINISH_BLOCKS k_finish2_2s!(b.qs, b.qs2, b.qm, Int32(FINISH_BLOCKS), outs, outm, Int32(k1), Int32(k2))
    end
    return nothing
end
"Reduce `(b.ps, b.pm) -> (outs[k1], outm[k1])` and `b.ps2 -> outm[k2]` in one reduction pass."
function finish_sm2!(b::RedBuf, nb, outs, outm, k1, k2)
    if nb <= 16384
        @cuda threads=1024 k_finish_sm2!(b.ps, b.pm, b.ps2, Int32(nb), outs, outm, outm, Int32(k1), Int32(k2))
    else
        @cuda threads=256 blocks=FINISH_BLOCKS k_finish_sm2!(b.ps, b.pm, b.ps2, Int32(nb), b.qs, b.qm, b.qm2, i32, i32)
        @cuda threads=FINISH_BLOCKS k_finish2_sm2!(b.qs, b.qm, b.qm2, Int32(FINISH_BLOCKS), outs, outm, Int32(k1), Int32(k2))
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
    length(dst) >= length(src) || throw(BoundsError(dst, length(src)))
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
