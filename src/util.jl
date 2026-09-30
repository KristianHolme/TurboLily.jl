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

# compile-time unrolled loop over 1:N
@inline unroll(f::F, ::Val{N}) where {F,N} = (ntuple(i->(f(i); nothing), Val(N)); nothing)

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
    any(ext .<= 0) && return nothing
    th = threads_for(Val(D))
    ext3 = (ext..., ntuple(_->1, 3-D)...)
    bl = cld.(ext3, th)
    @cuda threads=th blocks=bl k(Int32.(lo), Int32.(hi), args...)
    nothing
end
function launch1(k::K, n::Integer, args...; threads=256) where K
    n <= 0 && return nothing
    @cuda threads=threads blocks=cld(n,threads) k(Int32(n), args...)
    nothing
end
inside_lo(::Grid{D}) where D = ntuple(_->2, D)
inside_hi(g::Grid{D}) where D = ntuple(d->Int(g.n[d])-1, D)

# ---------------------------------------------------------------- reductions
# Block reduction of a (sum,max) pair; thread 1 of each block atomically
# accumulates into outs[k] (Float64 sum) and outm[k] (max of non-negative Float32,
# stored as its Int32 bit pattern, which orders like the float).
@inline function warp_red(s::Float32, m::Float32)
    off = Int32(16)
    while off > 0
        s += shfl_down_sync(0xffffffff, s, off)
        m = max(m, shfl_down_sync(0xffffffff, m, off))
        off >>= 1
    end
    s, m
end
@inline function block_red_atomic!(outs, outm, k, s::Float32, m::Float32)
    shs = CuStaticSharedArray(Float32, 32)
    shm = CuStaticSharedArray(Float32, 32)
    s, m = warp_red(s, m)
    lane = (threadIdx().x - i32) % Int32(32) + i32
    wid = (threadIdx().x - i32) ÷ Int32(32) + i32
    nw = (blockDim().x + Int32(31)) ÷ Int32(32)
    if lane == i32
        @inbounds shs[wid] = s; @inbounds shm[wid] = m
    end
    sync_threads()
    if wid == i32
        s = lane <= nw ? @inbounds(shs[lane]) : 0f0
        m = lane <= nw ? @inbounds(shm[lane]) : 0f0
        s, m = warp_red(s, m)
        if lane == i32
            CUDA.@atomic outs[k] += Float64(s)
            CUDA.@atomic outm[k] = max(outm[k], reinterpret(Int32, m))
        end
    end
    nothing
end

"Host-side reduction buffers: `s` (Float64 sums), `m` (Int32-encoded maxima)."
struct RedBuf
    s::CuVector{Float64}
    m::CuVector{Int32}
    hs::Vector{Float64}
    hm::Vector{Int32}
end
RedBuf(n=8) = RedBuf(CUDA.zeros(Float64,n), CUDA.zeros(Int32,n), zeros(Float64,n), zeros(Int32,n))
reset!(b::RedBuf) = (fill!(b.s,0); fill!(b.m,0); b)
function fetch!(b::RedBuf)
    copyto!(b.hs, b.s); copyto!(b.hm, b.m)
    b
end
getsum(b::RedBuf,k) = b.hs[k]
getmax(b::RedBuf,k) = reinterpret(Float32, b.hm[k])
