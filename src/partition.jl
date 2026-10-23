# In-place, bidirectional vectorized partition (docs/DESIGN.md §3).
# Offsets are 0-based element offsets from the pointer `p`.

@inline vld(::Val{N}, p::Ptr{T}, i::Int) where {N,T} = vload(Vec{N,T}, p + i * sizeof(T))
@inline vst(v::Vec, p::Ptr{T}, i::Int) where {T} = vstore(v, p + i * sizeof(T))
# U consecutive vectors from offset i. (A function barrier: a closure over the
# reassigned read positions would box them.)
@inline vldU(::Val{U}, ::Val{N}, p::Ptr, i::Int) where {U,N} =
    ntuple(u -> vld(Val(N), p, i + (u - 1) * N), Val(U))

# Software prefetch (read, high locality) of the cache line at `p`. The IR uses
# opaque pointers from LLVM 17 on, typed `i8*` pointers before.
@static if Base.libllvm_version >= v"17"
    @inline prefetch(p::Ptr) = Base.llvmcall(("""
        declare void @llvm.prefetch.p0(ptr, i32, i32, i32)
        define void @entry(i64 %a) #0 {
          %p = inttoptr i64 %a to ptr
          call void @llvm.prefetch.p0(ptr %p, i32 0, i32 3, i32 1)
          ret void
        }
        attributes #0 = { alwaysinline }
        """, "entry"), Cvoid, Tuple{UInt}, UInt(p))
else
    @inline prefetch(p::Ptr) = Base.llvmcall(("""
        declare void @llvm.prefetch.p0i8(i8*, i32, i32, i32)
        define void @entry(i64 %a) #0 {
          %p = inttoptr i64 %a to i8*
          call void @llvm.prefetch.p0i8(i8* %p, i32 0, i32 3, i32 1)
          ret void
        }
        attributes #0 = { alwaysinline }
        """, "entry"), Cvoid, Tuple{UInt}, UInt(p))
end

# Prefetch distance in keys, and the range size (bytes) from which prefetching
# pays off: below about the L2 size the hardware keeps up and the extra
# instructions only cost time.
const PFDIST = 512
const PF_MIN_BYTES = 2 << 20
use_prefetch(::Type{T}, n::Int) where {T} = n * sizeof(T) > PF_MIN_BYTES

# Partition one vector: keys ≤ pivot are written from the left write position
# `lw` upwards, the others end at the right write position `rw` (exclusive).
# Both stores are full width; the caller guarantees the extra lanes land in free
# slots.
@inline function part1(x::Vec{N,T}, pv::Vec{N,T}, p::Ptr{T}, lw::Int, rw::Int) where {N,T}
    y, c = vpartition(x, x <= pv)
    vst(y, p, lw)
    vst(y, p, rw - N)
    return lw + c, rw - (N - c)
end

@inline part_all(::Tuple{}, pv, p, lw, rw) = (lw, rw)
@inline function part_all(xs::Tuple, pv, p, lw, rw)
    lw, rw = part1(first(xs), pv, p, lw, rw)
    return part_all(Base.tail(xs), pv, p, lw, rw)
end

"""
    partition!(p, n, pivot, buf, Val(U), Val(PF)=Val(false), Val(branchfree=false)) -> m

Reorder the `n ≥ 2U*N` keys at `p` so that the first `m` are `≤ pivot` and the rest
are `> pivot`. `buf` must hold at least `2U*N + N` keys. `U` vectors are processed
per iteration. With `PF`, the keys `PFDIST` ahead of each read are prefetched.
"""
function partition!(p::Ptr{T}, n::Int, pivot::T, buf::Ptr{T}, ::Val{U},
                    ::Val{PF}=Val(false), ::Val{BF}=Val(false)) where {T,U,PF,BF}
    N = lanes(T)
    UN = U * N
    pv = Vec{N,T}(pivot)
    # Set aside UN keys from each end; this frees UN slots at both ends.
    unsafe_copyto!(buf, p, UN)
    unsafe_copyto!(buf + UN * sizeof(T), p + (n - UN) * sizeof(T), UN)
    left, right = UN, n - UN    # unread keys: [left, right)
    lw, rw = 0, n               # written: [0, lw) and [rw, n)

    # Load from the side with less free space; afterwards both sides have ≥ UN.
    while right - left >= UN
        fromleft = left - lw <= rw - right
        if BF
            src = ifelse(fromleft, left, right - UN)
            left += ifelse(fromleft, UN, 0)
            right -= ifelse(fromleft, 0, UN)
        elseif fromleft
            src = left
            left += UN
        else
            right -= UN
            src = right
        end
        if PF
            pa = p + (fromleft ? left + PFDIST : right - PFDIST) * sizeof(T)
            for c in 0:64:(UN * sizeof(T) - 1)
                prefetch(pa + c)
            end
        end
        lw, rw = part_all(vldU(Val(U), Val(N), p, src), pv, p, lw, rw)
    end
    while right - left >= N
        if left - lw <= rw - right
            src = left
            left += N
        else
            right -= N
            src = right
        end
        lw, rw = part1(vld(Val(N), p, src), pv, p, lw, rw)
    end

    # Fewer than N unread keys: append them to the buffer. Now [lw, rw) is free and
    # exactly as large as the buffer contents.
    r = right - left
    unsafe_copyto!(buf + 2UN * sizeof(T), p + left * sizeof(T), r)
    m = 2UN + r
    i = 0
    # While at least 2N keys remain, rw - lw ≥ 2N and the two stores are disjoint.
    while m - i >= 2N
        lw, rw = part1(vld(Val(N), buf, i), pv, p, lw, rw)
        i += N
    end
    # Exact scalar placement of the last < 2N keys: each key is written to both
    # ends of the free region, and only the end it belongs to advances.
    while i < m
        x = unsafe_load(buf, i + 1)
        unsafe_store!(p, x, lw + 1)
        unsafe_store!(p, x, rw)
        c = x <= pivot
        lw += c
        rw -= !c
        i += 1
    end
    return lw
end
