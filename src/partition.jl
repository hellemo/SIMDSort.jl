# In-place, bidirectional vectorized partition (docs/DESIGN.md §3).
#
# [Bra17, Algorithm 3]: keys set aside at both ends, the next vectors loaded from
# the side with less free space, each vector split to both ends. [WBGS22 §2.1]:
# unrolled loop, the last keys through a buffer. Our own: vpartition + two
# full-width stores (instead of compress-stores), software prefetch, and the
# three-way variant.
# Offsets are 0-based element offsets into the key reference `p` (keyref.jl).

# U consecutive vectors from offset i. (A function barrier: a closure over the
# reassigned read positions would box them.)
@inline vldU(::Val{U}, ::Val{N}, p, i::Int) where {U,N} =
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
@inline function part1(x::Vec{N,T}, pv::Vec{N,T}, p::KeyRef{T}, lw::Int, rw::Int) where {N,T}
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
function partition!(p::KeyRef{T}, n::Int, pivot::T, buf::KeyRef{T}, ::Val{U},
                    ::Val{PF}=Val(false), ::Val{BF}=Val(false)) where {T,U,PF,BF}
    N = lanes(T)
    UN = U * N
    pv = Vec{N,T}(pivot)
    # Set aside UN keys from each end; this frees UN slots at both ends.
    copykeys!(buf, p, UN)
    copykeys!(shift(buf, UN), shift(p, n - UN), UN)
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
            pa = fromleft ? left + PFDIST : right - PFDIST
            for c in 0:(64 ÷ sizeof(T)):(UN - 1)     # one prefetch per 64-byte line
                prefetch(p, pa + c)
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
    copykeys!(shift(buf, 2UN), shift(p, left), r)
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
        x = ld(buf, i + 1)
        st!(p, x, lw + 1)
        st!(p, x, rw)
        c = x <= pivot
        lw += c
        rw -= !c
        i += 1
    end
    return lw
end

# Three-way partition of one vector: keys < pivot go left, keys > pivot go right,
# keys equal to the pivot are dropped (the caller refills them).
@inline function part3(x::Vec{N,T}, pv::Vec{N,T}, p::KeyRef{T}, lw::Int, rw::Int) where {N,T}
    y, c = vpartition(x, x < pv)
    vst(y, p, lw)
    z, d = vpartition(x, !(x > pv))     # the keys > pivot come last
    vst(z, p, rw - N)
    return lw + c, rw - (N - d)
end

@inline part3_all(::Tuple{}, pv, p, lw, rw) = (lw, rw)
@inline function part3_all(xs::Tuple, pv, p, lw, rw)
    lw, rw = part3(first(xs), pv, p, lw, rw)
    return part3_all(Base.tail(xs), pv, p, lw, rw)
end

"""
    partition3!(p, n, pivot, buf, Val(U), Val(PF)=Val(false)) -> (a, b)

Three-way version of `partition!`: afterwards the keys in `[0, a)` are below the
pivot, those in `[b, n)` above it, and `[a, b)` holds the pivot value.
"""
function partition3!(p::KeyRef{T}, n::Int, pivot::T, buf::KeyRef{T}, ::Val{U},
                     ::Val{PF}=Val(false)) where {T,U,PF}
    N = lanes(T)
    UN = U * N
    pv = Vec{N,T}(pivot)
    copykeys!(buf, p, UN)
    copykeys!(shift(buf, UN), shift(p, n - UN), UN)
    left, right = UN, n - UN
    lw, rw = 0, n
    # Dropped keys only add free space, so the argument of `partition!` holds.
    while right - left >= UN
        fromleft = left - lw <= rw - right
        if fromleft
            src = left
            left += UN
        else
            right -= UN
            src = right
        end
        if PF
            pa = fromleft ? left + PFDIST : right - PFDIST
            for c in 0:(64 ÷ sizeof(T)):(UN - 1)     # one prefetch per 64-byte line
                prefetch(p, pa + c)
            end
        end
        lw, rw = part3_all(vldU(Val(U), Val(N), p, src), pv, p, lw, rw)
    end
    while right - left >= N
        if left - lw <= rw - right
            src = left
            left += N
        else
            right -= N
            src = right
        end
        lw, rw = part3(vld(Val(N), p, src), pv, p, lw, rw)
    end
    r = right - left
    copykeys!(shift(buf, 2UN), shift(p, left), r)
    m = 2UN + r
    i = 0
    # [lw, rw) holds at least as many slots as keys left in the buffer.
    while m - i >= N && rw - lw >= 2N
        lw, rw = part3(vld(Val(N), buf, i), pv, p, lw, rw)
        i += N
    end
    while i < m
        x = ld(buf, i + 1)
        st!(p, x, lw + 1)
        st!(p, x, rw)
        lw += x < pivot
        rw -= x > pivot
        i += 1
    end
    k = lw
    while k + N <= rw
        vst(pv, p, k)
        k += N
    end
    while k < rw
        st!(p, pivot, k + 1)
        k += 1
    end
    return lw, rw
end

# 64-bit keys (4 lanes): one permutation, looked up by the < and > masks
# together, orders a vector as [< pivot | = pivot | > pivot]; both stores use it.
const P3TAB4 = let t = Int32[]
    for code in 0:255
        L, G = code & 15, code >> 4
        lanes_lt = [i for i in 0:3 if (L >> i) & 1 == 1]
        lanes_gt = [i for i in 0:3 if (G >> i) & 1 == 1 && (L >> i) & 1 == 0]
        lanes_eq = [i for i in 0:3 if i ∉ lanes_lt && i ∉ lanes_gt]
        for i in [lanes_lt; lanes_eq; lanes_gt]
            push!(t, Int32(2i), Int32(2i + 1))
        end
    end
    t
end

@inline function part3(x::Vec{4,T}, pv::Vec{4,T}, p::KeyRef{T}, lw::Int, rw::Int) where {T}
    L = bitmask(x < pv) % Int
    G = bitmask(x > pv) % Int
    idx = @inbounds vload(Vec{8,Int32}, P3TAB4, 8 * (L | (G << 4)) + 1)
    y = reinterpret(Vec{4,T}, vpermute(reinterpret(Vec{8,Int32}, x), idx))
    vst(y, p, lw)
    vst(y, p, rw - 4)
    return lw + count_ones(L), rw - count_ones(G)
end
