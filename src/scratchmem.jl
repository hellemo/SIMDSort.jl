# Large temporary buffers of plain bits outside the GC heap (docs/DESIGN.md §8).
#
# A sort that allocates hundreds of MB of temporaries triggers full
# collections, and each one marks every live object. With 10^7 strings to
# sort that was 0.4 s of a 0.7 s sortperm on 8 threads, and on a 56-thread
# node String sortperm got slower than on one thread. Memory from malloc is not
# counted by the GC, so it triggers no collection.
#
#   buf, h = scratch_vector(T, n)
#   try ... finally free_scratch(h) end
#
# `buf` must not escape: it is freed by free_scratch. Small buffers, and
# buffers of non-bits types (which hold GC references), come from the GC
# (h == C_NULL). (No closure-taking
# helper: closures over reassigned variables get boxed.)

const MALLOC_MIN_BYTES = 1 << 20

function scratch_vector(::Type{T}, n::Integer) where {T}
    (isbitstype(T) && Int(n) * sizeof(T) >= MALLOC_MIN_BYTES) || return Vector{T}(undef, n), C_NULL
    nb = Int(n) * sizeof(T)
    p = Libc.malloc(nb)
    p == C_NULL && throw(OutOfMemoryError())
    count!(:malloc_bytes, nb)
    return unsafe_wrap(Array, Ptr{T}(p), Int(n); own = false), p
end

free_scratch(h::Ptr{Cvoid}) = (h == C_NULL || Libc.free(h); nothing)
