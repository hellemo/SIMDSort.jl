# Large temporary buffers of plain bits outside the GC heap (docs/src/design.md §8).
#
# A sort that allocates hundreds of MB of temporaries triggers full
# collections, and each one marks every live object. With 10^7 strings to
# sort that was 0.4 s of a 0.7 s sortperm on 8 threads, and on a 56-thread
# node String sortperm got slower than on one thread. Memory from malloc is not
# counted by the GC, so it triggers no collection.
#
#   arena = ScratchArena()
#   try
#       buf = alloc!(arena, T, n)
#       ...
#   finally
#       release!(arena)
#   end
#
# Buffers from an arena must not outlive release!. Small buffers, and buffers of
# non-bits types (which hold GC references), are ordinary GC arrays. An arena
# is filled from one task (allocate before spawning); its buffers may be used
# by any task. (No closure-taking helper: closures over reassigned variables
# get boxed.)

const MALLOC_MIN_BYTES = 1 << 20

"""
    ScratchArena()

Large temporary buffers outside the GC heap: `alloc!(arena, T, n)` gives a
`Vector{T}` (malloc'ed for bits types of `MALLOC_MIN_BYTES` or more, an
ordinary array otherwise), and `release!(arena)` frees them all; call it in a
`finally`. See the top of `src/scratchmem.jl` for why.
"""
struct ScratchArena
    blocks::Vector{Ptr{Cvoid}}
end
ScratchArena() = ScratchArena(Ptr{Cvoid}[])

"""
    alloc!(arena, T, n) -> Vector{T}

A buffer of `n` elements of type `T` that lives until `release!(arena)`.
"""
function alloc!(arena::ScratchArena, ::Type{T}, n::Integer) where {T}
    (isbitstype(T) && Int(n) * sizeof(T) >= MALLOC_MIN_BYTES) || return Vector{T}(undef, n)
    nb = Int(n) * sizeof(T)
    p = Libc.malloc(nb)
    p == C_NULL && throw(OutOfMemoryError())
    push!(arena.blocks, p)
    count!(:malloc_bytes, nb)
    return unsafe_wrap(Array, Ptr{T}(p), Int(n); own = false)
end

"""
    release!(arena)

Free the arena's malloc'ed buffers (they must no longer be used).
"""
function release!(arena::ScratchArena)
    foreach(Libc.free, arena.blocks)
    empty!(arena.blocks)
    return nothing
end
