# Large temporary buffers of plain bits outside the GC heap (docs/DESIGN.md §8).
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

struct ScratchArena
    blocks::Vector{Ptr{Cvoid}}
end
ScratchArena() = ScratchArena(Ptr{Cvoid}[])

function alloc!(arena::ScratchArena, ::Type{T}, n::Integer) where {T}
    (isbitstype(T) && Int(n) * sizeof(T) >= MALLOC_MIN_BYTES) || return Vector{T}(undef, n)
    nb = Int(n) * sizeof(T)
    p = Libc.malloc(nb)
    p == C_NULL && throw(OutOfMemoryError())
    push!(arena.blocks, p)
    count!(:malloc_bytes, nb)
    return unsafe_wrap(Array, Ptr{T}(p), Int(n); own = false)
end

function release!(arena::ScratchArena)
    foreach(Libc.free, arena.blocks)
    empty!(arena.blocks)
    return nothing
end
