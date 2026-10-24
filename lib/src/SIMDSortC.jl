"""
    SIMDSortC

C entry points for SIMDSort.jl, for a shared library built with JuliaC
(`lib/build.sh`). For each of the six primitive types (suffix `i32`, `i64`,
`u32`, `u64`, `f32`, `f64`):

    int simdsort_<t>(T *data, size_t n, int rev, int nthreads);
    int simdsortperm_<t>(const T *data, size_t n, int64_t *perm, int rev, int nthreads);

`simdsort_<t>` sorts `data` in place; `simdsortperm_<t>` writes the stable
sort permutation as 0-based indices to `perm`. Both return 0. The order is
`isless` (NaNs last, -0.0 before 0.0), reversed if `rev != 0`; at most
`nthreads` tasks run at once (the threads of the runtime are set when the
library is loaded, by `JULIA_NUM_THREADS`).
"""
module SIMDSortC

using SIMDSort

for (T, s) in ((Int32, :i32), (Int64, :i64), (UInt32, :u32), (UInt64, :u64),
               (Float32, :f32), (Float64, :f64))
    sortname = Symbol(:simdsort_, s)
    permname = Symbol(:simdsortperm_, s)
    @eval begin
        Base.@ccallable function $sortname(data::Ptr{$T}, n::Csize_t, rev::Cint, nthreads::Cint)::Cint
            # The caller's buffer, wrapped without copying (the C boundary).
            v = unsafe_wrap(Array, data, Int(n); own = false)
            simdsort!(v; rev = rev != 0, ntasks = max(1, Int(nthreads)))
            return Cint(0)
        end
        Base.@ccallable function $permname(data::Ptr{$T}, n::Csize_t, perm::Ptr{Int64}, rev::Cint,
                                           nthreads::Cint)::Cint
            v = unsafe_wrap(Array, data, Int(n); own = false)
            out = unsafe_wrap(Array, perm, Int(n); own = false)
            p = simdsortperm(v; rev = rev != 0, ntasks = max(1, Int(nthreads)))
            @inbounds for i in eachindex(p)
                out[i] = p[i] - 1
            end
            return Cint(0)
        end
    end
end

"""
    simdsort_nthreads() -> Cint

The number of threads of the library's Julia runtime (set when the library
is loaded).
"""
Base.@ccallable simdsort_nthreads()::Cint = Cint(Threads.nthreads())

end
