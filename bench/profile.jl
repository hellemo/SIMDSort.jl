# Where does the time go? Three measurements behind docs/src/design.md §7:
#   1. share of the sort spent in each component (statistical profile),
#   2. partition cost per key by range size, with and without prefetch,
#   3. base-case cost per call by range size.
#
#   julia --project=bench bench/profile.jl

using SIMDSort, Profile, Printf, Random
const Q = SIMDSort

const COMPONENTS = (:partition!, :small_sort!, :choose_pivot, :vextrema, :map_keys!,
                    :unmap_keys!, :mapkeys!, :float_scan, :nan_partition!, :order_zeros!,
                    :counting_sort!, :presorted!)

function components(label, a)
    b = copy(a)
    simdsort!(copy(a[1:10^5]))
    Profile.clear()
    Profile.init(n = 10^7, delay = 0.0005)
    @profile for _ in 1:3
        copyto!(b, a)
        simdsort!(b)
    end
    data = Profile.fetch()
    lidict = Profile.getdict(data)
    cnt = Dict{Symbol,Int}()
    seen = Set{Symbol}()
    for ip in data
        if ip == 0          # end of one backtrace
            foreach(f -> cnt[f] = get(cnt, f, 0) + 1, seen)
            empty!(seen)
        else
            foreach(fr -> push!(seen, fr.func), get(lidict, ip, []))
        end
    end
    total = cnt[:simdsort!]
    shares = [@sprintf("%s %.0f%%", f, 100 * cnt[f] / total) for f in COMPONENTS if haskey(cnt, f)]
    println(rpad(label, 18), join(shares, ", "))
end

function partition_cost(::Type{T}, n, prefetch) where {T}
    a = rand(T, n)
    b = copy(a)
    buf = Q.scratch(T)
    reps = max(1, 2^22 ÷ n)
    t = minimum(1:5) do _
        s = 0.0
        for _ in 1:reps
            copyto!(b, a)
            s += @elapsed GC.@preserve b buf Q.partition!(pointer(b), n, zero(T), pointer(buf),
                                                           Val(Q.UNROLL), Val(prefetch))
        end
        s / reps
    end
    return t / n * 1e9
end

function basecase_cost(::Type{T}, n) where {T}
    M = 2^20 ÷ n
    a = T <: AbstractFloat ? T.(randn(n * M)) : rand(T, n * M)
    buf = Q.scratch(T)
    f() = GC.@preserve a buf for j in 0:M-1
        Q.small_sort!(pointer(a) + j * n * sizeof(T), n, pointer(buf))
    end
    f()
    copyto!(a, T <: AbstractFloat ? T.(randn(n * M)) : rand(T, n * M))
    return @elapsed(f()) / M * 1e9
end

function main()
    Random.seed!(1)
    n = 2^26
    println("1. Share of samples per component, 2^26 keys, one thread")
    components("Int32", rand(Int32, n))
    components("Int64", rand(Int64, n))
    components("Float32 (randn)", randn(Float32, n))
    components("Float64 (randn)", randn(n))
    components("UInt64 in 0:1000", rand(UInt64(0):UInt64(1000), n))
    components("1000 wide values", rand(rand(UInt64, 1000), n))

    println("\n2. Partition, ns/key (no prefetch / prefetch)")
    for T in (Int32, Int64)
        cells = [@sprintf("2^%d: %.3f / %.3f", k, partition_cost(T, 2^k, false), partition_cost(T, 2^k, true))
                 for k in (10, 14, 18, 20, 22, 24, 26)]
        println(rpad(T, 8), join(cells, "  "))
    end

    println("\n3. Base case, ns per call")
    for T in (Int32, Float32, Int64, Float64)
        cells = [@sprintf("n=%d: %.0f", m, basecase_cost(T, m))
                 for m in (16, 32, 64, 128, 256) if m <= Q.basecase_max(T)]
        println(rpad(T, 8), join(cells, "  "))
    end
end

main()
