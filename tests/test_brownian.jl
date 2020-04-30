
import Random
using LinearAlgebra

include("../src/misc/declarations.jl")
include("../src/SDE/Brownian.jl")


N = 50
D = 1

Random.seed!(25)
λ_array, Bλ_batch = drawBrownianmotiontrajectorieswithoutstart(N, D)

Random.seed!(25)
Bλ_sequential, drawfunc = setupdrawBrownia(D, 1.0)

function storeB(λ_array, Bλ_sequential::Vector{T}, drawfunc) where T
    N = length(λ_array)

    out = Vector{Vector{T}}(undef, N)

    drawfunc(λ_array[1])
    out[1] = copy(Bλ_sequential)

    for n = 2:N
        drawfunc(λ_array[n]-λ_array[n-1])
        out[n] = copy(Bλ_sequential)
    end

    return out
end

collect_Bλ = storeB(λ_array, Bλ_sequential, drawfunc)
norm(Bλ_batch-collect_Bλ)
