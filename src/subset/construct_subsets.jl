# routines related to the construction of subsets of kernel centers.

# with merging of nearby positions.
function prepareallkernelcenters2(N_IS::Int,
                                limit_a::Vector{T},
                                limit_b::Vector{T},
                                f::Function,
                                N_𝑖::Int,
                                ϵ::T) where T <: Real


    w_IS, X_IS = runISuniformproposal(N_IS, limit_a, limit_b, f)

    # draw from the large pool of importance samples.
    𝑖 = GenericMCMCSamplers.drawcategorical(w_IS, N_𝑖)
    𝑖 = unique(𝑖)
    X = mergeclosepositions(X_IS[𝑖], ϵ)

    return X
end

function prepareallkernelcenters(N_IS::Int,
                                limit_a::Vector{T},
                                limit_b::Vector{T},
                                f::Function) where T <: Real


    w_IS, X_IS = runISuniformproposal(N_IS, limit_a, limit_b, f)

    return w_IS, X_IS
end

# uses categorical sampling, with w being the PMF.
# w_pool cannot have zero-valued entries.
# N_𝑖 is the maximum subset size.
function drawpartitionofkernelcenters( X_input::Vector{Vector{T}},
                            w_input::Vector{T},
                            N_𝑖::Int ) where T <: Real

    # initialize the pool of kernel centers.
    X_pool = copy(X_input)
    w_pool = copy(w_input)

    # get all subset sizes for each subset.
    M_array = batcharraygivenbatchsize( length(X_pool), N_𝑖 )
    N_subsets = length(M_array)

    # allocate output.
    X_array = Vector{Vector{Vector{T}}}(undef, N_subsets)

    for i = 1:N_subsets
        X_array[i] = Vector{Vector{T}}(undef, M_array[i])

        for m = 1:M_array[i]

            # make w_pool into a PMF.
            w_pool = w_pool ./ sum(w_pool)

            # draw from the pool of positions, X_pool.
            𝑖 = GenericMCMCSamplers.drawcategorical(w_pool)

            # store.
            X_array[i][m] = X_pool[𝑖]

            # remove drawn entry from pool.
            deleteat!(X_pool, 𝑖)
            deleteat!(w_pool, 𝑖)
        end
    end

    return X_array
end


# function checkproximity( X::Vector{Vector{T}},
#                         x::Vector{T},
#                         ϵ::T)::Bool where T <: Real
#     #
#     indicators = collect( norm(X[j]-x) > ϵ for j = 1:length(X) )
#     indicators[n] = true
#
#     return all(indicators)
# end

function mergepostprocessingofsets(X_array::Vector{Vector{Vector{T}}},
                                    ϵ::T) where T <: Real
    #
    out = copy(X_array)
    for i = 1:length(X_array)
        out[i] = mergeclosepositions(X_array[i], ϵ)
    end

    return out
end
