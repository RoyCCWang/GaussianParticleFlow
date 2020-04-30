function drawsamplinglocationsuniform(  N::Int,
                                        a::Vector{T},
                                        b::Vector{T})::Vector{Vector{T}} where T <: Real
    D = length(b)
    @assert length(a) == D

    X = Vector{Vector{T}}(undef,N)

    for n = 1:N
        X[n] = Vector{T}(undef,D)
        for d = 1:D
            X[n][d] = Utilities.convertcompactdomain(rand(), zero(T), one(T), a[d], b[d])
        end
    end

    return X
end

# uniform distribution over [a[d],b[d]]^{d = 1:D}.
function runISuniformproposal(   N::Int,
                                            a::Vector{T},
                                            b::Vector{T},
                                            p_tilde::Function)::Tuple{Vector{T},Vector{Vector{T}}} where T <: Real
    D = length(b)
    @assert length(a) == D

    println("Drawing from uniform proposal.")
    X = drawsamplinglocationsuniform(N, a, b)

    # q(x) doesn't depend on x, since we've uniform distribution as the proposal.
    q_x = one(T)/prod( b[d]-a[d] for d = 1:D )

    println("Computing IS weights for ", length(X), " pts.")
    w = computeISweightsuniformproposal(X, p_tilde, q_x)
    println("Done.")

    return w, X
end

function computeISweightsuniformproposal(X::Vector{Vector{T}},
                                        p_tilde::Function,
                                        q_x::T) where T <: Real
    N = length(X)

    w = Vector{T}(undef,N)
    for n = 1:N
        #println("n = ", n)
        w[n] = p_tilde(X[n])/q_x
    end

    sum_w = sum(w)
    for n = 1:N
        w[n] = w[n]/sum_w
    end

    return w
end
