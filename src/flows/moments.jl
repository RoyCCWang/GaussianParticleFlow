### scripts that relate to the moment updates.
# mit script fonts indicate variables that have a hat in the paper.



# all matrices are assumed to be dense matrices.
# term1 is inv_P0*m0.
function updatemoments( b::GaussianFlowSimpleBuffersType{T},
                        inv_P0::Matrix{T},
                        inv_P0_mul_m0::Vector{T},
                        inv_R::Matrix{T},
                        λ::T)::Tuple{Vector{T},Matrix{T}} where T <: Real

    # set up.
    𝐻 = b.𝐻
    𝑦 = b.𝑦

    # update moments.
    𝑃 = inv(inv_P0 + λ*𝐻'*inv_R*𝐻)
    𝑃 = Utilities.forcesymmetric(𝑃)

    𝑚 = 𝑃*(inv_P0_mul_m0 + λ*𝐻'*inv_R*𝑦)

    return 𝑚, 𝑃
end

function updatemoments( b::GaussianFlowBlockDiagonalBuffersType{T},
                        inv_P0::Matrix{T},
                        inv_P0_mul_m0::Vector{T},
                        inv_R::Vector{Matrix{T}},
                        λ::T)::Tuple{Vector{T},Matrix{T}} where T <: Real

    # set up.
    𝐻 = b.𝐻
    𝑦 = b.𝑦

    # 𝑃 = inv(inv_P0 + λ*𝐻'*inv_R*𝐻)
    A = applyrectmatrixcongruence(b.𝐻, inv_R)
    𝑃 = inv(inv_P0 + λ*A)
    𝑃 = Utilities.forcesymmetric(𝑃)

    #𝑚 = 𝑃*(inv_P0_mul_m0 + λ*𝐻'*inv_R*𝑦)
    c = applyHtRy(b.𝐻, inv_R, b.𝑦)
    𝑚 = 𝑃*(inv_P0_mul_m0 + λ*c)

    return 𝑚, 𝑃
end

function computelinearization!( b::GaussianFlowSimpleBuffersType{T},
                                ∂ψ::Function,
                                ψ::Function,
                                x::Vector{T},
                                y::Vector{T}) where T <: Real
    #
    b.𝐻[:] = ∂ψ(x)
    b.ψ_eval[:] = ψ(x)

    # 𝑦 = y - ψ(x) + ∂ψ(x)*x
    b.𝑦[:] = y - b.ψ_eval + b.𝐻*x

    return nothing
end

"""

y[n][m], n∈[N], m∈[M].
𝑦[m][n], n∈[N], m∈[M].

"""
function computelinearization!( b::GaussianFlowBlockDiagonalBuffersType{T},
                                ∂ψ::Function,
                                ψ::Function,
                                x::Vector{T},
                                y_set::Vector{Vector{T}}) where T <: Real


    #
    ∂ψ_∂θ::Vector{Vector{Vector{T}}} = ∂ψ(x)
    b.∂ψ_eval[:] = ∂ψ_∂θ

    𝐻_set = ∂ψtoblockdiagmatrix(∂ψ_∂θ) # I am here. change this to vector of matrices.
    b.𝐻_set[:] = 𝐻_set

    term3 = evalmatrixvectormultiply(𝐻_set, x)

    ψ_x::Vector{Vector{T}} = ψ(x)
    b.ψ_eval[:] = ψ_x

    # 𝑦 = y - ψ(x) + 𝐻*x
    𝑦_set = collect( y_set[m] - ψ_x[m] + term3[m] for m = 1:M )
    b.𝑦_set[:] = 𝑦_set

    return nothing
end
