# routines related to moments, derivatives of moments.
# mode is a Symbol that specifies if L == 1 or larger.
# Equations taken from Bunch et al.'s Gaussian flow 2016 paper.

function computelinearization!( b::GFAKBuffersType{T},
                                ∂ψ!::Function,
                                ψ!::Function,
                                x::Vector{T},
                                y::Vector{T},
                                ::Val{:SCALAR}) where T <: Real
    #
    ∂ψ!b.𝐻, (x)
    ψ!(b.ψ_eval, x)

    ### 𝑦 = y - ψ(x) + ∂ψ(x)*x. Eqn 17. of Bunch et al.

    ## for L not equal to 1.
    # b.𝑦[:] = y - b.ψ_eval + b.𝐻*x

    ## for L == 1.
    b.𝑦[1] = y[1] - b.ψ_eval[1] + dot(b.𝐻, x)

    return nothing
end


# all matrices are assumed to be dense matrices.
# term1 is inv_P0*m0.
function updatemoments( b::GFAKBuffersType{T},
                        inv_σ²_persist::Vector{T},
                        inv_σ²_mul_z_persist::Vector{T},
                        λ::T,
                        ::Val{:SCALAR})::Tuple{Vector{T},Matrix{T}} where T <: Real

    # set up.
    𝐻 = b.𝐻
    𝑦 = b.𝑦

    inv_P0 = [inv_σ²_persist]
    inv_R = [inv_σ²_persist]

    ### update moments. Eqn 18 of Bunch et. al.

    ## when L > 1.
    # 𝑃 = inv(inv_P0 + λ*𝐻'*inv_R*𝐻)
    # 𝑃 = Utilities.forcesymmetric(𝑃)

    # 𝑚 = 𝑃*(inv_P0_mul_m0 + λ*𝐻'*inv_R*𝑦)

    ## when L == 1.
    𝑃 = inv(inv_P0 + λ*𝐻'*inv_R*𝐻)
    𝑃 = Utilities.forcesymmetric(𝑃)

    𝑚 = 𝑃*(inv_P0_mul_m0 + λ*𝐻'*inv_R*𝑦)

    return 𝑚, 𝑃
end
