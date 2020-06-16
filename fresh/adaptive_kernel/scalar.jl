# Routines for using Gaussian Flow to sample an adaptive Gaussian kernel.
# Use the scalar observation variable formulation.

# evaluates the ln pdf of an isotropic normal that has common variance across dimensions.
# σ²_vec is a single element array that stores the common variance.
function evallnproductnormal(x::Vector{T}, m::Vector{T}, σ²_vec::Vector{T})::T where T <: Real
    @assert length(x) == length(m)
    D = length(x)

    σ² = σ²_vec[1]

    running_sum = zero(T)
    for d = 1:D
        running_sum += (x[d]-m[d])^2
    end
    running_sum = -running_sum/(2*σ²)

    term1 = -D/2*log(2*π)
    term2 = -D*log(σ²)/2

    return term1 + term2 + running_sum
end


function evallnproductnormallikelihood( x::Vector{T},
                                        y::Vector{T},
                                        ψ::Function,
                                        σ²::Vector{T})::T where T <: Real

    #
    #term2 = evallnMVN(y, ψ(x), S_y)
    #function evallnMVN(x, μ::Vector{T}, Σ::Matrix{T})::T where T <: Real

    return evallnproductnormal(y, ψ(x), σ²)
end

function setuponetomanymap( x::Vector{T},
                            ϕ::Function,
                            L::Int) where T <: Real

    #
    ϕ_x = ϕ(x)

    out = Vector{T}(undef, L)
    fill!(out, ϕ_x)

    return out
end

function setupGFAKscalar( γ::T,
                                    z::Vector{T},
                                    a::T,
                                    ϕ::Function,
                                    dϕ::Function,
                                    d2ϕ::Function,
                                    L::Int = 1) where T <: Real
    #
    D_x = length(z)

    ψ = xx->setuponetomanymap(xx, ϕ, L)

    σ²_persist = Vector{T}(undef, 1)
    z_persist = Vector{T}(undef, D_x)
    y_persist = Vector{T}(undef, L)
    updateGFAKpersists!(σ²_persist,
                                    z_persist,
                                    y_persist,
                                    a,
                                    z,
                                    ψ(z))

    #
    p = GFAKParamsType(
                        z_persist = z_persist,
                        σ²_persist = σ²_persist,
                        y = y_persist,

                        inv_σ²_persist = [ one(T)/σ²_persist[1] ],
                        inv_σ²_mul_z_persist = z_persist ./ σ²_persist[1])
    #

    ln_prior_pdf_func = xx->evallnproductnormal(xx, z_persist, σ²_persist)
    ln_likelihood_func = xx->evallnproductnormallikelihood(xx, y_persist, ψ, σ²_persist)

    m = GaussianFlowMutatingMethodsType( ψ = ψ,
                        ∂ψ = dϕ,
                        ∂2ψ = d2ϕ,
                        ln_prior_pdf_func = ln_prior_pdf_func,
                        ln_likelihood_func = ln_likelihood_func)

    #
    b = GFAKBuffersType(L, D_x, y[1])

    config = GaussianFlowConfigType(γ)

    return p, m, b, config
end


function updateGFAKpersists!(   σ²_persist::Vector{T},
                                z_persist::Vector{T},
                                y_persist::Vector{T},
                                a::T,
                                z::Vector{T},
                                y::Vector{T}) where T <: Real
    #
    L = length(y)

    σ²_persist[1] = L/(2*a)

    resize!(z_persist, length(z))
    for d = 1:length(z)
        z_persist[d] = z[d]
    end

    resize!(y_persist, L)
    for d = 1:L
        y_persist[d] = y[d]
    end

    return nothing
end


function GFAKBuffersType(D_y::Int, D_x::Int, val::T) where T <: Real

    return GFAKBuffersType(
        ψ_eval = Vector{T}(undef, D_y),

        𝐻 = Matrix{T}(undef, D_y, D_x),
        ∂2ψ_eval = Vector{Vector{T}}(undef, D_y),

        𝑦 = Vector{T}(undef, D_y),

        # moments.
        𝑚_a = Vector{T}(undef, D_x),
        𝑃_a = Matrix{T}(undef, D_x, D_x),
        𝑚_b = Vector{T}(undef, D_x),
        𝑃_b = Matrix{T}(undef, D_x, D_x),

        # derivatives.
        ∂𝑚_a_∂x = Matrix{T}(undef, D_x, D_x),
        ∂𝑚_b_∂x = Matrix{T}(undef, D_x, D_x),

        ∂𝑃_b_∂x = Vector{Matrix{T}}(undef, D_x),
        ∂𝑃_b_inv𝑃_a_∂x = Vector{Matrix{T}}(undef, D_x),

        ∂𝑃_b_sqrt_∂x = Vector{Matrix{T}}(undef, D_x),
        ∂𝑃_b_inv𝑃_a_sqrt_∂x = Vector{Matrix{T}}(undef, D_x) )

end

"""
Evaluates inv(A + u*u') when A is of the form a*Id, where a > 0.
"""
function evalrank1inverseupdatediagm()

end


function updateGFbuffers!( p::GFAKParamsType,
                           m::GaussianFlowMutatingMethodsType,
                           b::GFAKBuffersType,
                           config::GFAKConfigType,
                           λ_a,
                           λ_b,
                           x::Vector{T}) where T


    # linearize at x_a.
    computelinearization!(b, m.∂ψ!, m.ψ!, x, p.y, config.mode)

    # get udpated moments at x_a.
    b.𝑚_a[:], b.𝑃_a[:] = updatemoments(b, p.inv_P0,
                                p.inv_P0_mul_m0,
                                p.inv_R, λ_a, config.mode)
    b.𝑚_b[:], b.𝑃_b[:] = updatemoments(b, p.inv_P0,
                                p.inv_P0_mul_m0,
                                p.inv_R, λ_b, config.mode)

    #### prepare derivative sfor Jacobian.
    # ∂𝐻t_∂x::Vector{Matrix{T}} = get∂𝐻tfunc(x_a)
    #
    # ∂𝑚_a_∂x = compute∂𝑚wrt∂x(∂𝐻t_∂x, λ_a, R, 𝐻, 𝑃_a, 𝑚_a, x, 𝑦)
    # ∂𝑚_b_∂x = compute∂𝑚wrt∂x(∂𝐻t_∂x, λ_b, R, 𝐻, 𝑃_b, 𝑚_b, x, 𝑦)
    #
    # ∂𝑃_b_∂x = compute∂𝑃wrt∂x(∂𝐻t_∂x, λ_b, R, 𝐻, 𝑃_b)
    # ∂𝑃_b_inv𝑃_a_∂x = compute∂𝑃binv𝑃awrt∂x(∂𝐻t_∂x, λ_a, λ_b, R, 𝐻, 𝑃_a, 𝑃_b)

    b.∂2ψ_eval[:] = m.∂2ψ(x)

    computeGFderivatives!(p, b, λ_a, λ_b, x)
    b.∂𝑃_b_sqrt_∂x[:] = computemsqrtderivatives(b.𝑃_b, b.∂𝑃_b_∂x)

    𝑃_b_inv𝑃_a = b.𝑃_b*inv(b.𝑃_a)
    b.∂𝑃_b_inv𝑃_a_sqrt_∂x[:] = computemsqrtderivatives(𝑃_b_inv𝑃_a,
                                                b.∂𝑃_b_inv𝑃_a_∂x)

    return nothing
end
