####### for weight integration.

# equation 27.
function computelnabsdetJofstateupdate(  λ_a::T,
                                        λ_b::T,
                                        problem_params,
                                         problem_methods,
                                         GF_buffers::GaussianFlowSimpleBuffersType{T},
                                         GF_config,
                                        x::Vector{T},
                                        ϵ_a::Vector{T},
                                        ϵ_b::Vector{T}) where T <: Real
    # parse.
    𝐻 = GF_buffers.𝐻
    𝑃_a = GF_buffers.𝑃_a
    𝑃_b = GF_buffers.𝑃_b
    𝑚_a = GF_buffers.𝑚_a
    𝑚_b = GF_buffers.𝑚_b
    𝑦 = GF_buffers.𝑦

    R = problem_params.R
    γ = GF_config.γ

    # set up.
    ∂𝐻t_∂x::Vector{Matrix{T}} = get∂𝐻tfunc(x_a)

    D_x = length(∂𝐻t_∂x)
    Δλ = λ_b - λ_a
    Δϵ = ϵ_b - ϵ_a
    x_minus_𝑚_a = x - 𝑚_a
    𝑃_b_inv𝑃_a = 𝑃_b*inv(𝑃_a)

    # # prepare derivatives.
    # ∂𝑚_a_∂x = compute∂𝑚wrt∂x(∂𝐻t_∂x, λ_a, R, 𝐻, 𝑃_a, 𝑚_a, x, 𝑦)
    # ∂𝑚_b_∂x = compute∂𝑚wrt∂x(∂𝐻t_∂x, λ_b, R, 𝐻, 𝑃_b, 𝑚_b, x, 𝑦)
    #
    # #∂𝑃_a_∂x = compute∂𝑃wrt∂x(∂𝐻t_∂x, λ_a, R, 𝐻, 𝑃_a)
    # ∂𝑃_b_∂x = compute∂𝑃wrt∂x(∂𝐻t_∂x, λ_b, R, 𝐻, 𝑃_b)
    # ∂𝑃_b_inv𝑃_a_∂x = compute∂𝑃binv𝑃awrt∂x(∂𝐻t_∂x, λ_a, λ_b, R, 𝐻, 𝑃_a, 𝑃_b)
    #
    # #∂𝑃_a_sqrt_∂x = computemsqrtderivatives(𝑃_a, ∂𝑃_a_∂x)
    # ∂𝑃_b_sqrt_∂x = computemsqrtderivatives(𝑃_b, ∂𝑃_b_∂x)
    # ∂𝑃_b_inv𝑃_a_sqrt_∂x = computemsqrtderivatives(𝑃_b_inv𝑃_a, ∂𝑃_b_inv𝑃_a_∂x)

    # other recurring factors.
    exp_half_factor = exp(-0.5*γ*Δλ)
    factor12 = real.(LinearAlgebra.sqrt(𝑃_b*inv(𝑃_a)))

    exp_factor = sqrt( (one(T) - exp(-γ*Δλ))/Δλ )

    # first term.
    J = ∂𝑚_b_∂x + exp_half_factor*factor12*(LinearAlgebra.I - ∂𝑚_a_∂x)

    # the other terms.
    for i = 1:D_x
        for j = 1:D_x

            term2 = sum( ∂𝑃_b_sqrt_∂x[j][i,k]*Δϵ[k] for k = 1:D_x )

            tmp = sum( ∂𝑃_b_inv𝑃_a_sqrt_∂x[j][i,k]*x_minus_𝑚_a[k] for k = 1:D_x )
            term3 = exp_half_factor*tmp

            J[i,j] = J[i,j] + term2 + term3

        end
    end

    return logabsdet(J)[1]
end

# equation 28.
function compute∂𝑚wrt∂x(∂𝐻t_∂x::Vector{Matrix{T}},
                        λ::T,
                        R::Matrix{T},
                        𝐻::Matrix{T},
                        𝑃::Matrix{T},
                        𝑚::Vector{T},
                        x::Vector{T},
                        𝑦::Vector{T})::Matrix{T} where T <: Real

    #
    D_x = size(𝐻,2)
    @assert length(x) == D_x == length(∂𝐻t_∂x)

    ∂𝑚_∂x = Matrix{T}(undef,D_x,D_x)
    for j = 1:D_x
        ∂𝐻t_∂xj = ∂𝐻t_∂x[j]
        ∂𝐻_∂xj = ∂𝐻t_∂xj'

        term1 = ∂𝐻t_∂xj*(R\(𝑦 - 𝐻*𝑚))
        term2 = 𝐻'*(R\(∂𝐻_∂xj*(x - 𝑚)))

        ∂𝑚_∂x[:,j] = λ*𝑃*(term1 + term2)
    end

    return ∂𝑚_∂x
end

# equation 28.
function compute∂𝑃wrt∂x(   ∂𝐻t_∂x::Vector{Matrix{T}},
                            λ::T,
                            R::Matrix{T},
                            𝐻::Matrix{T},
                            𝑃::Matrix{T})::Vector{Matrix{T}} where T <: Real

    #
    D_x = size(𝐻,2)
    @assert length(∂𝐻t_∂x) == D_x

    ∂𝑃_∂x = Vector{Matrix{T}}(undef,D_x)
    for j = 1:D_x
        ∂𝐻t_∂xj = ∂𝐻t_∂x[j]
        ∂𝐻_∂xj = ∂𝐻t_∂xj'

        ∂𝑃_∂x[j] = -λ*𝑃*( ∂𝐻t_∂xj*(R\𝐻) + 𝐻'*(R\∂𝐻_∂xj) )*𝑃
    end

    return ∂𝑃_∂x
end

# equation 28.
function compute∂𝑃binv𝑃awrt∂x(∂𝐻t_∂x::Vector{Matrix{T}},
                            λ_a::T,
                            λ_b::T,
                            R::Matrix{T},
                            𝐻::Matrix{T},
                            𝑃_a::Matrix{T},
                            𝑃_b)::Vector{Matrix{T}} where T <: Real

    #
    D_x = size(𝐻,2)
    @assert length(∂𝐻t_∂x) == D_x

    ∂𝑃binv𝑃a_∂x = Vector{Matrix{T}}(undef,D_x)
    for j = 1:D_x
        ∂𝐻t_∂xj = ∂𝐻t_∂x[j]
        ∂𝐻_∂xj = ∂𝐻t_∂xj'

        factor1 = (λ_a*LinearAlgebra.I - λ_b*𝑃_b*inv(𝑃_a))
        ∂𝑃binv𝑃a_∂x[j] = 𝑃_b*( ∂𝐻t_∂xj*(R\𝐻) + 𝐻'*(R\∂𝐻_∂xj) )*factor1
    end

    return ∂𝑃binv𝑃a_∂x
end
