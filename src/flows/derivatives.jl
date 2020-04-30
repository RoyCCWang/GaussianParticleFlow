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

    ∂𝑚_a_∂x = GF_buffers.∂𝑚_a_∂x
    ∂𝑚_b_∂x = GF_buffers.∂𝑚_b_∂x

    ∂𝑃_b_∂x = GF_buffers.∂𝑃_b_∂x
    ∂𝑃_b_inv𝑃_a_∂x = GF_buffers.∂𝑃_b_inv𝑃_a_∂x

    ∂𝑃_b_sqrt_∂x = GF_buffers.∂𝑃_b_sqrt_∂x
    ∂𝑃_b_inv𝑃_a_sqrt_∂x = GF_buffers.∂𝑃_b_inv𝑃_a_sqrt_∂x

    # set up.
    #∂𝐻t_∂x::Vector{Matrix{T}} = get∂𝐻tfunc(x_a)

    D_x = length(𝑚_a)
    Δλ = λ_b - λ_a
    Δϵ = ϵ_b - ϵ_a
    x_minus_𝑚_a = x - 𝑚_a
    #𝑃_b_inv𝑃_a = 𝑃_b*inv(𝑃_a)

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

    # b = GF_buffers
    # println("b.𝑚_a = ", b.𝑚_a)
    # println("b.𝑚_b = ", b.𝑚_b)
    # println("b.𝑃_a = ", b.𝑃_a)
    # println("b.𝑃_b = ", b.𝑃_b)
    #
    # println("b.∂𝑚_a_∂x = ", b.∂𝑚_a_∂x)
    # println("b.∂𝑚_b_∂x = ", b.∂𝑚_b_∂x)
    # println("b.∂𝑃_b_∂x = ", b.∂𝑃_b_∂x)
    # println("b.∂𝑃_b_inv𝑃_a_∂x = ", b.∂𝑃_b_inv𝑃_a_∂x)
    # println("b.∂𝑃_b_sqrt_∂x = ", b.∂𝑃_b_sqrt_∂x)
    # println("b.∂𝑃_b_inv𝑃_a_sqrt_∂x = ", b.∂𝑃_b_inv𝑃_a_sqrt_∂x)
    #
    # @assert 1==2

    # other recurring factors.
    exp_half_factor = exp(-0.5*γ*Δλ)
    factor12 = real.(LinearAlgebra.sqrt(𝑃_b*inv(𝑃_a)))

    sqrt_γ_factor = sqrt( (one(T) - exp(-γ*Δλ))/Δλ )

    # first term.
    J = ∂𝑚_b_∂x + exp_half_factor*factor12*(LinearAlgebra.I - ∂𝑚_a_∂x)

    # the other terms.
    for i = 1:D_x
        for j = 1:D_x

            #term2 = sum( ∂𝑃_b_sqrt_∂x[j][i,k]*Δϵ[k] for k = 1:D_x )
            term2 = sqrt_γ_factor*sum( ∂𝑃_b_sqrt_∂x[j][i,k]*Δϵ[k] for k = 1:D_x )

            tmp = sum( ∂𝑃_b_inv𝑃_a_sqrt_∂x[j][i,k]*x_minus_𝑚_a[k] for k = 1:D_x )
            term3 = exp_half_factor*tmp

            J[i,j] = J[i,j] + term2 + term3

        end
    end

    return logabsdet(J)[1]
end

# equation 28.
function computeGFderivatives!( p::GaussianFlowSimpleParamsType{T},
                        b::GaussianFlowSimpleBuffersType{T},
                        λ_a::T,
                        λ_b::T,
                        x::Vector{T}) where T <: Real
# I am here.
    #
    #D_x = size(𝐻,2)
    #@assert length(x) == D_x == length(∂𝐻t_∂x)
    D_x = length(x)
    D_y = length(p.y)

    # parse.
    𝐻 = b.𝐻
    ∂2ψ_eval = b.∂2ψ_eval

    𝑦 = b.𝑦

    𝑚_a = b.𝑚_a
    𝑃_a = b.𝑃_a
    𝑚_b = b.𝑚_b
    𝑃_b = b.𝑃_b

    # ∂𝑚_a_∂x = b.∂𝑚_a_∂x
    # ∂𝑚_b_∂x = b.∂𝑚_b_∂x
    #
    # ∂𝑃_b_∂x = b.∂𝑃_b_∂x
    # ∂𝑃_b_inv𝑃_a_∂x = b.∂𝑃_b_inv𝑃_a_∂x
    #
    # ∂𝑃_b_sqrt_∂x = b.∂𝑃_b_sqrt_∂x
    # ∂𝑃_b_inv𝑃_a_sqrt_∂x = b.∂𝑃_b_inv𝑃_a_sqrt_∂x

    # checks.
    @assert size(b.∂𝑚_a_∂x) == (D_x, D_x)
    @assert size(b.∂𝑚_b_∂x) == (D_x, D_x)

    @assert length(b.∂𝑃_b_∂x) == length(b.∂𝑃_b_inv𝑃_a_∂x) == length(b.∂𝑃_b_sqrt_∂x) == length(b.∂𝑃_b_inv𝑃_a_sqrt_∂x) == D_x
    #∂𝑃binv𝑃a_∂x = Vector{Matrix{T}}(undef,D_x)
    #∂𝑃_∂x = Vector{Matrix{T}}(undef,D_x)

    for j = 1:D_x
        # get second derivative matrix.
        ∂𝐻t_∂xj = getHtmatrix(j, ∂2ψ_eval, D_x)

        # println("∂𝐻t_∂xj = ", ∂𝐻t_∂xj)
        # @assert size(∂𝐻t_∂xj) == (D_x, D_y)
        #
        # @assert 1==2

        ∂𝐻_∂xj = ∂𝐻t_∂xj'


        ## equation 28.
        term1a = ∂𝐻t_∂xj*(R\(𝑦 - 𝐻*𝑚_a))
        term2a = 𝐻'*(R\(∂𝐻_∂xj*(x - 𝑚_a)))

        b.∂𝑚_a_∂x[:,j] = λ_a*𝑃_a*(term1a + term2a)

        #
        term1b = ∂𝐻t_∂xj*(R\(𝑦 - 𝐻*𝑚_b))
        term2b = 𝐻'*(R\(∂𝐻_∂xj*(x - 𝑚_b)))

        b.∂𝑚_b_∂x[:,j] = λ_b*𝑃_b*(term1b + term2b)

        #
        b.∂𝑃_b_∂x[j] = -λ_b*𝑃_b*( ∂𝐻t_∂xj*(R\𝐻) + 𝐻'*(R\∂𝐻_∂xj) )*𝑃_b

        #
        factor1 = (λ_a*LinearAlgebra.I - λ_b*𝑃_b*inv(𝑃_a))
        b.∂𝑃_b_inv𝑃_a_∂x[j] = 𝑃_b*( ∂𝐻t_∂xj*(R\𝐻) + 𝐻'*(R\∂𝐻_∂xj) )*factor1
    end

    return nothing
end
