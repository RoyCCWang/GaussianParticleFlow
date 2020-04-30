### designed for generic (i.e., no special structure) latent variable and observation.


function traversesdepathapproxflow( x0::Vector{T},
                                    λ_array::LinRange{T},
                                    Bλ_array::Vector{Vector{T}},
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config) where T

    # set up.
    N_steps = length(λ_array)
    @assert length(Bλ_array) == N_steps

    ln_w0 = zero(T)

    # allocate.
    x = Vector{Vector{T}}(undef, N_steps)
    ln_w = Vector{T}(undef,N_steps)

    # to start.
    ϵ_a = zeros(T, length(Bλ_array[1]))
    ϵ_b = Bλ_array[1]
    λ_a = zero(T)
    λ_b = λ_array[1]

    # update.
    updateGFbuffers!(  problem_params,
                            problem_methods,
                            GF_buffers,
                            GF_config,
                            λ_a,
                            λ_b,
                            x0)

    x[1] = computeflowparticleapproxstate(  λ_a,
                                            λ_b,
                                            x0,
                                            problem_params,
                                            problem_methods,
                                            GF_buffers,
                                            GF_config,
                                            ϵ_a,
                                            ϵ_b)

    ln_w[1] = computeflowparticleapproxweight(  λ_a,
                                                λ_b,
                                                x0,
                                                x[1],
                                                problem_params,
                                                problem_methods,
                                                GF_buffers,
                                                GF_config,
                                                ϵ_a,
                                                ϵ_b,
                                                ln_w0)
    #
    # println("x[1] = ", x[1])
    # println("ln_w[1] = ", ln_w[1])
    # @assert 1==2

    for n = 2:length(x)
        ϵ_a = Bλ_array[n-1]
        ϵ_b = Bλ_array[n]
        λ_a = λ_array[n-1]
        λ_b = λ_array[n]

        #
        updateGFbuffers!(  problem_params,
                                problem_methods,
                                GF_buffers,
                                GF_config,
                                λ_a,
                                λ_b,
                                x[n-1])

        x[n] = computeflowparticleapproxstate(  λ_a,
                                                λ_b,
                                                x[n-1],
                                                problem_params,
                                                problem_methods,
                                                GF_buffers,
                                                GF_config,
                                                ϵ_a,
                                                ϵ_b)

        #
        ln_w[n] = computeflowparticleapproxweight(  λ_a,
                                                    λ_b,
                                                    x[n-1],
                                                    x[n],
                                                    problem_params,
                                                    problem_methods,
                                                    GF_buffers,
                                                    GF_config,
                                                    ϵ_a,
                                                    ϵ_b,
                                                    ln_w[n-1])
    end

    return x, ln_w
end


# equation 25.
# x is the state at λ_a.
function computeflowparticleapproxstate(λ_a::T,
                                        λ_b::T,
                                        x::Vector{T},
                                        problem_params,
                                        problem_methods,
                                        GF_buffers,
                                        GF_config,
                                        ϵ_a::Vector{T},
                                        ϵ_b::Vector{T}) where T <: Real
    # set up
    γ = GF_config.γ
    inv_P0_mul_m0 = problem_params.inv_P0_mul_m0
    inv_P0 = problem_params.inv_P0
    inv_R = problem_params.inv_R
    y = problem_params.y

    ψ = problem_methods.ψ
    ∂ψ = problem_methods.∂ψ

    𝐻 = GF_buffers.𝐻
    𝑦 = GF_buffers.𝑦

    𝑚_a = GF_buffers.𝑚_a
    𝑃_a = GF_buffers.𝑃_a
    𝑚_b = GF_buffers.𝑚_b
    𝑃_b = GF_buffers.𝑃_b


    # #
    # 𝐻, 𝑦 = computelinearization!(∂ψ, ψ, x, y)
    # # get udpated moments at x_a.
    # 𝑚_a, 𝑃_a = updatemoments(inv_P0, inv_P0_mul_m0, 𝐻, inv_R, 𝑦, λ_a)
    # 𝑚_b, 𝑃_b = updatemoments(inv_P0, inv_P0_mul_m0, 𝐻, inv_R, 𝑦, λ_b)

    term1 = 𝑚_b
    #C = 𝑃_b*inv(𝑃_a)
    C = Utilities.forcesymmetric(𝑃_b*inv(𝑃_a))

    term2 = exp(-0.5*γ*(λ_b-λ_a))*real.(LinearAlgebra.sqrt(C))*(x - 𝑚_a)

    numerator = one(T) - exp(-γ*(λ_b-λ_a))
    denominator = λ_b - λ_a
    # println("weight1 = ", exp(-γ*(λ_b-λ_a)))
    # println(" weight2 = ", sqrt(numerator/denominator))
    # @assert 1==22
    term3 = sqrt(numerator/denominator) .* Utilities.naivesqrtpsdmatrix(𝑃_b) *(ϵ_b - ϵ_a)

    return term1 + term2 + term3
end


# equation 25.
# x is the state at λ_a.
function computeflowparticleapproxweight(λ_a::T,
                                        λ_b::T,
                                        x_a::Vector{T},
                                        x_b::Vector{T},
                                        problem_params,
                                        problem_methods,
                                        GF_buffers,
                                        GF_config,
                                        ϵ_a::Vector{T},
                                        ϵ_b::Vector{T},
                                        ln_w_a::T) where T <: Real
    #
    # set up
    γ = GF_config.γ
    inv_P0_mul_m0 = problem_params.inv_P0_mul_m0
    inv_P0 = problem_params.inv_P0
    inv_R = problem_params.inv_R
    y = problem_params.y

    ψ = problem_methods.ψ
    ∂ψ = problem_methods.∂ψ
    ∂2ψ = problem_methods.∂2ψ
    ln_prior_pdf_func = problem_methods.ln_prior_pdf_func
    ln_likelihood_func = problem_methods.ln_likelihood_func

    # linearize at x and compute moments.
    # 𝐻, 𝑦 = computelinearization( ∂ψ, ψ, x_a, y)
    # 𝑚_a, 𝑃_a = updatemoments(inv_P0, inv_P0_mul_m0, 𝐻, inv_R, 𝑦, λ_a)
    # 𝑚_b, 𝑃_b = updatemoments(inv_P0, inv_P0_mul_m0, 𝐻, inv_R, 𝑦, λ_b)

    # b = GF_buffers
    #
    # println("b.𝐻 = ", b.𝐻)
    # println("b.𝑦 = ", b.𝑦)
    # println("b.𝑚_a = ", b.𝑚_a)
    # println("b.𝑃_a = ", b.𝑃_a)
    # println("b.𝑚_b = ", b.𝑚_b)
    # println("b.𝑃_b = ", b.𝑃_b)
    # @assert 1==2

    #
    numerator1 = ln_prior_pdf_func(x_b)
    numerator2 = ln_likelihood_func(x_b)*λ_b

    denominator1  = ln_prior_pdf_func(x_a)
    denominator2 = ln_likelihood_func(x_a)*λ_a

    lnabsdetJ = computelnabsdetJofstateupdate( λ_a,
                                    λ_b,
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config,
                                    x_a,
                                    ϵ_a,
                                    ϵ_b)

    #
    # println("lnabsdetJ = ", lnabsdetJ)
    # @assert 1==2

    ln_w_b_unnorm = ln_w_a + numerator1 + numerator2 + lnabsdetJ - denominator1 - denominator2

    # println("ln_w_a = ", ln_w_a)
    # println("numerator1 = ", numerator1)
    # println("numerator2 = ", numerator2)
    # println("lnabsdetJ = ", lnabsdetJ)
    # println("denominator1 = ", denominator1)
    # println("denominator2 = ", denominator2)
    # println("λ_a = ", λ_a)
    # println("λ_b = ", λ_b)
    # println("ln_w_b_unnorm = ", ln_w_b_unnorm)
    # println()

    return ln_w_b_unnorm
end


"""
"""
function updateGFbuffers!( p::GaussianFlowParamsType,
                           m::GaussianFlowMethodsType,
                           b::GaussianFlowBuffersType,
                           config::GaussianFlowConfigType,
                           λ_a,
                           λ_b,
                           x::Vector{T}) where T


    # linearize at x_a.
    computelinearization!(b, m.∂ψ, m.ψ, x, p.y)

    # get udpated moments at x_a.
    b.𝑚_a[:], b.𝑃_a[:] = updatemoments(b, p.inv_P0,
                                p.inv_P0_mul_m0,
                                p.inv_R, λ_a)
    b.𝑚_b[:], b.𝑃_b[:] = updatemoments(b, p.inv_P0,
                                p.inv_P0_mul_m0,
                                p.inv_R, λ_b)

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
