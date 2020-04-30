
function computeflowparticleapproxstate(λ_a::T,
                                        λ_b::T,
                                        x::Vector{T},
                                        problem_params,
                                        problem_methods,
                                        GF_buffers,
                                        GF_config)::Vector{T} where T <: Real
    # set up
    𝑚_a = GF_buffers.𝑚_a
    𝑃_a = GF_buffers.𝑃_a
    𝑚_b = GF_buffers.𝑚_b
    𝑃_b = GF_buffers.𝑃_b

    term1 = 𝑚_b
    #C = 𝑃_b*inv(𝑃_a)
    C = Utilities.forcesymmetric(𝑃_b*inv(𝑃_a))

    term2 = real.(LinearAlgebra.sqrt(C))*(x - 𝑚_a)

    return term1 + term2
end

# γ set to 0.
function computelnabsdetJofstateupdate(  λ_a::T,
                                        λ_b::T,
                                        problem_params,
                                         problem_methods,
                                         GF_buffers::GaussianFlowSimpleBuffersType{T},
                                         GF_config,
                                        x::Vector{T}) where T <: Real
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
    #Δϵ = ϵ_b - ϵ_a
    x_minus_𝑚_a = x - 𝑚_a
    #𝑃_b_inv𝑃_a = 𝑃_b*inv(𝑃_a)

    # other recurring factors.
    factor12 = real.(LinearAlgebra.sqrt(𝑃_b*inv(𝑃_a)))

    #exp_factor = sqrt( (one(T) - exp(-γ*Δλ))/Δλ )

    # first term.
    J = ∂𝑚_b_∂x + factor12*(LinearAlgebra.I - ∂𝑚_a_∂x)

    # the other terms.
    for i = 1:D_x
        for j = 1:D_x

            term3 = sum( ∂𝑃_b_inv𝑃_a_sqrt_∂x[j][i,k]*x_minus_𝑚_a[k] for k = 1:D_x )

            J[i,j] = J[i,j] + term3

        end
    end

    return logabsdet(J)[1]
end

function computeflowparticleapproxweight(λ_a::T,
                                        λ_b::T,
                                        x_a::Vector{T},
                                        x_b::Vector{T},
                                        problem_params,
                                        problem_methods,
                                        GF_buffers,
                                        GF_config,
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
                                    x_a)

    #
    # println("lnabsdetJ = ", lnabsdetJ)
    # @assert 1==2

    ln_w_b_unnorm = ln_w_a + numerator1 + numerator2 + lnabsdetJ - denominator1 - denominator2


    return ln_w_b_unnorm
end

# γ set to 0.
function traversesdepathapproxflow2( x0::Vector{T},
                                    λ_array::LinRange{T},
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config) where T <: Real

    # set up.
    N_steps = length(λ_array)
    #D = length(x0)
    #B_λ, drawfunc = setupdrawBrownia(D, one(T))

    # drawfunc(Δλ)

    ln_w0 = zero(T)

    # allocate.
    x = Vector{Vector{T}}(undef, N_steps)
    ln_w = Vector{T}(undef,N_steps)

    # to start.
    #ϵ_a = zeros(T, length(Bλ_array[1]))
    #ϵ_b = Bλ_array[1]
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
                                            GF_config)

    ln_w[1] = computeflowparticleapproxweight(  λ_a,
                                                λ_b,
                                                x0,
                                                x[1],
                                                problem_params,
                                                problem_methods,
                                                GF_buffers,
                                                GF_config,
                                                ln_w0)
    #
    # println("x[1] = ", x[1])
    # println("ln_w[1] = ", ln_w[1])
    # @assert 1==2

    for n = 2:length(x)
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
                                                GF_config)

        #
        ln_w[n] = computeflowparticleapproxweight(  λ_a,
                                                    λ_b,
                                                    x[n-1],
                                                    x[n],
                                                    problem_params,
                                                    problem_methods,
                                                    GF_buffers,
                                                    GF_config,
                                                    ln_w[n-1])
    end

    return x, ln_w
end

# γ set to zero.
function paralleltraverseSDEs(  drawxfunc::Function,
                                N_discretizations::Int,
                                m_0::Vector{T},
                                P_0::Matrix{T},
                                R,
                                y,
                                ψ::Function,
                                ∂ψ::Function,
                                ∂2ψ::Function,
                                ln_prior_pdf_func::Function,
                                ln_likelihood_func::Function,
                                M::Int,
                                N_batches::Int) where T <: Real

    # # work on intervals.
    # M_for_each_batch = Vector{Int}(undef, N_batches)
    # 𝑀::Int = round(Int, M/N_batches)
    # fill!(M_for_each_batch, 𝑀)
    # M_for_each_batch[end] = abs(M - (N_batches-1)*𝑀)
    #
    # @assert M == sum(M_for_each_batch) # sanity check.

    # takes 2.
    M_for_each_batch = Vector{Int}(undef, N_batches)

    fill!(M_for_each_batch, div(M,N_batches))

    N_batches_w_extra = mod(M,N_batches)
    for i = 1:N_batches_w_extra
        M_for_each_batch[i] += 1
    end
    @assert M == sum(M_for_each_batch) # sanity check.

    # set up neccessary objects.
    problem_params,
        problem_methods,
        GF_buffers,
        GF_config = setupGFquantities( zero(T),
                                m_0,
                                P_0,
                                R,
                                y,
                                ψ,
                                ∂ψ,
                                ∂2ψ,
                                ln_prior_pdf_func,
                                ln_likelihood_func)
    #
    λ_array = LinRange(1/(N_discretizations-1), one(T), N_discretizations)


    ##prepare worker function.
    workerfunc = xx->traverseSDEs2(λ_array, drawxfunc,
                                    N_discretizations,
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config,
                                    xx)

    # compute solution.
    sol = pmap(workerfunc, M_for_each_batch )

    # unpack solution.
    xp_array, ln_wp_array, x_array = unpackpmap(sol, M)

    ln_wp_array = collect( ln_wp_array[n][1] for n = 1:length(ln_wp_array))

    return xp_array, ln_wp_array, x_array
end

# simulates a separate Bλ_array and λ_array for each particle.
# Computes faster.
function traverseSDEs2(  λ_array, drawxfunc::Function,
                        N_discretizations::Int,
                        problem_params::GaussianFlowSimpleParamsType{T},
                        problem_methods::GaussianFlowMethodsType,
                        GF_buffers::GaussianFlowSimpleBuffersType{T},
                        GF_config::GaussianFlowConfigType{T},
                        N_particles::Int)::Tuple{Vector{Vector{T}},
                                                Vector{Vector{T}},
                                                Vector{Vector{T}}} where T <: Real


    # allocate outputs.
    x_array = Vector{Vector{T}}(undef, N_particles)
    xp_array = Vector{Vector{T}}(undef, N_particles)
    ln_wp_array = Vector{Vector{T}}(undef, N_particles)

    # traverse particles.
    for n = 1:N_particles
        x = drawxfunc(1.0)

        𝑥, 𝑤 = traversesdepathapproxflow2(   x,
                                         λ_array,
                                         problem_params,
                                         problem_methods,
                                         GF_buffers,
                                         GF_config)
        xp_array[n] = 𝑥[end]
        ln_wp_array[n] = [ 𝑤[end] ]
        x_array[n] = x
        #println("n = ", n)
    end

    return xp_array, ln_wp_array, x_array
end
