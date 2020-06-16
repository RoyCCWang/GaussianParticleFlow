

# γ set to 0.
function traverseadaptivekernel( x0::Vector{T},
                                    λ_array::LinRange{T},
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config) where T <: Real

    # set up.
    N_steps = length(λ_array)
    ln_w0 = zero(T)

    # allocate.
    x = Vector{Vector{T}}(undef, N_steps) # debug. really just need two.
    ln_w = Vector{T}(undef, N_steps) # debug. really just need two.

    # to start.
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
