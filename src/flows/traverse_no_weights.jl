



##### no weights.

# γ set to 0, for now.
# no weights.
# returns final m_λ and P_λ.
function traversegflow( x0::Vector{T},
                                    λ_array::LinRange{T},
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config) where T <: Real

    # set up.
    N_steps = length(λ_array)

    # allocate.
    x = Vector{Vector{T}}(undef, N_steps)

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

    end

    return x, GF_buffers.𝑚_b, GF_buffers.𝑃_b
end


function paralleltraverseSDEs3(  drawxfunc::Function,
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
        GF_config = setupGFquantities( γ,
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
    workerfunc = xx->traverseSDEs3(λ_array, drawxfunc,
                                    N_discretizations,
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config,
                                    xx)

    # compute solution.
    sol = pmap(workerfunc, M_for_each_batch )

    # unpack solution.
    xp_array, x_array = unpackpmapnoweights(sol, M)

    #ln_wp_array = collect( ln_wp_array[n][1] for n = 1:length(ln_wp_array))

    #return xp_array, ln_wp_array, x_array
    return xp_array, x_array
end

function traverseSDEs3(  λ_array::LinRange{T},
                        drawxfunc::Function,
                        N_discretizations::Int,
                        problem_params::GaussianFlowSimpleParamsType{T},
                        problem_methods::GaussianFlowMethodsType,
                        GF_buffers::GaussianFlowSimpleBuffersType{T},
                        GF_config::GaussianFlowConfigType{T},
                        N_particles::Int)::Tuple{Vector{Vector{T}},
                                                Vector{Vector{T}}} where T <: Real


    # allocate outputs.
    x_array = Vector{Vector{T}}(undef, N_particles)
    xp_array = Vector{Vector{T}}(undef, N_particles)
    #ln_wp_array = Vector{Vector{T}}(undef, N_particles)

    # traverse particles.
    for n = 1:N_particles
        x = drawxfunc(1.0)

        𝑥 = traversegflow(   x,
                                         λ_array,
                                         problem_params,
                                         problem_methods,
                                         GF_buffers,
                                         GF_config)
        xp_array[n] = 𝑥[end]
        #ln_wp_array[n] = [ 𝑤[end] ]
        #ln_wp_array[n] = [ 1.0 ] # dummy.
        x_array[n] = x
        #println("n = ", n)
    end

    #return xp_array, ln_wp_array, x_array
    return xp_array, x_array
end

function unpackpmapnoweights(sol::Array{Tuple{ Array{Array{T,1},1},
                                      Array{Array{T,1},1}},1},
                    M::Int)::Tuple{Vector{Vector{T}},Vector{Vector{T}}} where T <: Real

    N_batches = length(sol)

    x_array = Vector{Vector{T}}(undef,M)
    xp_array = Vector{Vector{T}}(undef,M)

    st::Int = 0
    fin::Int = 0
    for j = 1:N_batches

        st = fin + 1
        fin = st + length(sol[j][1]) - 1

        xp_array[st:fin] = sol[j][1]
        x_array[st:fin] = sol[j][2]
    end

    return xp_array, x_array
end
