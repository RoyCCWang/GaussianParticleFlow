
function unpackpmap(sol::Array{Tuple{ Array{Array{T,1},1},
                                      Array{Array{T,1},1},
                                      Array{Array{T,1},1}},1},
                    M::Int)::Tuple{Vector{Vector{T}},Vector{Vector{T}},Vector{Vector{T}}} where T <: Real

    N_batches = length(sol)

    x_array = Vector{Vector{T}}(undef,M)
    xp_array = Vector{Vector{T}}(undef,M)
    ln_wp_array =  Vector{Vector{T}}(undef,M)

    st::Int = 0
    fin::Int = 0
    for j = 1:N_batches

        st = fin + 1
        fin = st + length(sol[j][1]) - 1

        xp_array[st:fin] = sol[j][1]
        ln_wp_array[st:fin] = sol[j][2]
        x_array[st:fin] = sol[j][3]
    end

    return xp_array, ln_wp_array, x_array
end

function setupGFquantities( γ::T,
                            m_0::Vector{T},
                            P_0::Matrix{T},
                            R::Matrix{T},
                            y::Vector{T},
                            ψ::Function,
                            ∂ψ::Function,
                            ∂2ψ::Function,
                            ln_prior_pdf_func::Function,
                            ln_likelihood_func::Function) where T
    #
    p = GaussianFlowSimpleParamsType(m_0 = m_0,
                        P_0 = P_0,
                        R = R,
                        y = y,

                        inv_R = inv(R),
                        inv_P0_mul_m0 = P_0\m_0,
                        inv_P0 = inv(P_0))
    #
    m = GaussianFlowMethodsType( ψ = ψ,
                        ∂ψ = ∂ψ,
                        ∂2ψ = ∂2ψ,
                        ln_prior_pdf_func = ln_prior_pdf_func,
                        ln_likelihood_func = ln_likelihood_func)

    #
    b = GaussianFlowSimpleBuffersType(length(y), length(m_0), y[1])

    config = GaussianFlowConfigType(γ)

    return p, m, b, config
end

# simulates a separate Bλ_array and λ_array for each particle.
# Computes faster.
function traverseSDEs(  drawxfunc::Function,
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

        # This will get better sample diversity.
        λ_array, Bλ_array = drawBrownianmotiontrajectorieswithoutstart(N_discretizations, length(x))

        𝑥, 𝑤 = traversesdepathapproxflow(   x,
                                         λ_array,
                                         Bλ_array,
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



"""
    bar(x[, y])

Compute the Bar index between `x` and `y`. If `y` is missing, compute
the Bar index between all pairs of columns of `x`.

# Examples
```julia-repl
julia> bar([1, 2], [1, 2])
1
```
"""
function paralleltraverseSDEs(  drawxfunc::Function,
                                N_discretizations::Int,
                                γ::T,
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

    ##prepare worker function.
    workerfunc = xx->traverseSDEs(drawxfunc,
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
