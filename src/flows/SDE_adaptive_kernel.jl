


# original
# function runGFISonadaptivekernel(z::Vector{T},
#                             ϕ::Function,
#                             dϕ::Function,
#                             d2ϕ::Function,
#                             N_discretizations::Int,
#                             γ::T,
#                             N_particles::Int,
#                             N_batches::Int,
#                             a_SqExp::T) where T <: Real
#
#     # set up prior.
#     D_x = length(z)
#     σ = 1/sqrt(2*a_SqExp)
#
#     drawxfunc = xx->Utilities.drawnormal(z, σ)
#     m_0 = z
#     P_0 = Matrix{T}(LinearAlgebra.I, D_x, D_x) .* σ^2
#
#     # set up observation.
#     y = [ ϕ(z) ]
#
#     # set up likelihood.
#     R = Matrix{T}(undef,1,1)
#     R[1,1] = σ^2 # single warpmap, single dimension for each y.
#
#     ψ = xx->[ ϕ(xx) ]
#
#     # set up SDE.
#     λ_array, Bλ_array = drawBrownianmotiontrajectorieswithoutstart(N_discretizations, D_x)
#
#     get𝐻func = xx->convertcolvectorowvec(dϕ(xx))
#     get∂𝐻tfunc = xx->convertmatrixtonestedcolmats(d2ϕ(xx))
#
#     ln_prior_pdf_func = xx->evallnMVN(xx, m_0, P_0)
#     ln_likelihood_func = xx->evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R)
#
#
#     ## traverse SDE.
#     xp_array, ln_wp_array, x_array = paralleltraverseSDEs(drawxfunc,
#                                 λ_array,
#                                 Bλ_array,
#                                 γ,
#                                 m_0,
#                                 P_0,
#                                 ψ,
#                                 R,
#                                 y,
#                                 get𝐻func,
#                                 get∂𝐻tfunc,
#                                 ln_prior_pdf_func,
#                                 ln_likelihood_func,
#                                 N_particles,
#                                 N_batches)
#
#     ## post-processing.
#     # normalize weights.
#
#     return xp_array, ln_wp_array, x_array
# end


"""
    runGFISonadaptivekernel

Traverse N particles. density is the one associated with
the single-warp map adaptive kernel that is centered at z.
"""
function runGFISonadaptivekernel(z::Vector{T},
                            ϕ::Function,
                            dϕ::Function,
                            d2ϕ::Function,
                            N_discretizations::Int,
                            γ::T,
                            N_particles::Int,
                            N_batches::Int,
                            a_SqExp::T) where T <: Real
    # set up prior.
    D_x = length(z)
    σ, P_0, R, ψ, get𝐻func, get∂𝐻tfunc, λ_array,
        Bλ_array = setupGFISadaptivekernel(D_x, ϕ, dϕ, d2ϕ,
                        N_discretizations, a_SqExp)
    # set up quantities that vary with z.
    y = [ ϕ(z) ]
    drawxfunc = xx->Utilities.drawnormal(z, σ)
    m_0 = z
    ln_prior_pdf_func = xx->evallnMVN(xx, m_0, P_0)
    ln_likelihood_func = xx->evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R)

    return runGFISonadaptivekernel(z,
                                    ϕ,
                                    γ,
                                    N_particles,
                                    N_batches,
                                    σ,
                                    P_0,
                                    R,
                                    ψ,
                                    get𝐻func,
                                    get∂𝐻tfunc,
                                    N_discretizations)
end


"""
    drawfromquerydensity

Draw N_samples Gaussian flow importance samples from the query distribution.
c is the set of RKHS coefficients. Each must be positive.
X is the set of training positions.
θ_a is the kernel.
"""
function runGFISonquery( X::Vector{Vector{T}},
                                c::Vector{T},
                                ϕ::Function,
                                dϕ::Function,
                                d2ϕ::Function,
                                N_discretizations::Int,
                                γ::T,
                                N_samples::Int,
                                max_N_batches::Int,
                                a_SqExp::T) where T <: Real
    #
    D_x = length(X[1])

    # set up quantities that don't depend on the chosen
    #   kernel mixture.
    σ, P_0, R, ψ, get𝐻func, get∂𝐻tfunc, λ_array,
        Bλ_array = setupGFISadaptivekernel(  D_x,
                                            ϕ,
                                            dϕ,
                                            d2ϕ,
                                            N_discretizations,
                                            a_SqExp)

    # draw from mixture.
    𝑐 = convert(Vector{BigFloat}, c)
    w = 𝑐 ./ sum(𝑐)
    𝑖 = GenericMCMCSamplers.drawcategorical(w, N_samples)

    # set up z's.
    𝑖_unique = unique(𝑖)
    M = length(𝑖_unique)

    z_array = X[𝑖_unique]
    N_z_array = collect( count( 𝑖 .== 𝑖_unique[m] ) for m = 1:M )

    # allocate output.
    xp_array = Vector{Vector{T}}(undef, N_samples)
    ln_wp_array = Vector{T}(undef, N_samples)
    x_array = Vector{Vector{T}}(undef, N_samples)

    # traverse SDE.
    st = 0
    fin = 0
    for m = 1:M

        z = z_array[m]
        N_particles = N_z_array[m]

        #println("On m = ", m, ", N_particles = ", N_particles, ", z = ", z)
        #N_batches::Int = max( round(Int, N_particles/N_processes), 1 )
        N_batches::Int = max_N_batches
        if N_particles < N_batches
            N_batches = 1
        end

        st = fin +1
        fin = st + N_particles -1

        xp_array[st:fin], ln_wp_array[st:fin],
            x_array[st:fin] = runGFISonadaptivekernel(  z,
                                                ϕ,
                                                γ,
                                                N_particles,
                                                N_batches,
                                                σ,
                                                P_0,
                                                R,
                                                ψ,
                                                get𝐻func,
                                                get∂𝐻tfunc,
                                                N_discretizations)
    end

    return xp_array, ln_wp_array, x_array
end
