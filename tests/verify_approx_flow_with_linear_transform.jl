# test exact Gaussian flow.



import FileIO
import Printf
import PyPlot
import Random

using LinearAlgebra
import Interpolations

using FFTW

import Statistics

import Distributions
import Utilities

import ForwardDiff
import StatsFuns

import AdaptiveRKHS



include("../src/SDE/Brownian.jl")
include("../src/flows/approx_flow.jl")
include("routines/sde.jl")
include("../src/flows/moments.jl")
include("../src/misc/utilities.jl")

include("routines/simulation_tools.jl")
include("../src/flows/exact_flow.jl")

include("../src/importance_sampler/IS_engine.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)


#demo_string = "mixture"
#demo_string = "normal"

D_x = 2
D_y = 3
N_batches = 10

# true distribution for latent variable, x.
m_true = randn(Float64,D_x)
P_true = randn(Float64,D_x,D_x)
P_true = P_true'*P_true

true_dist_x = Distributions.MvNormal(m_true, P_true)

# generate a value of the latent variable.
x_generating = rand(true_dist_x)

# observation model.
#ψ = exampleψfunc2Dto3D1

## debug.
A_true = randn(D_y, D_x)
ψ = xx->A_true*xx
## end debug.

σ = 0.02
R = diagm( 0 => collect(σ for d = 1:D_y) )



# generate observation.
true_dist_y = Distributions.MvNormal(ψ(x_generating),R)
y = rand(true_dist_y)

# prior.
m_0 = randn(Float64,D_x)
P_0 = randn(Float64,D_x,D_x)
P_0 = P_0'*P_0

prior_dist = Distributions.MvNormal(m_0, P_0)

## debug.
# true posterior.
m_p, Σ_p = getexactmoments(1.0, m_0, P_0, A_true, R, y)
## end debug.



# approx. flow required functions.
get𝐻func = xx->ForwardDiff.jacobian(ψ, xx)

hessian_funcs = gethessianfuncs(ψ, D_y)
get∂𝐻tfunc = xx->compute∂𝐻tover∂x(hessian_funcs, xx, D_y)

ln_prior_pdf_func = xx->Distributions.logpdf(prior_dist, xx)
#ln_likelihood_func = xx->Distributions.logpdf(Distributions.MvNormal(ψ(xx), R), y)
ln_likelihood_func = xx->evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R)

# D_y = length(y)
# get𝐻func2 = xx->Calculus.jacobian(ψ, xx, :central)
# hessian_funcs = gethessianfuncsND(ψ, D_y)
# get∂𝐻tfunc2 = xx->compute∂𝐻tover∂x(hessian_funcs, xx, D_y)
#
# get𝐻func = xx->convertcolvectorowvec(d_ϕ_map_itp(xx))
# get∂𝐻tfunc = xx->convertmatrixtonestedcolmats(d2_ϕ_map_itp(xx))

# x0 = randn(D)
#
# @time a = get𝐻func(x0)
# @time b = get𝐻func2(x0)
#
# @time c = get∂𝐻tfunc(x0)
# @time d = get∂𝐻tfunc2(x0)
#
# println("a = ", a)
# println("b = ", b)
# println("c = ", c)
# println("d = ", d)
# println()
#
# @assert 1==2

### debug.
# x = randn(D_x)
# # u1 = get𝐻func(x)
# # u2 = get∂𝐻tfunc(x)
# # v1 = ln_likelihood_func(x)
# # v2 = ln_prior_pdf_func(x)
#
# λ = rand()
#
# 𝐻 = get𝐻func(x)
# 𝑦 = y - ψ(x) + 𝐻*x
# m1, P1 = updatemoments( inv(P_0), inv(P_0)*m_0, 𝐻, inv(R), 𝑦, λ)
# m2, P2 = getexactmoments(λ, m_0, P_0, get𝐻func(x), R, y )
# println("discrepancy: exact moments and approx moment updates: m,P:  ", norm(m1-m2) + norm(P1-P2))
#
# ϵ_a = Bλ_array[34]
# ϵ_b = Bλ_array[35]
# λ_a = λ_array[34]
# λ_b = λ_array[35]
#
# inv_P0_mul_m0 = P_0\m_0
# inv_R = inv(R)
# inv_P0 = inv(P_0)
#
# x1_app = computeflowparticleapproxstate(    λ_a,
#                                             λ_b,
#                                             x,
#                                             inv_P0_mul_m0,
#                                             inv_P0,
#                                             inv_R,
#                                             γ,
#                                             get𝐻func,
#                                             ψ,
#                                             y,
#                                             ϵ_a,
#                                             ϵ_b)
# #
# x1_exa = computeflowparticlestateeqn15( x,
#                                         λ_a, λ_b,
#                                         m_0, P_0, 𝐻, R, y,
#                                         ϵ_a, ϵ_b, γ)
# #
#
# # ϵ_a = zeros(T, length(Bλ_array[1]))
# # ϵ_b = Bλ_array[1]
# # λ_a = zero(T)
# # λ_b = λ_array[1]
# #
# # ln_w_app = computeflowparticleapproxweight(  λ_a,
# #                                             λ_b,
# #                                             x0,
# #                                             x,
# #                                             γ,
# #                                             get𝐻func,
# #                                             get∂𝐻tfunc,
# #                                             ψ,
# #                                             y,
# #                                             ϵ_a,
# #                                             ϵ_b,
# #                                             inv_P0_mul_m0,
# #                                             inv_P0,
# #                                             R,
# #                                             ln_prior_pdf_func,
# #                                             ln_likelihood_func,
# #                                             ln_w0)
#
# #@assert 1==2
### end debug.





### flow.

## set up SDE.
N_discretizations = 1000
γ = 0.1
N_particles = 100

# set up Brownian motion.
λ_array, Bλ_array = drawBrownianmotiontrajectorieswithoutstart(N_discretizations, D_x)

### traverse the SDE for each particle. (sequential version.)
# draw samples from the prior.


# traverse the SDE solution for each particle.
x_array = collect( rand(prior_dist) for i = 1:N_particles )
xp_array = Vector{Vector{Float64}}(undef,N_particles)
ln_wp_array = Vector{Float64}(undef,N_particles)
for n = 1:N_particles

    𝑥, 𝑤 = traversesdepathapproxflow(   x_array[n],
                                        λ_array,
                                        Bλ_array,
                                        γ,
                                        m_0,
                                        P_0,
                                        ψ,
                                        R,
                                        y,
                                        get𝐻func,
                                        get∂𝐻tfunc,
                                        ln_prior_pdf_func,
                                        ln_likelihood_func)
    xp_array[n] = 𝑥[end]
    ln_wp_array[n] = 𝑤[end]
    Printf.@printf("done particle %d\n", n)
end

# ### parallel version.
# drawxfunc = xx->rand(prior_dist)
# @time xp_array, ln_wp_array, x_array = paralleltraverseSDEs(drawxfunc,
#                             N_discretizations,
#                             γ,
#                             m_0,
#                             P_0,
#                             ψ,
#                             R,
#                             y,
#                             get𝐻func,
#                             get∂𝐻tfunc,
#                             ln_prior_pdf_func,
#                             ln_likelihood_func,
#                             N_particles,
#                             N_batches)

# normalize weights.
ln_W = StatsFuns.logsumexp(ln_wp_array)
w_array = collect( exp(ln_wp_array[n] - ln_W) for n = 1:N_particles )


##### using equation 15. Use the same prior samples as the approx. flow.

# traverse the SDE solution for each particle.
xp_array2 = Vector{Vector{Float64}}(undef,N_particles)
for n = 1:N_particles
    x0 = x_array[n]

    𝑥 = traversesdepathexactflowviaeqn15(x0, λ_array, Bλ_array, γ, m_0, P_0, A_true, R, y)
    xp_array2[n] = 𝑥[end]
end

# sample covmat:
xp_array_weighted = xp_array.*w_array
m_s = sum(xp_array_weighted)
Q = getcovmatfromparticles(xp_array, m_s, w_array)


m_s2 = Statistics.mean(xp_array2)
Q2 = 1/(N_particles-1) * sum( (xp_array2[n]-m_s2)*(xp_array2[n]-m_s2)' for n = 1:N_particles)

println("x, generating = ", x_generating)
println()

println("ground truth posterior parameters:")
println("m_p = ", m_p)
println("Σ_p = ", Σ_p)
println()

println("approx: MLE posterior parameters:")
println("m_s = ", m_s)
println("Q = ", Q)
println()


println("Equation 15: MLE posterior parameters:")
println("m_s = ", m_s2)
println("Q = ", Q2)
println()

discrepancy = sum( norm(xp_array[n]-xp_array2[n]) for n = 1:N_particles )
println("discrepancy: x's posterior samples between approx. flow and exact flow: ", discrepancy)
