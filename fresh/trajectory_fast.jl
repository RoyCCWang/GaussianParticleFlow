# test exact Gaussian flow.


using Distributed

@everywhere import HCubature
@everywhere import Utilities
@everywhere import Printf
@everywhere import PyPlot
@everywhere import Random

@everywhere using LinearAlgebra
@everywhere import Interpolations

@everywhere using FFTW

@everywhere import Statistics

@everywhere import Distributions

@everywhere import Calculus

@everywhere import ForwardDiff
@everywhere import StatsFuns

using Test
using BenchmarkTools

import VisualizationTools

#@everywhere import Seaborn
@everywhere include("../src/misc/declarations.jl")

@everywhere include("../src/SDE/Brownian.jl")
@everywhere include("../src/flows/approx_flow.jl")
@everywhere include("../tests/routines/sde.jl")
@everywhere include("../src/flows/moments.jl")
@everywhere include("../src/flows/derivatives.jl")
@everywhere include("../src/misc/utilities.jl")
@everywhere include("../src/misc/utilities2.jl")

@everywhere include("../tests/routines/simulation_tools.jl")
@everywhere include("../src/flows/exact_flow.jl")

@everywhere include("../src/importance_sampler/IS_engine.jl")
@everywhere include("../src/diagnostics/functionals.jl")


@everywhere include("../src/flows/traverse_with_weights.jl")
@everywhere include("../src/flows/traverse_no_weights.jl")

## TODO I am here. go through routines to ensure as fast as possible
#    implementation of GFlow, with options to cache Brownian motion, or
#    draw on demand.

# how can transport help with likelihoods?

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

N_batches = 16



#
max_integral_evals = typemax(Int) #1000000
initial_div = 1000

#demo_string = "mixture"
#demo_string = "normal"

# D_x = 2
# D_y = 3
# ψ = exampleψfunc2Dto3D1

D_x = 2
D_y = 2
ψ = exampleψfunc2Dto2D1

# D_x = 4
# D_y = 2
# ψ = exampleψfunc4Dto2D1

limit_a = ones(Float64, D_x) .* -10.9
limit_b = ones(Float64, D_x) .* 10.9

# oracle latent variable, x.
x_generating = [1.22; -0.35]

# observation model aside from ψ.

σ = 0.02
R = diagm( 0 => collect(σ for d = 1:D_y) )

# generate observation.
true_dist_y = Distributions.MvNormal(ψ(x_generating),R)
y = rand(true_dist_y)

# prior.
m_0 = randn(Float64, D_x)
P_0 = Utilities.generaterandomposdefmat(D_x)

prior_dist = Distributions.MvNormal(m_0, P_0)


prior_func = xx->exp(Distributions.logpdf(prior_dist, xx))

# function of x!
likelihood_func = xx->exp(evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R))


∂ψ = xx->Calculus.jacobian(ψ, xx, :central)
#∂ψ = xx->ForwardDiff.jacobian(ψ, xx)
D_y = length(y)
∂2ψ = xx->eval∂2ψND(xx, ψ, D_y)

ln_prior_pdf_func = xx->Distributions.logpdf(prior_dist, xx)
ln_likelihood_func = xx->evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R)

### flow.

## set up SDE.
#N_discretizations = 1000
N_discretizations = 1000
γ = 0.0
#N_particles = 20

x = rand(prior_dist)

# println("old: traversesdepathapproxflow")
# Random.seed!(25)
# problem_params,
#     problem_methods,
#     GF_buffers,
#     GF_config = setupGFquantities( γ,
#                             m_0,
#                             P_0,
#                             R,
#                             y,
#                             ψ,
#                             ∂ψ,
#                             ∂2ψ,
#                             ln_prior_pdf_func,
#                             ln_likelihood_func)
#
λ_array, Bλ_array = drawBrownianmotiontrajectorieswithoutstart(N_discretizations, D_x);
#
#
# @time 𝑥, 𝑤 = traversesdepathapproxflow(   x,
#                                  λ_array,
#                                  Bλ_array,
#                                  problem_params,
#                                  problem_methods,
#                                  GF_buffers,
#                                  GF_config)

#
println("new: traversesdepathapproxflow")
Random.seed!(25)

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

𝑥2, 𝑤2 = traversesdepathapproxflow2( x,
                                    λ_array,
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config)
#
# ## compare.
# discrepancy_x = norm(𝑥2-𝑥)
# discrepancy_w = norm(𝑤2-𝑤)
#
# println("discrepancy_x = ", discrepancy_x)
# println("discrepancy_w = ", discrepancy_w)
# println()

println("traversegflow")
𝑥3, 𝑚, 𝑃 = traversegflow( x,
                                    λ_array,
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config)

## compare.
discrepancy_x = norm(𝑥3-𝑥2)

println("2 and 3: discrepancy_x = ", discrepancy_x)
println()

println("Diagnostics for this simulation:")
println("intiial x:", x)
println("final x:  ", 𝑥3[end])
println("𝑚 =       ", 𝑚)
println("𝑃 = ")
display(𝑃)
println()


println("traversegflow")
x = rand(prior_dist)
𝑥3, 𝑚, 𝑃 = traversegflow( x,
                                    λ_array,
                                    problem_params,
                                    problem_methods,
                                    GF_buffers,
                                    GF_config)


println("Diagnostics for this simulation:")
println("intiial x:", x)
println("final x:  ", 𝑥3[end])
println("𝑚 =       ", 𝑚)
println("𝑃 = ")
display(𝑃)
println()
