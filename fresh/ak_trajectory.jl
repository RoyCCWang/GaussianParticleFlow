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


### adaptive kernel.
@everywhere include("./adaptive_kernel/scalar.jl")
@everywhere include("./adaptive_kernel/traverse.jl")

@everywhere include("../tests/routines/L1_moments.jl")
@everywhere include("../tests/routines/Bunch_updates.jl")

## TODO I am here. go through routines to ensure as fast as possible
#    implementation of GFlow, with options to cache Brownian motion, or
#    draw on demand.

# how can transport help with likelihoods?

PyPlot.close("all")
fig_num = 1

Random.seed!(25)
#Random.seed!(435)
#Random.seed!(244432)


N_batches = 16



#
max_integral_evals = typemax(Int) #1000000
initial_div = 1000



D_x = 6
θ_a = rand()

A_ϕ = randn(D_x, D_x)
ϕ = xx->sinc(dot(xx,A_ϕ*xx))
dϕ = ϕ
d2ϕ = ϕ

# hardcode set up for L = 1, for now.
σ² = rand()
σ²_vec = [ σ² ]
R = ones(Float64, 1, 1)
R[1] = σ²

P_0 = diagm( σ² .* ones(D_x) )
m_0 = randn(D_x) # this is z_input.
z_input = m_0

ψ = xx->[ ϕ(xx) ]
y = randn(1)

prior_dist = Distributions.MvNormal(m_0, P_0)

ln_prior_pdf_func = xx->Distributions.logpdf(prior_dist, xx)
ln_likelihood_func = xx->evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R)

inv_P0 = diagm(1/σ² * ones(Float64, D_x))
inv_R = diagm(1/σ² * ones(Float64, L))


###### verify moment expressions. pakage into test script when done.
L = 1

x0 = randn(D_x)
λ0 = rand()
λ1 = λ0 + rand()

ψ_scalar = xx->ψ(xx)[1]
y_scalar = y[1]

### verify state update.

x1_func_Bunch = xx->computestateupdateBunch(xx, λ0, λ1, ψ, z_input, inv_P0, inv_R, y, 0.0)
x1_func = xx->computestateupdate(xx, λ0, λ1, ψ_scalar,
                                z_input, σ², y_scalar)

x1_Bunch = x1_func_Bunch(x0)
x1 = x1_func(x0)

dx1_func_AD = xx->ForwardDiff.jacobian(x1_func, xx)
dx1_func = xx->computedx1L1(xx, λ0, λ1, ψ_scalar, z_input, σ², y_scalar)

dx1_x0_AD = dx1_func_AD(x0)
dx1_x0 = dx1_func(x0)

println("verify 𝑚 formula.")
println("x1, L = 1: l-2 discrepancy between my formula and Bunch is  ", norm(x1_Bunch-x1))
println("dx1, L = 1: l-2 discrepancy between my formula and Bunch is  ", norm(dx1_x0_AD-dx1_x0))
println()

@assert 1==222

### flow.

## set up SDE.
#N_discretizations = 1000
N_discretizations = 1000
γ = 0.0
#N_particles = 20

# x = rand(prior_dist)
#
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
# λ_array, Bλ_array = drawBrownianmotiontrajectorieswithoutstart(N_discretizations, D_x);
# #
# #
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

L = 1
z = m_0

# function evaladaptiveGaussiankernel(x::Vector{T2},
#                                     z::Vector{T},
#                                     a::T,
#                                     ϕ,
#                                     ϕ_z::T = ϕ(z))::T2 where {T,T2}
#     #
#     @assert length(x) == length(z)
#
#     term1::T2 = zero(T2)
#     for d = 1:length(x)
#         term1 += (x[d]-z[d])^2
#     end
#
#     term2::T2 = (ϕ(x)-ϕ(z))^2
#
#     out = exp(-a*(term1+term2))
#
#     return out
# end

# I am here.


problem_params,
    problem_methods,
    GF_buffers,
    GF_config = setupGFAKscalar( γ,
                            z,
                            θ_a,
                            ϕ,
                            dϕ,
                            d2ϕ,
                            L)
@assert 1333==3
@time 𝑥2, 𝑤2 = traverseGFAK(  x,
                                λ_array,
                                problem_params,
                                problem_methods,
                                GF_buffers,
                                GF_config)
#
## compare.
discrepancy_x = norm(𝑥2-𝑥)
discrepancy_w = norm(𝑤2-𝑤)

println("discrepancy_x = ", discrepancy_x)
println("discrepancy_w = ", discrepancy_w)
println()
