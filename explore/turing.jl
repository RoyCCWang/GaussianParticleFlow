# demonstrative example.

# https://stanjulia.github.io/Stan.jl/stable/INSTALLATION.html

using Distributed

using LinearAlgebra
import Random
import Distributions

import Utilities

import Turing
import StatsPlots
import DynamicPPL

import DataFrames

import HCubature

include("../tests/routines/simulation_tools.jl")
include("../src/diagnostics/functionals.jl")
include("../src/misc/utilities.jl")

Random.seed!(25)



### model parameters.

D_x = 2
D_y = 2
ψ = exampleψfunc2Dto2D1


### generate data.

# oracle distribution.
# m_true = randn(Float64, D_x)
# P_true = randn(Float64, D_x, D_x)
# P_true = P_true'*P_true
#
# true_dist_x = Distributions.MvNormal(m_true, P_true)


# draw a realization for the latent variable.
#x_generating = rand(true_dist_x)
x_generating = [1.22; -0.35]

# use it to generate data.
σ = 0.02
R = diagm( 0 => collect(σ for d = 1:D_y) )

# prior.
m_0 = randn(Float64, D_x)
P_0 = Utilities.generaterandomposdefmat(D_x)

prior_dist = Distributions.MvNormal(m_0, P_0)

#x0 = x_generating


# generate observation.
true_dist_y = Distributions.MvNormal(ψ(x_generating),R)
y = rand(true_dist_y)

### MCMC.


# Define a simple Normal model with unknown mean and variance.
Turing.@model gfdemo(y, m0, P0, ψ, Σ) = begin
  x ~ Turing.MvNormal(m0, P0)
  y ~ Turing.MvNormal(ψ(x), Σ)
end

#  Run sampler, collect results
x0 = zeros(Float64,D_x)
chain_length = 10000
chn = Turing.sample(gfdemo(y, m_0, P_0, ψ, R), Turing.NUTS(0.65), chain_length)

K = DataFrames.DataFrame(chn)

x_MCMC_d_n = collect( K[!,d] for d = 1:D_x )
x_MCMC = collect( collect( K[n,d] for d = 1:size(K,2) ) for n = 1:size(K,1) )

# Summarise results (currently requires the master branch from MCMCChains)
Turing.describe(chn)

# Plot and save results
#p = StatsPlots.plot(chn)
#savefig("gdemo-plot.png")


prior_func = xx->exp(Distributions.logpdf(prior_dist, xx))

# function of x!
likelihood_func = xx->exp(evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R))


B = [   0.219055  0.290879;
        0.290879  0.398671]
g = xx->exp(-dot(xx,B,xx))
#g_eval_GF = evalexpectation(g, xp_array, w_array)
g_eval_MCMC = evalexpectation(g, x_MCMC)

println("Evaluating g, NI.")
# limit_a = [-80.0; -80.0]
# limit_b = [80.0; 80.0]

limit_a = [minimum(x_MCMC_d_n[1]); minimum(x_MCMC_d_n[2])]
limit_b = [maximum(x_MCMC_d_n[1]); maximum(x_MCMC_d_n[2])]

max_integral_evals = typemax(Int)
initial_div = 5
@time g_eval_NI, val_h, err_h, val_Z, err_Z = evalexpectation(g,
                                        likelihood_func,
                                        prior_func,
                limit_a, limit_b, max_integral_evals, initial_div)

println("NI: 𝔼[g] over posterior   = ", g_eval_NI)
println("val_h = ", val_h)
println("MCMC: 𝔼[g] over posterior = ", g_eval_MCMC)
#println("GF: 𝔼[g] over posterior   = ", g_eval_GF)
println()

# use Mathematica to do integration over entire number line.
# https://mathematica.stackexchange.com/questions/78000/multidimensional-numeric-integration
