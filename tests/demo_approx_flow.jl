# test exact Gaussian flow.

using Distributed

@everywhere import HCubature
@everywhere import Utilities
@everywhere import FileIO
@everywhere import Printf
@everywhere import PyPlot
@everywhere import Random

@everywhere using LinearAlgebra
@everywhere import Interpolations

@everywhere using FFTW

@everywhere import Statistics

@everywhere import Distributions
@everywhere import Utilities

@everywhere import Calculus

@everywhere import ForwardDiff
@everywhere import StatsFuns

@everywhere using Mamba

#@everywhere import Seaborn

@everywhere include("../src/misc/declarations.jl")
@everywhere include("../src/misc/utilities2.jl")

@everywhere include("../src/SDE/Brownian.jl")
@everywhere include("../src/flows/approx_flow.jl")
@everywhere include("routines/sde.jl")
@everywhere include("../src/flows/moments.jl")
@everywhere include("../src/misc/utilities.jl")

@everywhere include("routines/simulation_tools.jl")
@everywhere include("../src/flows/exact_flow.jl")

@everywhere include("../src/importance_sampler/IS_engine.jl")
@everywhere include("../src/diagnostics/functionals.jl")


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

limit_a = ones(Float64, D_x) .* -999.9
limit_b = ones(Float64, D_x) .* 999.9

# true distribution for latent variable, x.
m_true = randn(Float64,D_x)
P_true = randn(Float64,D_x,D_x)
P_true = P_true'*P_true

true_dist_x = Distributions.MvNormal(m_true, P_true)

# generate a value of the latent variable.
x_generating = rand(true_dist_x)

# observation model aside from ψ.

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

x0 = x_generating

prior_func = xx->exp(Distributions.logpdf(prior_dist, xx))

# function of x!
likelihood_func = xx->exp(evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R))
#w = xx->evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R)

# f = xx->sinc(norm(xx)^2)
# f_eval_NI, val_h, err_h, val_Z, err_Z = evalexpectation(f,
#                                         likelihood_func,
#                                         prior_func,
#                 limit_a, limit_b, max_integral_evals, initial_div)
#
# println("NI: 𝔼[f] over posterior   = ", f_eval_NI)
# println("val_h = ", val_h)
#
# @assert 1==2


# approx. flow required functions.
#get𝐻func = xx->ForwardDiff.jacobian(ψ, xx)
#hessian_funcs = gethessianfuncs(ψ, D_y)

get𝐻func = xx->Calculus.jacobian(ψ, xx, :central)
hessian_funcs = gethessianfuncsND(ψ, D_y)
get∂𝐻tfunc = xx->compute∂𝐻tover∂x(hessian_funcs, xx, D_y)

ln_prior_pdf_func = xx->Distributions.logpdf(prior_dist, xx)
ln_likelihood_func = xx->evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R)



### MCMC.

model = Model(

  y = Stochastic(1,
    (mu) ->  MvNormal(mu, R),
    false
  ),

  mu = Logical(1,
    x->ψ(x),
    false
  ),

  x = Stochastic(1,
    () -> MvNormal(m_0, P_0)
  )

)

scheme1 = [NUTS(:x)]
#
setsamplers!(model, scheme1)

line = Dict{Symbol, Any}(
  :y => y
)

N_chains = 3
inits = [
  Dict{Symbol, Any}(
    :y => line[:y],
    :x => zeros(Float64,length(m_0))
  )
for i in 1:N_chains
]

setsamplers!(model, scheme1)
println("Starting MCMC simulation.")
@time sim1 = mcmc(model, line, inits, 100000, burnin=250, thin=2, chains=N_chains)

describe(sim1)

x_MCMC = packageupMambaMCMC(sim1.value)


### flow.

## set up SDE.
N_discretizations = 1000
γ = 0.1
N_particles = 1000
#N_particles = 10000 # 3 minutes
#N_particles = 100000 # 30 minutes?

# set up Brownian motion.
λ_array, Bλ_array = drawBrownianmotiontrajectorieswithoutstart(N_discretizations, D_x)

### traverse the SDE for each particle.
println("preparing particles.")
drawxfunc = xx->rand(prior_dist)
@time xp_array, ln_wp_array, x_array = paralleltraverseSDEs(drawxfunc,
                            N_discretizations,
                            γ,
                            m_0,
                            P_0,
                            ψ,
                            R,
                            y,
                            get𝐻func,
                            get∂𝐻tfunc,
                            ln_prior_pdf_func,
                            ln_likelihood_func,
                            N_particles,
                            N_batches)



# normalize weights.
ln_W = StatsFuns.logsumexp(ln_wp_array)
w_array = collect( exp(ln_wp_array[n] - ln_W) for n = 1:N_particles )

w_sq_array = collect( exp(2*ln_wp_array[n] - 2*ln_W) for n = 1:N_particles )
ESS_GF = 1/(sum(w_sq_array))

println("estimated ESS of Gaussian Flow: ", ESS_GF)
println()

# sample covmat:
xp_array_weighted = xp_array.*w_array
m_s = sum(xp_array_weighted)
Q = getcovmatfromparticles(xp_array, m_s, w_array)


println("x, generating = ", x_generating)
println()

A = [   0.85438   0.906057;
        0.906057  1.12264 ]
f = xx->sinc(dot(xx,A*xx))^2
f_eval_GF = evalexpectation(f, xp_array, w_array)
f_eval_MCMC = evalexpectation(f, x_MCMC)
f_eval_NI, val_h, err_h, val_Z, err_Z = evalexpectation(f,
                                        likelihood_func,
                                        prior_func,
                limit_a, limit_b, max_integral_evals, initial_div)

println("NI: 𝔼[f] over posterior   = ", f_eval_NI)
println("val_h = ", val_h)
println("MCMC: 𝔼[f] over posterior = ", f_eval_MCMC)
println("GF: 𝔼[f] over posterior   = ", f_eval_GF)
println()

B = [   0.219055  0.290879;
        0.290879  0.398671]
g = xx->exp(-dot(xx,B*xx))
g_eval_GF = evalexpectation(g, xp_array, w_array)
g_eval_MCMC = evalexpectation(g, x_MCMC)
g_eval_NI, val_h, err_h, val_Z, err_Z = evalexpectation(g,
                                        likelihood_func,
                                        prior_func,
                limit_a, limit_b, max_integral_evals, initial_div)

println("NI: 𝔼[g] over posterior   = ", g_eval_NI)
println("val_h = ", val_h)
println("MCMC: 𝔼[g] over posterior = ", g_eval_MCMC)
println("GF: 𝔼[g] over posterior   = ", g_eval_GF)
println()

println("approx: MLE posterior parameters:")
println("m_s = ", m_s)
println("Q = ", Q)
println()

# Visualize via 2D histogram. Since there is a cosine, might be a multimodal density.
#include("plot2Dhistogram.jl")

n_bins = 20
display_limit_a = [-5.0; -5.0]
display_limit_b = [5.0; 5.0]
fig_num = Utilities.plot2Dhistogram(fig_num, xp_array, n_bins,
                display_limit_a, display_limit_b, true, "xp_array")
fig_num = Utilities.plot2Dhistogram(fig_num, xp_array_weighted, n_bins,
                display_limit_a, display_limit_b, true, "xp_array_weighted")

# discrepancy = sum( norm(xp_array[n]-xp_array2[n]) for n = 1:N_particles )
# println("discrepancy: x's posterior samples between approx. flow and exact flow: ", discrepancy)


# next, devise MCMC to verify the Bayesian posterior.
