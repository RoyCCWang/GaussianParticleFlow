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
limit_a = ones(Float64, D_x) .* -999.9
limit_b = ones(Float64, D_x) .* 999.9
initial_div = 1000


D_x = 2
D_y = 2
ψ = exampleψfunc2Dto2D1

# true distribution for latent variable, x.
x_generating = [1.22; -0.35]

# observation model aside from ψ.

σ = 0.02
R = diagm( 0 => collect(σ for d = 1:D_y) )

# prior.
m_0 = randn(Float64, D_x)
P_0 = Utilities.generaterandomposdefmat(D_x)

prior_dist = Distributions.MvNormal(m_0, P_0)

# generate observation.
true_dist_y = Distributions.MvNormal(ψ(x_generating),R)

N_obs = 5
Y = collect( rand(true_dist_y) for n = 1:N_obs )

R_set = collect( R for n = 1:N_obs )
Σ_y = Utilities.makeblockdiagonalmatrix(R_set)

# inverse is invertblockmatrix!(inv_K, K_set)

# https://math.hecker.org/2011/06/25/multiplying-block-diagonal-matrices/
# implement blockdiagonal matrix multiplication of diff size blocks.


# I am here:
#  to do:
# - block diagonal R for product likelihood,
#       this means the Hessians are also block diag.
# - update GF to use bloack diagonal R and Hessian.
# - IS_engine.jl, approx_flow.jl, moments.jl
#
# - add adaptive step size control, with
#   minimum step size.
