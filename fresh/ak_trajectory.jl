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
@everywhere include("./adaptive_kernel/L1_moments.jl")

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



###### verify moment expressions. pakage into test script when done.
L = 1

x0 = randn(D_x)
λ0 = rand()
λ1 = λ0 + rand()




inv_P0 = diagm(1/σ² * ones(Float64, D_x))
inv_R = diagm(1/σ² * ones(Float64, L))

getPquantities_Bunch = (xx,λλ)->computePquantitiesBunch(xx, λλ, ψ, inv_P0, inv_R)

getPquantities_mine = (xx,λλ)->computePquantitiesL1(xx, λλ, ψ, σ²)

B, inv_B = getPquantities_Bunch(x0, λ0)
C, inv_C = getPquantities_mine(x0, λ0)

println("verify my P update.")
println("l-2 discrepancy of P: mine vs. Bunch is ", norm(B-C))
println("l-2 discrepancy of inv(P): mine vs. Bunch is ", norm(inv_B-inv_C))
println()



# # verify sqrtm formula.
# u = randn(D_x)
# K = rand()* 5.3
#
# Q = LinearAlgebra.I + K .* u*u'
#
# x = ( sqrt(1+dot(u,u)*K)-1 )/dot(u,u)
# sqrt_Q = LinearAlgebra.I + x .* u*u'
# println("verify sqrt formula.")
# println("l-2 discrepancy of sqrt(Q) vs. Q is ", norm(sqrt_Q-sqrt(Q)))
# println()



### verify formula for d(uᵀu).

function evalduᵀu(ψ, x::Vector{T})::Vector{T} where T
    D = length(x)

    dψ_x = ForwardDiff.gradient(ψ, x)
    d2ψ_x = ForwardDiff.hessian(ψ, x)

    return collect( sum( 2*dψ_x[i]*d2ψ_x[i,j] for i = 1:D ) for j = 1:D )
end

function evaluᵀu(ψ, x::Vector{T})::T where T

    u = ForwardDiff.gradient(ψ, x)

    return dot(u,u)
end

ψ_scalar = xx->ψ(xx)[1]

f = xx->evaluᵀu(ψ_scalar, xx)
df_AD = xx->ForwardDiff.gradient(f, xx)

df_AN = xx->evalduᵀu(ψ_scalar, xx)

## comment out for speed.
# A = df_AD(x0)
# B = df_AN(x0)
#
# println("verify d(uᵀu) formula.")
# println("d(uᵀu), L = 1: l-2 discrepancy between AN and AD is ", norm(A-B))
# println()



# skip to speed up.
# ### verify d𝐺 formula.
# get𝐺_Bunch = xx->compute𝐺Bunch(xx, λ0, λ1, ψ, inv_P0, inv_R)
# get𝐺quantities_mine = xx->compute𝐺(xx, λ0, λ1, ψ_scalar, σ²)
#
# G_Bunch = get𝐺_Bunch(x0)
# dG_Bunch = computegradientformatrixfunctionND(x0, get𝐺_Bunch, D_x, D_x)
#
# G_mine, dG_mine = get𝐺quantities_mine(x0)
#
# println("verify 𝐺 formula.")
# println("𝐺, L = 1: l-2 discrepancy between my formula and Bunch is  ", norm(G_Bunch-G_mine))
# println("d𝐺, L = 1: l-2 discrepancy between my formula and Bunch is ", norm(dG_Bunch-dG_mine))
# println()


### Verify dm.
import FiniteDiff


### next, devise fast expression for weight Jacobian and state updates.

### consider how remedy negative determinant. look at logabsdet.
### consider how to remedy logabsdet = -Inf, i.e. determinant at x0 is zero,
#       i.e. state update is not invertible there. add noise?

# intermediate tests, leading up to d𝑚.
y_hat_func = xx->computeyhatL1(xx, ψ, y)
dy_hat_AD = xx->ForwardDiff.jacobian(y_hat_func, xx)

d2ψ_scalar = xx->ForwardDiff.hessian(ψ_scalar, xx)

dy_hat_x0_mine = collect( dot( d2ψ_scalar(x0)[:,j], x0) for j = 1:D_x )
dy_hat_x0_AD = vec(dy_hat_AD(x0))

println("dy_hat discrepancy: mine vs. AD is ", norm(dy_hat_x0_mine-dy_hat_x0_AD))
println()


y_scalar = y[1]
κ_func = xx->computeκ(xx, z_input, λ0, ψ_scalar, σ², y_scalar)
κ_x0 = κ_func(x0)

dκ_AD = xx->ForwardDiff.gradient(κ_func, xx)
dκ_AD_x0 = dκ_AD(x0)

dκ = xx->computedκ(xx, z_input, λ0, ψ_scalar, σ², y_scalar)
dκ_x0 = dκ(x0)

println("dκ discrepancy: mine vs. AD is ", norm(dκ_x0-dκ_AD_x0))
println()
#@assert 1==2



#####
get𝑚_Bunch = xx->compute𝑚Bunch(xx, λ0, ψ, z_input, inv_P0, inv_R, y)

getd𝑚_Bunch = xx->computed𝑚Bunch(xx, λ0, ψ, z_input, inv_P0, inv_R, y)

𝑚_Bunch = get𝑚_Bunch(x0)

d𝑚_Bunch_mat = ForwardDiff.jacobian(xx->get𝑚_Bunch(xx), x0)
d𝑚_Bunch = collect( vec(d𝑚_Bunch_mat[j,:]) for j = 1:D_x )

#𝑚_mine, d𝑚_mine, dk_mine = get𝑚quantities_mine(x0)

𝑚_func_mine = xx->compute𝑚(xx, z_input, λ0, ψ_scalar, σ², y_scalar)
d𝑚_AD = xx->ForwardDiff.jacobian(𝑚_func_mine, xx)

d𝑚_mine = xx->computed𝑚(xx, z_input, λ0, ψ_scalar, σ², y_scalar)

d𝑚_mine_x0 = d𝑚_mine(x0)
d𝑚_mine_x0_mat = convertnestedvector(d𝑚_mine_x0)
#d𝑚_mine_x0_mat = d𝑚_mine_x0_mat'

d𝑚_AD_x0 = d𝑚_AD(x0)

println("d𝑚: l-2 discrepancy between AN and AD is ", norm(d𝑚_AD_x0-d𝑚_mine_x0_mat))

@assert 1==2

println("verify 𝑚 formula.")
println("𝑚, L = 1: l-2 discrepancy between my formula and Bunch is  ", norm(𝑚_Bunch-𝑚_mine))
println("d𝑚, L = 1: l-2 discrepancy between my formula and Bunch is ", norm(d𝑚_Bunch-d𝑚_mine))
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
