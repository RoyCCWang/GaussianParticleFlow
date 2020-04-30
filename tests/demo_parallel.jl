# test exact Gaussian flow.

using Distributed
@everywhere using SharedArrays
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

@everywhere import ForwardDiff
@everywhere import StatsFuns

PyPlot.close("all")
fig_num = 1

@everywhere include("../src/misc/declarations.jl")
@everywhere include("../src/misc/utilities2.jl")

include("../src/SDE/Brownian.jl")
include("../src/flows/approx_flow.jl")
include("routines/sde.jl")
include("../src/flows/moments.jl")
include("../src/misc/utilities.jl")

include("routines/simulation_tools.jl")

Random.seed!(25)


#demo_string = "mixture"
#demo_string = "normal"

D_x = 2
D_y = 3
N_particles = 1000

# true distribution for latent variable, x.
m_true = randn(Float64,D_x)
P_true = randn(Float64,D_x,D_x)
P_true = P_true'*P_true

true_dist_x = Distributions.MvNormal(m_true, P_true)

# generate a value of the latent variable.
x_generating = rand(true_dist_x)

# observation model.
ψ = exampleψfunc2Dto3D1
R = randn(Float64,D_y,D_y)
R = R'*R

# generate observation.
true_dist_y = Distributions.MvNormal(ψ(x_generating),R)
y = rand(true_dist_y)

# prior.
m_0 = randn(Float64,D_x)
P_0 = randn(Float64,D_x,D_x)
P_0 = P_0'*P_0

prior_dist = Distributions.MvNormal(m_0, P_0)


# approx. flow required functions.

#ψ = examp2eψfunc2Dto1D1 # debug

get𝐻func = xx->ForwardDiff.jacobian(ψ, xx)

hessian_funcs = gethessianfuncs(ψ, D_y)
get∂𝐻tfunc = xx->compute∂𝐻tover∂x(hessian_funcs, xx, D_y)

ln_prior_pdf_func = xx->Distributions.logpdf(prior_dist, xx)
ln_likelihood_func = xx->evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R)

# x = randn(D_x) # debug
# u1 = get𝐻func(x) # debug
# u2 = get∂𝐻tfunc(x) # debug
# v1 = ln_likelihood_func(x)
# v2 = ln_prior_pdf_func(x)
# @assert 1==2






### flow.

## set up SDE.
N_discretizations = 1000
γ = 0.1

# set up Brownian motion.
λ_array, Bλ_array = drawBrownianmotiontrajectorieswithoutstart(N_discretizations, D_x)

### traverse the SDE for each particle.

# draw samples from the prior.
x_array = collect( rand(prior_dist) for i = 1:N_particles )

## traverse the SDE solution for each particle.
# weights, point_n x N.
results_storage = SharedArray{Float64}(N_particles, D_x, N_particles)

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
    #xp_array[n] = 𝑥[end]
    #ln_wp_array[n] = 𝑤[end]
    xp_mat[:,n] = 𝑥[end]
    ln_wp_array[n] = 𝑤[end]
    Printf.@printf("done particle %d\n", n)
end

workerfunc = xx->processinterval(xx, 𝓟, sample_position_template, sin, L, verbose_flag)

    sol = map(workerfunc, collect(i for i = 2:length(𝓟)) )

    sol2 = pmap(workerfunc, collect(i for i = 2:length(𝓟)) )


xp_array = collect( xp_mat[:,n] for n = 1:N_particles )
ln_wp_array = collect( ln_wp_array[n] for n = 1:N_particles )

# sample covmat:
#tmp = collect( log.(xp_array[n]) + ln_wp_array[n] - log(N_particles) for n = 1:N_particles )
#m_s = exp.( StatsFuns.logsumexp(tmp) )
m_s = sum( xp_array[n]*exp(ln_wp_array[n]) for n = 1:N_particles )/N_particles
Q = 1/(N_particles-1) * sum( (xp_array[n]-m_s)*(xp_array[n]-m_s)' for n = 1:N_particles)

println("x, generating = ", x_generating)
println()

# println("ground truth posterior parameters:")
# println("m_p = ", m_p)
# println("Σ_p = ", Σ_p)
# println()

println("MLE posterior parameters:")
println("m_s = ", m_s)
println("Q = ", Q)
println()
