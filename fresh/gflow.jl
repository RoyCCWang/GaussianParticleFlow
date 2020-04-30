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
N_discretizations = 1000
γ = 0.1/2
N_particles = 10000

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
                            R,
                            y,
                            ψ,
                            ∂ψ,
                            ∂2ψ,
                            ln_prior_pdf_func,
                            ln_likelihood_func,
                            N_particles,
                            N_batches)
#

# normalize weights.
ln_W = StatsFuns.logsumexp(ln_wp_array)
w_array = collect( exp(ln_wp_array[n] - ln_W) for n = 1:N_particles )

ln_w_sq_array = collect( 2*ln_wp_array[n] - 2*ln_W for n = 1:N_particles )
ESS_GF = 1/exp(StatsFuns.logsumexp(ln_w_sq_array))

println("estimated ESS of Gaussian Flow: ", ESS_GF)
println()

# sample covmat:
xp_array_weighted = xp_array.*w_array
m_s = sum(xp_array_weighted)
Q = getcovmatfromparticles(xp_array, m_s, w_array)


# visualize.
if D_x == 2
    n_bins = 500
    fig_num = VisualizationTools.plot2Dhistogram(fig_num,
                                xp_array,
                                n_bins,
                                limit_a,
                                limit_b;
                                use_bounds = true,
                                title_string = "xp locations",
                                colour_code = "jet",
                                use_color_bar = true,
                                axis_equal_flag = true,
                                flip_vertical_flag = false)

end

println("x, generating = ", x_generating)
println()

xp1 = collect( xp_array[n][1] for n = 1:length(xp_array) )
xp2 = collect( xp_array[n][2] for n = 1:length(xp_array) )

##### test functional. mean.
A = [   0.85438   0.906057;
        0.906057  1.12264 ]
#f = xx->sinc(dot(xx,A*xx))^2
f = xx->xx[1]
f_eval_GF = evalexpectation(f, xp_array, w_array)

println("GF: 𝔼[f] over posterior   = ", f_eval_GF)
 # should be around 1.34 for 2D to 2D.



println("NI test:")
@time f_eval_NI, val_h, err_h, val_Z, err_Z = evalexpectation(f,
                                 likelihood_func,
                                 prior_func,
         limit_a, limit_b, max_integral_evals, initial_div)

println("NI: 𝔼[f] over posterior   = ", f_eval_NI)
println("val_h = ", val_h)
println()

# got:
# preparing particles.
# 187.390355 seconds (1.37 M allocations: 68.901 MiB, 0.01% gc time)
# estimated ESS of Gaussian Flow: 22.578857939372952
#
# x, generating = [1.22, -0.35]
#
# GF: 𝔼[f] over posterior   = 1.3354071091016657
# NI test:
#  51.455690 seconds (879.18 M allocations: 40.474 GiB, 8.08% gc time)
# NI: 𝔼[f] over posterior   = 1.3489336018075437
# val_h = 0.02186045790028874
