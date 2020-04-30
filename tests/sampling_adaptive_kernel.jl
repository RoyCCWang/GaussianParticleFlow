# final scheme. adaptive kernel.

using Distributed

@everywhere import StaticArrays

@everywhere import Printf
@everywhere import PyPlot
@everywhere import Random
@everywhere import Optim

@everywhere using LinearAlgebra

@everywhere using FFTW

@everywhere import Statistics

@everywhere import Distributions
@everywhere import HCubature
@everywhere import Interpolations
@everywhere import SpecialFunctions

@everywhere import SignalTools
@everywhere import RKHSRegularization
@everywhere import Utilities

@everywhere import Convex
@everywhere import SCS

@everywhere import Calculus
@everywhere import ForwardDiff

#@everywhere import GenericMCMCSamplers
#@everywhere import stickyHDPHMM

@everywhere using Printf
@everywhere import GSL

@everywhere import ForwardDiff
@everywhere import StatsFuns

import VisualizationTools

#@everywhere using Mamba

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

#@everywhere include("../src/RKHS/warpmap.jl")

@everywhere include("../src/flows/SDE_adaptive_kernel.jl")


PyPlot.close("all")
fig_num = 1

Random.seed!(25)



D = 2
d_select = 2
τ = 15.0

integration_limit_a = -999.9 .* ones(Float64,D)
integration_limit_b = 999.9 .* ones(Float64,D)
max_integral_evals = typemax(Int)
initial_div = 1000


### coordinates set up.
N_array = [100; 100]
x_ranges = collect( LinRange(-τ, τ, N_array[d]) for d = 1:D )
X_nD = Utilities.ranges2collection(x_ranges, Val(D))

τ = 15.0

Nq_array = [200; 200]
xq_ranges = collect( LinRange(-τ, τ, Nq_array[d]) for d = 1:D )
Xq_nD = Utilities.ranges2collection(xq_ranges, Val(D))


# prepare all marginal joint densities.
A = [   0.85438   0.906057;
        0.906057  1.12264 ]
f = xx->sinc(dot(xx,A*xx))^2

@time f_Xq_nD = f.(Xq_nD)
println()

fig_num = VisualizationTools.visualizemeshgridpcolor(xq_ranges, f_Xq_nD, [], ".", fig_num,
                                            "target density")
#




####### get warpmap.

attenuation_factor_at_cut_off = 2.0
N_bands = 5
reciprocal_cut_off_percentages = ones(N_bands) ./collect(LinRange(1.0,0.2,N_bands))
ω_set = collect( π/(reciprocal_cut_off_percentages[i]*sqrt(2*log(attenuation_factor_at_cut_off))) for i = 1:length(reciprocal_cut_off_percentages) )
pass_band_factor = abs(ω_set[1]-ω_set[2])*0.2
amplification_factor_hr = 50.0

f_X_nD = f.(X_nD)
ϕ = RKHSRegularization.getRieszwarpmapsamples(f_X_nD, Val(:simple), Val(:uniform), ω_set, pass_band_factor)
ϕ_map_itp, dϕ_map_itp, d2ϕ_map_itp = Utilities.setupcubicitp(ϕ, x_ranges[1:d_select], amplification_factor_hr)

fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges[1:d_select], ϕ_map_itp.(X_nD), [], ".",
                fig_num, "ϕ_map_itp, X_full"; x1_title_string = "x1",
                                                x2_title_string = "x2")

### get canonical and adaptive kernel.
a = 1/2.34
θ_canonical = RKHSRegularization.GaussianKernel1DType(a)
θ_a = RKHSRegularization.AdaptiveKernelType(θ_canonical, ϕ_map_itp)

### visualize kernels.
#
marker_pt = [1.16; 0.02] #[3.8; 3.3]
title_string = @sprintf("kernel profile, itp, center = (%.3f,%.3f)",
                            marker_pt[1], marker_pt[2])
k_𝓧_itp = xx->RKHSRegularization.evalkernel(convert(Vector{Float64},xx), marker_pt, θ_a)
k_𝓧_itp_X_nD = k_𝓧_itp.(X_nD)
fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges[1:d_select],
                            k_𝓧_itp_X_nD, [], "r.",
                            fig_num, title_string; x1_title_string = "x1",
                                                x2_title_string = "x2")
#@assert 1==2222

# A = [   0.85438   0.906057;
#         0.906057  1.12264 ]
# h = xx->sinc(dot(xx,A*xx))^2
#
# h_eval_NI, val_integrand, err_integrand, val_Z, err_Z = evalexpectation(h,
#                                         k_𝓧_itp,
#                 integration_limit_a, integration_limit_b,
#                 max_integral_evals, initial_div)
#
# #
# println("NI: 𝔼[h] over posterior   = ", h_eval_NI)
# println("val_integrand = ", val_integrand)
# @assert 1==232



#### sample adaptive kernel.
N_viz = 500
z = marker_pt


σ = 1/sqrt(2*a)

D_x = D

drawxfunc = xx->Utilities.drawnormal(z, σ)

ψ = xx->[ ϕ_map_itp(xx) ]

N_discretizations = 1000
γ = 0.1
#N_particles = 10000 # 4.25 min, 255 sec.
#N_particles = 20000 #
N_particles = 1000
λ_array, Bλ_array = drawBrownianmotiontrajectorieswithoutstart(N_discretizations, D_x)

m_0 = z
P_0 = Matrix{Float64}(LinearAlgebra.I, D_x, D_x) .* σ^2

R = Matrix{Float64}(undef,1,1)
R[1,1] = σ^2

y = [ ϕ_map_itp(z) ]

D_y = length(y)
get𝐻func2 = xx->Calculus.jacobian(ψ, xx, :central)
hessian_funcs = gethessianfuncsND(ψ, D_y)
get∂𝐻tfunc2 = xx->compute∂𝐻tover∂x(hessian_funcs, xx, D_y)

get𝐻func = xx->convertcolvectorowvec(d_ϕ_map_itp(xx))
get∂𝐻tfunc = xx->convertmatrixtonestedcolmats(d2_ϕ_map_itp(xx))

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

ln_prior_pdf_func = xx->evallnMVN(xx, m_0, P_0)
ln_likelihood_func = xx->evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R)


# # sequential.
# # traverse the SDE solution for each particle.
# x_array = collect( drawxfunc(1.0) for i = 1:N_particles )
# xp_array = Vector{Vector{Float64}}(undef,N_particles)
# ln_wp_array = Vector{Float64}(undef,N_particles)
# for n = 1:N_particles
#
#     𝑥, 𝑤 = traversesdepathapproxflow(   x_array[n],
#                                         λ_array,
#                                         Bλ_array,
#                                         γ,
#                                         m_0,
#                                         P_0,
#                                         ψ,
#                                         R,
#                                         y,
#                                         get𝐻func,
#                                         get∂𝐻tfunc,
#                                         ln_prior_pdf_func,
#                                         ln_likelihood_func)
#     xp_array[n] = 𝑥[end]
#     ln_wp_array[n] = 𝑤[end]
#     Printf.@printf("done particle %d\n", n)
# end
#
# @assert 1==2

N_batches = 16
println("Computing particles.")
@time xp_array, ln_wp_array, x_array = runGFISonadaptivekernel(z,
                    ϕ_map_itp,
                    dϕ_map_itp,
                    d2ϕ_map_itp,
                    N_discretizations,
                    γ,
                    N_particles,
                    N_batches,
                    θ_a.canonical_params.ϵ_sq)



# N_batches = 16
# println("Computing particles.")
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
# #
# # normalize weights.
# ln_W = StatsFuns.logsumexp(ln_wp_array)
# w_array = collect( exp(ln_wp_array[n] - ln_W) for n = 1:N_particles )

ln_W = StatsFuns.logsumexp(ln_wp_array)
w_array = collect( exp(ln_wp_array[n] - ln_W) for n = 1:N_particles )

xp_array_weighted = xp_array.*w_array

w_sq_array = collect( exp(2*ln_wp_array[n] - 2*ln_W) for n = 1:N_particles )

ESS_GF = 1/(sum(w_sq_array))
println("estimated ESS of Gaussian Flow: ", ESS_GF)
println()


### plot histogram.
n_bins = 100
display_limit_a = [-τ; -τ]
display_limit_b = [τ; τ]
fig_num = Utilities.plot2Dhistogram(fig_num, xp_array, n_bins,
                display_limit_a, display_limit_b, true, "xp_array")
fig_num = Utilities.plot2Dhistogram(fig_num, xp_array_weighted, n_bins,
                display_limit_a, display_limit_b, true, "xp_array_weighted")



### test functional.

# B = A
B = [   0.219055  0.290879;
        0.290879  0.398671]
h = xx->exp(-dot(xx,B*xx))

h_eval_NI, val_integrand, err_integrand, val_Z,
    err_Z = evalexpectation(h, k_𝓧_itp,
                integration_limit_a, integration_limit_b,
                max_integral_evals, initial_div)

#
println("NI: 𝔼[h] over posterior   = ", h_eval_NI)
println("val_integrand = ", val_integrand)
println("err_integrand = ", err_integrand)


h_eval_GF = evalexpectation(h, xp_array, w_array)
println("GF: 𝔼[h] over posterior   = ", h_eval_GF)

#h_eval_MCMC = evalexpectation(h, x_MCMC)

# empirical CDF:
# https://towardsdatascience.com/what-why-and-how-to-read-empirical-cdf-123e2b922480
#
# viualize weighted empirical CDF
# https://blogs.sas.com/content/iml/2016/08/29/weighted-percentiles.html
