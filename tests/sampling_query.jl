
# final scheme. adaptive kernel.

using Distributed

@everywhere import JLD
@everywhere import FileIO

@everywhere using Printf
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


@everywhere import GSL
@everywhere import StatsFuns


@everywhere import StaticArrays

import VisualizationTools

@everywhere include("../src/misc/declarations.jl")
@everywhere include("../src/SDE/Brownian.jl")
@everywhere include("../src/flows/approx_flow.jl")
@everywhere include("routines/sde.jl")
@everywhere include("../src/flows/moments.jl")
@everywhere include("../src/misc/utilities.jl")

@everywhere include("routines/simulation_tools.jl")
@everywhere include("../src/flows/exact_flow.jl")

@everywhere include("../src/importance_sampler/IS_engine.jl")
@everywhere include("../src/diagnostics/functionals.jl")
@everywhere include("../src/diagnostics/test_functions.jl")

@everywhere include("../src/RKHS/fit_density.jl")
@everywhere include("../src/importance_sampler/IS_uniform.jl")

#@everywhere include("../src/RKHS/warpmap.jl")
#@everywhere include("../src/RKHS/RKHS_helpers.jl")

@everywhere include("../src/flows/SDE_adaptive_kernel.jl")


PyPlot.close("all")
fig_num = 1

Random.seed!(25)


D = 2

N_array = [30; 30]

τ = 1e-2
limit_a = [τ; τ]
limit_b = [1-τ; 1-τ]

f_normalized, x_ranges = getmixture2Dbetacopula1(τ, N_array)
f = xx->f_normalized(xx)/35


Xq_nD = Utilities.ranges2collection(x_ranges, Val(D))
X = vec(Xq_nD)

#X = collect( collect( Utilities.convertcompactdomain(rand(), 0.0, 1.0, limit_a[d], limit_b[d]) for d = 1:D ) for n = 1:N_realizations )
println("Computing f(X):")
@time f_X = f.(X)
println()


zero_tol_RKHS = 1e-13
prune_tol = 1.1*zero_tol_RKHS
max_iters_RKHS = 5000
σ² = 1e-5

amplification_factor = 1.0 # 0.0
attenuation_factor_at_cut_off = 2.0
N_bands = 5
#a_RQ = 0.7 / 350
a_SqExp = 500.0
c_out, 𝓧, θ_a, dϕ, d2ϕ = fitdensityitp(f_X, X, x_ranges,
                f, max_iters_RKHS,
                a_SqExp, σ², amplification_factor,
                N_bands, attenuation_factor_at_cut_off,
                zero_tol_RKHS, prune_tol)

fq = xx->RKHSRegularization.evalquery(xx, c_out, 𝓧, θ_a)
# visualize full joint density.
if D == 2
    #Xq_nD = Utilities.ranges2collection(xq_ranges, Val(D))
    fq_Xq_nD = fq.(Xq_nD)
    fig_num = Utilities.visualizemeshgridpcolor(xq_ranges,
                            fq_Xq_nD, 𝓧, "x", fig_num, "fq")

    # visualize kernel footprint.
    θ_canonical = θ_a.canonical_params
    marker_pt = [0.5; 0.5]
    title_string = @sprintf("k(⋅,z), z = (%.1f,%.1f)", marker_pt[1], marker_pt[2])
    k_z = xx->RKHSRegularization.evalkernel(xx, marker_pt, θ_canonical)
    k_z_Xq_nD = k_z.(Xq_nD)
    fig_num = Utilities.visualizemeshgridpcolor(xq_ranges,
                    k_z_Xq_nD, [], "r.",
                    fig_num, title_string, "x1", "x2")
end

println("working on fq_pdf_tilde.")
fq_ln_pdf_tilde2 = RKHSquerytopdf(c_out, 𝓧, θ_a.canonical_params.ϵ_sq, θ_a.warpfunc)
fq_pdf_tilde2 = xx->exp(fq_ln_pdf_tilde2(xx))

println("Working on normalizing constant of fq_pdf2")
integration_limit_a = -9.9 .* ones(Float64,D)
integration_limit_b = 9.9 .* ones(Float64,D)
max_integral_evals = 10000 #typemax(Int) # 1000
initial_div = 1 # 1000
# @time val_Z2, err_Z2 = evalintegral( fq_pdf_tilde2,
#                         integration_limit_a,
#                         integration_limit_b,
#                         max_integral_evals,
#                         initial_div)
# #
# fq_pdf2 = xx->fq_pdf_tilde2(convert(Vector{Float64},xx))/val_Z2
# @time val_chk, err_chk = evalintegral( fq_pdf2,
#                         integration_limit_a,
#                         integration_limit_b,
#                         max_integral_evals,
#                         initial_div)
# println("val_chk = ", val_chk)
# println("err_chk = ", err_chk)
# println()


fq_pdf_tilde = xx->evaladaptivequery(xx, c_out, 𝓧, θ_a.canonical_params.ϵ_sq, θ_a.warpfunc)

println("Working on normalizing constant of fq_pdf")
@time val_Z0, err_Z0 = evalintegral( fq_pdf_tilde,
                        integration_limit_a,
                        integration_limit_b,
                        max_integral_evals,
                        initial_div)
#
fq_pdf = xx->fq_pdf_tilde(xx)/val_Z0

@time val_chk, err_chk = evalintegral( fq_pdf,
                        integration_limit_a,
                        integration_limit_b,
                        max_integral_evals,
                        initial_div)
println("val_chk = ", val_chk)
println("err_chk = ", err_chk)
# visualize full joint density.
if D == 2
    #Xq_nD = Utilities.ranges2collection(xq_ranges, Val(D))
    fq_pdf_Xq_nD = fq_pdf.(Xq_nD)
    fig_num = Utilities.visualizemeshgridpcolor(xq_ranges,
                            fq_pdf_Xq_nD, 𝓧, "x", fig_num, "fq_pdf")

end


B = [   0.219055  0.290879;
        0.290879  0.398671]
h = xx->exp(-dot(xx,B*xx))

println("Evaluating functional.")
@time h_eval_NI, val_integrand, err_integrand, val_Z, err_Z = evalexpectation(h,
                                        fq_pdf,
                integration_limit_a, integration_limit_b,
                max_integral_evals, initial_div)
#
# #
# println("NI: 𝔼[h] over posterior   = ", h_eval_NI)
# println("val_integrand = ", val_integrand)
# println("err_integrand = ", err_integrand)

#@assert 1==2

### draw GF importance samples.
# N_discretizations = 10000
# γ = 0.9
# # need to crank up N_discretizations if γ is large.

#γ = 0.69*N_discretizations #5e8#5e6

N_discretizations = 1000
γ =  0.1
#γ = 0.0
#N_processes = 7
max_N_batches = 16
N_samples = 1000 #20000

println("Starting runGFISonquery:")
ϕ = xx->θ_a.warpfunc(xx)
#ϕ = xx->0.0*θ_a.warpfunc(xx)
@time xp_array, ln_wp_array, x_array = runGFISonquery( 𝓧, c_out, ϕ, dϕ, d2ϕ, N_discretizations, γ, N_samples, max_N_batches, a_SqExp )


### diagnostics.
ln_W = StatsFuns.logsumexp(ln_wp_array)
w_array = collect( exp(ln_wp_array[n] - ln_W) for n = 1:N_samples )

xp_array_weighted = xp_array.*w_array

w_sq_array = collect( exp(2*ln_wp_array[n] - 2*ln_W) for n = 1:N_samples )

ESS_GF = 1/(sum(w_sq_array))
println("estimated ESS of Gaussian Flow: ", ESS_GF)
println()


### plot histogram.
n_bins = 100
display_limit_a = limit_a
display_limit_b = limit_b
fig_num = Utilities.plot2Dhistogram(fig_num, xp_array, n_bins,
                display_limit_a, display_limit_b, true, "xp_array")

#
fig_num = Utilities.plot2Dhistogram(fig_num, xp_array, n_bins,
                display_limit_a, display_limit_b, false, "xp_array, no display bounds")



# I am here. code the normalized prob. distribution of each adaptive kernel;
#   just brute-force eval their normalizing constants.
# then numerically integrate.

### test functional.

# B = A
#
println("NI: 𝔼[h] over posterior   = ", h_eval_NI)
println("val_integrand = ", val_integrand)
println("err_integrand = ", err_integrand)


# h_eval_GF = evalexpectation(h, xp_array, w_array)
# println("GF: 𝔼[h] over posterior   = ", h_eval_GF)

h_eval_GF2, a, b = evalexpectation2(h, xp_array, ln_wp_array)
println("GF2: 𝔼[h] over posterior   = ", h_eval_GF2)
println()


println("Evaluating mean.")
max_integral_evals = typemax(Int)
initial_div = 5
@time h_eval_NI, val_integrand, err_integrand, val_Z, err_Z = evalexpectation(xx->xx[1],
                                        fq_pdf,
                integration_limit_a, integration_limit_b,
                max_integral_evals, initial_div)
#
println("NI: 𝔼[x] over posterior   = ", h_eval_NI)

mean_GF2, a, b = evalexpectation2(xx->xx[1], xp_array, ln_wp_array)
println("GF2: 𝔼[x] over posterior   = ", mean_GF2)
