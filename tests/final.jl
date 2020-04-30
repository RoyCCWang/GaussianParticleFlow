
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
@everywhere import AdaptiveRKHS
@everywhere import Utilities

@everywhere import Convex
@everywhere import SCS

@everywhere import Calculus
@everywhere import ForwardDiff

@everywhere import GenericMCMCSamplers
@everywhere import stickyHDPHMM

@everywhere import GSL
@everywhere import StatsFuns

@everywhere using Mamba
@everywhere import StaticArrays

@everywhere import KRTransportMap

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

# @everywhere include("../src/RKHS/fit_density.jl")
# @everywhere include("../src/RKHS/warpmap.jl")
# @everywhere include("../src/RKHS/RKHS_helpers.jl")

@everywhere include("../src/flows/SDE_adaptive_kernel.jl")

@everywhere include("../src/subset/construct_subsets.jl")

@everywhere include("./unit_interval/unit_interval_helpers.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)


##### set up coordinates and other settings.

D = 2

N_array = [30; 30]

τ = 1e-2
limit_a = [τ; τ]
limit_b = [1-τ; 1-τ]

integration_limit_a = limit_a
integration_limit_b = limit_b
max_integral_evals = 10000 #typemax(Int) # 1000
initial_div = 1

Nq_array = [200; 200]
xq_ranges = collect( LinRange(limit_a[d], limit_b[d], Nq_array[d]) for d = 1:D )
Xq_nD = Utilities.ranges2collection(xq_ranges, Val(D))

##### set up oracle unnormalized density, f.
f_normalized, x_ranges = getmixture2Dbetacopula1(τ, N_array)
f = xx->f_normalized(xx)/35

println("Working on normalizing constant of fq_pdf")
@time val_Z_f, err_Z_f = evalintegral( f,
                        integration_limit_a,
                        integration_limit_b,
                        max_integral_evals,
                        initial_div)
#
f_pdf = xx->f(xx)/val_Z_f

@time val_chk, err_chk = evalintegral( f_pdf,
                        integration_limit_a,
                        integration_limit_b,
                        max_integral_evals,
                        initial_div)
println("f_pdf, normalizing constant: val_chk = ", val_chk)
println("f_pdf, normalizing constant: err_chk = ", err_chk)
println()

# visualize full joint density.
if D == 2
    #Xq_nD = Utilities.ranges2collection(xq_ranges, Val(D))
    f_Xq_nD = f.(Xq_nD)
    fig_num = Utilities.visualizemeshgridpcolor(xq_ranges, f_Xq_nD, [], ".", fig_num,
                                            "f")
    #
    #Xq_nD = Utilities.ranges2collection(xq_ranges, Val(D))
    f_pdf_Xq_nD = f_pdf.(Xq_nD)
    fig_num = Utilities.visualizemeshgridpcolor(xq_ranges, f_pdf_Xq_nD, [], ".", fig_num,
                                            "f_pdf")
end



##### deconstruction.
w_config = packageupunitinterval(f, limit_a, limit_b)

φ_LP, φ_HP, dφ_LP, dφ_HP, d2φ_LP, d2φ_HP, X_full,
                f_oracle, f_oracle_multiplier,
                c_LP, X_LP, c_HPc, X_HPc,
                c_f_proxy, X_f_proxy, a_f_proxy,
                w_IS, X_IS = KRTransportMap.getwarpmapspline2( f,
                                                limit_a,
                                                limit_b,
                                                w_config)
    #
    ## prepare adaptive kernel.


ϕ = φ_LP
dϕ = dφ_LP
d2ϕ = d2φ_LP

ψ = φ_HP
dψ = dφ_HP
d2ψ = d2φ_HP

### prepare kernel.
a_SqExp = 500.0 #500.0
θ_canonical = AdaptiveRKHS.GaussianKernel1DType(a_SqExp)
θ_a = KRTransportMap.getadaptivekernel(θ_canonical, w_config.warp_weights, ϕ, ψ)

### prepare kernel centers
X = KRTransportMap.selectkernelcenters(X_full, w_IS, X_IS, ϕ, ψ, w_config)
f_oracle_X = f_oracle.(X)

# visualize kernel centers.
fig_num = Utilities.visualizemeshgridpcolor(xq_ranges, f_Xq_nD, X, "x", fig_num,
                                        "f, X")

#
# visualize kernel centers.
fig_num = Utilities.visualizemeshgridpcolor(xq_ranges, ϕ.(Xq_nD), X, "x", fig_num,
                                        "ϕ, X")
#
# visualize kernel centers.
fig_num = Utilities.visualizemeshgridpcolor(xq_ranges, ψ.(Xq_nD), X, "x", fig_num,
                                        "ψ, X")

# reason is because the kernel fitting I just exported use square root, which is
#   not kernel density. The fit is not stellar as well, vs. no warp map.

##### fit density.
println("Fitting density.")

# compact interval conversion too slow.
# a_SqExp = 1.0 # wrt to [-15, 15]^D
# θ_canonical = AdaptiveRKHS.GaussianKernel1DType(a_SqExp)
#@time c, 𝓧, fq, f_oracle = fitdensitywithsettings(f, θ_canonical, limit_a, limit_b)

a_SqExp = 500.0 #500.0
θ_canonical = AdaptiveRKHS.GaussianKernel1DType(a_SqExp)
@time c, 𝓧, θ_a, fq, f_oracle = getconfigforunitintervaldensities(f, θ_canonical, limit_a, limit_b)

zero_tol_RKHS = 1e-13
prune_tol = 1.1*zero_tol_RKHS
max_iters_RKHS = 50000
σ² = 1e-9
@time c_canonical, 𝓧_can, θ_can = fitdensityisonormal(f_oracle.(𝓧), 𝓧, max_iters_RKHS,
            a_SqExp, σ², zero_tol_RKHS, prune_tol )

f_can = xx->AdaptiveRKHS.evalquery(xx, c_canonical, 𝓧_can, θ_can)
println("Finished fitting density.")
println()

# visualize full joint density.
if D == 2
    #Xq_nD = Utilities.ranges2collection(xq_ranges, Val(D))
    fq_Xq_nD = fq.(Xq_nD)
    fig_num = Utilities.visualizemeshgridpcolor(xq_ranges,
                            fq_Xq_nD, 𝓧, "x", fig_num, "fq")
    #
    f_can_Xq_nD = f_can.(Xq_nD)
    fig_num = Utilities.visualizemeshgridpcolor(xq_ranges,
                            f_can_Xq_nD, 𝓧_can, "x", fig_num, "f_can")

    # visualize kernel footprint.
    marker_pt = (limit_b-limit_a) ./ 2 # marker at the middle of the sample space.
    title_string = @sprintf("canonical k(⋅,z), z = (%.1f,%.1f)", marker_pt[1], marker_pt[2])
    k_z = xx->AdaptiveRKHS.evalkernel(xx, marker_pt, θ_canonical)
    k_z_Xq_nD = k_z.(Xq_nD)
    fig_num = Utilities.visualizemeshgridpcolor(xq_ranges,
                    k_z_Xq_nD, [], "r.",
                    fig_num, title_string, "x1", "x2")

    #
    title_string = @sprintf("adaptive k(⋅,z), z = (%.1f,%.1f)", marker_pt[1], marker_pt[2])
        k_a_z = xx->AdaptiveRKHS.evalkernel(xx, marker_pt, θ_a)
    k_a_z_Xq_nD = k_a_z.(Xq_nD)
    fig_num = Utilities.visualizemeshgridpcolor(xq_ranges,
                    k_a_z_Xq_nD, [], "r.",
                    fig_num, title_string, "x1", "x2")
end
