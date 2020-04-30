# Parno's experiment.

using Distributed
@everywhere using SharedArrays

@everywhere import Printf
@everywhere import PyPlot
@everywhere import Random


@everywhere using LinearAlgebra

@everywhere using FFTW

@everywhere import HCubature

@everywhere import SignalTools
@everywhere import AdaptiveRKHS
@everywhere import Utilities
@everywhere import KRTransportMap


@everywhere include("../src/RKHS/RKHS_helpers.jl")
@everywhere include("../src/diagnostics/functionals.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

max_integral_evals_marginalization = 10000 #500 #typemax(Int)

## select dimension.
D = 1
N_array = [200]

f = xx->exp(-10*norm(xx)^2)*cos(7*norm(xx))^2+0.2*exp(-norm(xx)^2)

limit_a = [ -15.0 ]
limit_b = [ 15.0 ]

# limit_a = [ -10.0 ]
# limit_b = [ 30.0 ]

x_ranges = collect( LinRange(limit_a[1], limit_b[1], N_array[d]) for d = 1:D )


println("limit_a = ", limit_a)
println("limit_b = ", limit_b)
println()

# visualize.
N_display = 500
xq = LinRange(limit_a[1], limit_b[1], N_display)
#xq_vec = collect( [xq[n]] for n = 1:length(xq) )

PyPlot.figure(fig_num)
fig_num += 1

#PyPlot.plot(Xq, gq.(Xq))
PyPlot.plot(xq, f.(xq))

PyPlot.title("f")
PyPlot.legend()


c, θ, 𝓧, g,
    q_1D, f_oracle_Z = KRTransportMap.fitadaptiveRKHSdensity1D(f, limit_a[1], limit_b[1])

#
max_integral_evals = 10000
cdf_g = xx->evalintegral(g, limit_a[1], xx, max_integral_evals = max_integral_evals)
cdf_f_oracle = xx->evalintegral(f_oracle, limit_a[1], xx, max_integral_evals = max_integral_evals)


PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(xq, f_oracle_Z.(xq), "--", label = "f_oracle_Z")
PyPlot.plot(xq, g.(xq), "x")
PyPlot.plot(xq, g.(xq), label = "g")

PyPlot.title("Fitted Density")
PyPlot.legend()

u = 0.5
@time q_1D_u = q_1D(u)

println("q_1D(u)        = ", q_1D_u)
println("cdf_g(q_1D(u)) = ", cdf_g(q_1D_u))
println()

println("Timing: CDF.")
@time cdf_g(q_1D_u)
@time cdf_g(q_1D_u)
println()
