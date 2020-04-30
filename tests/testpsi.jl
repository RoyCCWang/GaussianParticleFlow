
# GPODE on FHN, first component.

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
@everywhere import DifferentialEquations

@everywhere include("../src/SDE/Brownian.jl")
@everywhere include("../src/flows/approx_flow.jl")
#@everywhere include("routines/sde.jl")
@everywhere include("../src/flows/moments.jl")
@everywhere include("../src/misc/utilities.jl")

#@everywhere include("routines/simulation_tools.jl")
@everywhere include("../src/flows/exact_flow.jl")

@everywhere include("../src/importance_sampler/IS_engine.jl")
@everywhere include("../src/diagnostics/functionals.jl")
@everywhere include("../src/diagnostics/test_functions.jl")

@everywhere include("../src/RKHS/fit_density.jl")
@everywhere include("../src/importance_sampler/IS_uniform.jl")

@everywhere include("../src/RKHS/warpmap.jl")
@everywhere include("../src/RKHS/RKHS_helpers.jl")

@everywhere include("../src/flows/SDE_adaptive_kernel.jl")

@everywhere include("../src/GPODE/proposal.jl")
@everywhere include("../src/GPODE/GPODE_engine.jl")
@everywhere include("../src/GPODE/models/FHN.jl")
@everywhere include("../src/GPODE/DE_fit.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)


D_θ = 3

u0 = [-1.0; 1.0]
tspan = (0.0, 20.0)

γ_oracle = 3.0
α_oracle = 0.2
β_oracle = 0.2
p_oracle = [γ_oracle; α_oracle; β_oracle]

#

sol = setupFHN(u0, tspan, p_oracle)


### get data.
N_obs = 15 #30 #15
time_stamp = LinRange(0.1, tspan[end], N_obs)
time_stamp_vec = collect(time_stamp)

σ_y_oracle = 0.25

y_array = Vector{Vector{Float64}}(undef, 3)
y_array[1] = collect( sol(time_stamp[i])[1] + σ_y_oracle for i = 1:N_obs )
y_array[2] = collect( sol(time_stamp[i])[2] + σ_y_oracle for i = 1:N_obs )

y1 = y_array[1]
y2 = y_array[2]

# ### fit GP
# X = collect( [time_stamp_vec[i]] for i = 1:N_obs ) # GP fits time, a 1-D input.
#
# θ_oracle = AdaptiveRKHS.GaussianKernel1DType(1.0/10)
# σ²_oracle = 1e-7
# u_GP_array = fitDEsolutionGP(θ_oracle, σ²_oracle, X, y_array)



### visualize.
Nq = 500
xq_range = LinRange(tspan[1], tspan[end], Nq)
xq = collect( [ xq_range[i] ] for i = 1:length(xq_range) )

x1_display = collect( sol(xq_range[i])[1] for i = 1:length(xq_range) )
x2_display = collect( sol(xq_range[i])[2] for i = 1:length(xq_range) )


title_string = Printf.@sprintf("FHN dim 1: V(t)")
PyPlot.figure(fig_num)
fig_num += 1
PyPlot.plot(xq_range, x1_display, label = "x1")
PyPlot.plot(time_stamp, y1, ".", label = "y1")
#PyPlot.plot(xq, u_GP_array[1].(xq), label = "u_GP 1")
PyPlot.title(title_string)
PyPlot.xlabel("Time")
PyPlot.ylabel("Volts")
PyPlot.legend()

title_string = Printf.@sprintf("FHN dim 2: R(t)")
PyPlot.figure(fig_num)
fig_num += 1
PyPlot.plot(xq_range, x2_display, label = "x2")
PyPlot.plot(time_stamp, y2, ".", label = "y2")
#PyPlot.plot(xq, u_GP_array[2].(xq), label = "u_GP 2")
PyPlot.title(title_string)
PyPlot.xlabel("Time")
PyPlot.ylabel("Recovery")
PyPlot.legend()




##### tensorial derivatives of order 2.

𝓧 = time_stamp_vec
t_dummy = 1.0

θ = p_oracle
#∂FHN_∂θ = collect( Vector{Float64}(undef, 3) for i = 1:2 )
#eval∂FHNwrtp!( ∂FHN_∂θ, sol(𝓧[1]), θ, 𝓧[1] )


f_𝓧_θ = zeros(Float64, 2, N_obs)
updatef𝓧θ!(f_𝓧_θ, 𝓧, sol)

ψ_eval = zeros(Float64, 2, N_obs)
ψFHN!(ψ_eval, θ, 𝓧, f_𝓧_θ)

# I am here.
∂ψ_∂θ = collect( collect( Vector{Float64}(undef, 3) for m = 1:2 ) for n = 1:N_obs )
∂ψwrt∂θFHN!(∂ψ_∂θ, θ, 𝓧, f_𝓧_θ)

D = 3
v = Vector{Float64}(undef, 6)
@time Utilities.writesymmetric!(v, 2, 3, 3, 1.23)

∂2ψ_∂θ2 = collect( collect( Vector{Float64}(undef, 6) for m = 1:2 ) for n = 1:N_obs )
#eval∂2FHNwrtp!(∂2ψ_∂θ2[1], sol(𝓧[1]), θ, 𝓧[1])
∂2ψwrt∂θ2FHN!(∂2ψ_∂θ2, θ, 𝓧, f_𝓧_θ)

j = 2
H_j_set = getHmatrix(j, ∂2ψ_∂θ2, D)

# m = 1, N = 1. Verify second derivative.
A = packagesym(∂2ψ_∂θ2[1][2], D)
display(A)

#ψ = pp->ψFHNfull(𝓧, u0, tspan, pp)
ψ = pp->ψFHN2stage(𝓧, sol, pp)

println("verify ψ:")
println(norm(ψ_eval - ψ(θ))) # should be zero.
println()

H_j_set_ND = getHmatrixND(ψ, 2, D, N_obs, θ)
