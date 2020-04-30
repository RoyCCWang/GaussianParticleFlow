
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
#@everywhere import stickyHDPHMM

@everywhere import GSL
@everywhere import StatsFuns

#@everywhere using Mamba
@everywhere import StaticArrays
@everywhere import DifferentialEquations

@everywhere include("../src/misc/declarations.jl")

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

@everywhere include("../src/misc/utilities2.jl")

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

D = 3
j = 2
M = 2

# step 1 of 2-step.
f_𝓧_θ = zeros(Float64, 2, N_obs)
updatef𝓧θ!(f_𝓧_θ, 𝓧, sol)

# psi.
ψ_eval_mat = zeros(Float64, 2, N_obs)
ψFHN!(ψ_eval_mat, θ, 𝓧, f_𝓧_θ)
ψ_eval = evalψFHN!(ψ_eval_mat, θ, 𝓧, f_𝓧_θ)

# first-order derivatives.
∂ψ_∂θ = collect( collect( Vector{Float64}(undef, 3) for m = 1:2 ) for n = 1:N_obs )
∂ψwrt∂θFHN!(∂ψ_∂θ, θ, 𝓧, f_𝓧_θ)

# second-order derivatives.
∂2ψ_∂θ2 = collect( collect( Vector{Float64}(undef, 6) for m = 1:2 ) for n = 1:N_obs )
#eval∂2FHNwrtp!(∂2ψ_∂θ2[1], sol(𝓧[1]), θ, 𝓧[1])
∂2ψwrt∂θ2FHN!(∂2ψ_∂θ2, θ, 𝓧, f_𝓧_θ)


H_j_set = getHmatrix(j, ∂2ψ_∂θ2, D)


N = 2
M = 3
R_set = collect( randn(N,N) for m = 1:M )
R = Utilities.makeblockdiagonalmatrix( R_set )

H_set = collect( randn(N,D) for m = 1:M )
H = [ H_set[1]; H_set[2]; H_set[3] ]
H2 = verticalblocktomatrix(H_set)

out = H'*R*H
out2 = sum( H_set[m]'*R_set[m]*H_set[m] for m = 1:M )
println("discrepancy between out and out2 is ", norm(out-out2))
println()

x = randn(D)
out = H*x
out2 = evalmatrixvectormultiply(H_set,x)


@assert 1==2

##### set up inference distribution.

### get finite-dimensional distribution.
a_SqExp = 50.0
θ_GP = AdaptiveRKHS.GaussianKernel1DType(a_SqExp)

# latent states.
Σ_x = AdaptiveRKHS.constructkernelmatrix(X, θ_GP)
m_x = zeros(Float64, N_obs)
K = Σ_x

# parameters.
m_θ = zeros(Float64, D_θ)
Σ_θ = Utilities.generaterandomposdefmat(D_θ)

# observables.
σ_y = 0.1 #1.68
Σ_y = (σ_y^2) .* Matrix{Float64}(LinearAlgebra.I, N_obs, N_obs)

# derivatives of latent states.
t_dummy = 1.0
f = (xx,θθ)->collect( evalFHN([xx[i]; 1.23; 1.34],θθ, t_dummy)[1] for i = 1:length(xx) )

σ_x_dot = 0.15
A_mat = Utilities.generaterandomposdefmat(N_obs)
Σ_x_dot = A_mat + (σ_x_dot^2) .* Matrix{Float64}(LinearAlgebra.I, N_obs, N_obs)
D_mat = randn(N_obs, N_obs)



### construct distributions.
ln_p_θ = θθ->evallnMVN(θθ, m_θ, Σ_θ)
ln_p_x = xx->evallnMVN(xx, m_x, Σ_x)

ln_p_x_dot = (xx,θθ)->evallnMVN(D_mat*xx,
                        f(xx,θθ), Σ_x_dot)
#
ln_p_y = (yy,xx)->evallnMVN(yy, xx, Σ_y)
ln_p_tilde = (xx,θθ)->(ln_p_θ(θθ) + ln_p_x(xx) +
                    ln_p_y(y1,xx) +
                    ln_p_x_dot(xx,θθ))
p_tilde = xx->exp(ln_p_tilde(xx[1:N_obs], xx[N_obs+1:end]))
