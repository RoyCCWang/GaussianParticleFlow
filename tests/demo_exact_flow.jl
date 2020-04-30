# test exact Gaussian flow.



import FileIO
import Printf
import PyPlot
import Random

using LinearAlgebra
import Interpolations

using FFTW

import Statistics

import Distributions
import Utilities

PyPlot.close("all")
fig_num = 1


include("../src/SDE/Brownian.jl")
include("../src/flows/exact_flow.jl")
include("routines/sde.jl")

Random.seed!(25)

N_particles = 1000
D_x = 2
D_y = 2

m_0 = randn(Float64,D_x)
S = randn(Float64,D_x,D_x)
P_0 = S'*S
prior_dist = Distributions.MvNormal(m_0,P_0)

m_true = randn(Float64,D_x)
S = randn(Float64,D_x,D_x)
Σ_true = S'*S
true_dist_x = Distributions.MvNormal(m_true,Σ_true)



H = randn(D_y,D_x)
S = randn(Float64,D_y,D_y)
R = S'*S

likelihood_func = xx->Distributions.MvNormal(H*xx,R)

# generate observation.
x_generating = rand(true_dist_x)
y = rand(likelihood_func(x_generating))

##### flow.

### set up SDE.
N_discretizations = 1000
γ = 0.1

##### using appendix D.

# set up Brownian motion.
λ_array, Bλ_array = drawBrownianmotiontrajectorieswithstart(N_discretizations, D_x)

### traverse the SDE for each particle.

# draw samples from the prior.
x_array = collect( rand(prior_dist) for i = 1:N_particles )

# traverse the SDE solution for each particle.
xp_array = Vector{Vector{Float64}}(undef,N_particles)
for n = 1:N_particles
    x0 = x_array[n]

    𝑥 = traversesdepathexactflow(x0, λ_array, Bλ_array, γ, m_0, P_0, H, R, y)
    xp_array[n] = 𝑥[end]
end

##### using equation 15.
# draw samples from the prior.
λ_array, Bλ_array = drawBrownianmotiontrajectorieswithoutstart(N_discretizations, D_x)
x_array2 = collect( rand(prior_dist) for i = 1:N_particles )

# traverse the SDE solution for each particle.
xp_array2 = Vector{Vector{Float64}}(undef,N_particles)
for n = 1:N_particles
    x0 = x_array2[n]

    𝑥 = traversesdepathexactflowviaeqn15(x0, λ_array, Bλ_array, γ, m_0, P_0, H, R, y)
    xp_array2[n] = 𝑥[end]
end

### evaluation.

# true posterior.
m_p, Σ_p = getexactmoments(1.0, m_0, P_0, H, R, y)

# sample covmat:
m_s = Statistics.mean(xp_array)
Q = 1/(N_particles-1) * sum( (xp_array[n]-m_s)*(xp_array[n]-m_s)' for n = 1:N_particles)

m_s2 = Statistics.mean(xp_array2)
Q2 = 1/(N_particles-1) * sum( (xp_array2[n]-m_s2)*(xp_array2[n]-m_s2)' for n = 1:N_particles)

println("x, generating = ", x_generating)
println()

println("ground truth posterior parameters:")
println("m_p = ", m_p)
println("Σ_p = ", Σ_p)
println()

println("Appendix D's formula: MLE posterior parameters:")
println("m_s = ", m_s)
println("Q = ", Q)
println()

println("Equation 15: MLE posterior parameters:")
println("m_s = ", m_s2)
println("Q = ", Q2)
println()
