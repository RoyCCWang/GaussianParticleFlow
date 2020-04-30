# demonstrative example.

using Distributed

@everywhere using LinearAlgebra
@everywhere import Random
@everywhere import Distributions
@everywhere using Mamba

Random.seed!(25)

function exampleψfunc2Dto2D1(x::Vector{T})::Vector{T} where T <: Real
    @assert length(x) == 2

    out = Vector{T}(undef,2)

    out[1] = cos(x[1])*x[2] + x[2]^2 -x[1]
    out[2] = out[1]*x[1]^3

    return out
end

### model parameters.

D_x = 2
D_y = 2
ψ = exampleψfunc2Dto2D1

σ = 0.02
R = diagm( 0 => collect(σ for d = 1:D_y) )

### generate data.

# oracle distribution.
m_true = randn(Float64, D_x)
P_true = randn(Float64, D_x, D_x)
P_true = P_true'*P_true

true_dist_x = Distributions.MvNormal(m_true, P_true)

@assert 1==2

# draw a realization for the latent variable.
x_generating = rand(true_dist_x)

# use it to generate data.
true_dist_y = Distributions.MvNormal(ψ(x_generating),R)
y = rand(true_dist_y)

### MCMC.

model = Model(

  y = Stochastic(1,
    (mu) ->  MvNormal(mu, R),
    false
  ),

  mu = Logical(1,
    x->ψ(x),
    false
  ),

  x = Stochastic(1,
    () -> MvNormal(m_0, P_0)
  )

)

scheme1 = [NUTS(:x)]
#
setsamplers!(model, scheme1)

line = Dict{Symbol, Any}(
  :y => y
)

N_chains = 3
inits = [
  Dict{Symbol, Any}(
    :y => line[:y],
    :x => zeros(Float64,length(m_0))
  )
for i in 1:N_chains
]

setsamplers!(model, scheme1)
println("Starting MCMC simulation.")
@time sim1 = mcmc(model, line, inits, 100000, burnin=250, thin=2, chains=N_chains)

describe(sim1)
