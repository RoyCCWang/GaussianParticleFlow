# demonstrative example.

# using Distributed
#
# @everywhere using LinearAlgebra
# @everywhere import Random
# @everywhere import Distributions
# using Mamba


using LinearAlgebra
import Random
import Distributions
using Mamba

Random.seed!(25)

D = 2

m = randn(Float64, D)
P = randn(Float64, D, D)
P = P'*P

dist_x = Distributions.MvNormal(m, P)
