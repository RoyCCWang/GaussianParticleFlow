using Distributed

using LinearAlgebra
import Random
import Distributions

import Utilities

import Zygote
import ForwardDiff

import HCubature

using BenchmarkTools

import Calculus
import FiniteDiff

include("../tests/routines/simulation_tools.jl")
include("../src/diagnostics/functionals.jl")
include("../src/misc/utilities.jl")

Random.seed!(25)

# D = 2 # dimension of input.
# h = exampleψfunc2Dto2D1

D_x = 4
D_y = 2
h = exampleψfunc4Dto2D1

println("First-order:")
println()

dh_ND = xx->Calculus.jacobian(h, xx, :central)

J = zeros(Float64, D_y, D_x)
dh_AD! = xx->ForwardDiff.jacobian!(J, h, xx)

x0 = randn(D_x)

println("ND timing")
@btime dh_ND(x0)
@btime dh_ND(x0)
# 896 ns.

println("AD timing")
@btime dh_AD!(x0)
@btime dh_AD!(x0)
# 565 ns.

dh_x0_ND = dh_ND(x0)

dh_AD!(x0)
dh_x0_AD = J

println("norm(dh_x0_AD-dh_x0_ND) = ", norm(dh_x0_AD-dh_x0_ND))
println()


##### second-order.
println("Second-order:")
println()

h_set = collect( xx->h(xx)[i] for i = 1:D_y )

d2h_ND = xx->collect( Calculus.hessian(h_set[i], xx) for i = 1:D_y )

J_set = collect( zeros(Float64, D_y, D_x) for i = 1:D_y )
d2h_AD = xx->ForwardDiff.jacobian(aa -> ForwardDiff.jacobian(h, aa), xx)
d2h_ND_nested = xx->FiniteDiff.finite_difference_jacobian(aa -> FiniteDiff.finite_difference_jacobian(h, aa), xx)

x0 = randn(D_x)

println("ND timing")
@btime d2h_ND(x0)
@btime d2h_ND(x0)
# 896 ns.

println("AD timing")
@btime d2h_AD(x0)
@btime d2h_AD(x0)
# 565 ns.

d2h_x0_ND = d2h_ND(x0)

#dh_AD!(x0)
d2h_x0_AD = d2h_AD(x0)
println("reshape timing")
@btime reshape(d2h_x0_AD,D_y,D_x,D_x)
@btime reshape(d2h_x0_AD,D_y,D_x,D_x)

d2h_x0_AD_array = reshape(d2h_x0_AD,D_y,D_x,D_x)

disscrepancy = sum( norm(d2h_x0_AD_array[i,:,:]-d2h_x0_ND[i]) for i = 1:D_y )
println("discrepancy between AD vs. ND for d2h(x0): ", disscrepancy)
println()

# are nested expressions reliable in FiniteDiff? No.
[d2h_ND_nested(x0)  d2h_x0_AD]
