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

include("../tests/routines/simulation_tools.jl")
include("../src/diagnostics/functionals.jl")
include("../src/misc/utilities.jl")

Random.seed!(25)

# #y := f(x)
# z := g(y)
# f = xx->sin(norm(xx)^2)
# g = yy->(yy^3+yy^2)
#
# h = xx->g(f(xx))

# D = 2 # dimension of input.
# h = exampleψfunc2Dto2D1
# ψ = exampleψfunc2Dto2D1Zygote

D = 4
h = exampleψfunc4Dto2D1
ψ = exampleψfunc4Dto2D1Zygote

dh_ND = xx->Calculus.jacobian(h, xx, :central)

dh_AD = xx->ForwardDiff.jacobian(h,xx)

f = xx->ψ(xx)[1]
dh_ZD = xx->Zygote.gradient(f,xx)[1]

x0 = randn(D)

println("ND timing")
@btime dh_ND(x0)
@btime dh_ND(x0)
# 896 ns.

println("ZD timing")
@btime dh_ZD(x0)
@btime dh_ZD(x0)
# 43 μs. very slow.

println("AD timing")
@btime dh_AD(x0)
@btime dh_AD(x0)
# 565 ns.

#f = xx->ψ([xx;x0[2:end]])[1]
#dh_ZD = xx->Zygote.gradient(f,xx)[1]
#@btime dh_ZD(x0[1])

@assert 444==3


d2h_AD = xx->ForwardDiff.hessian(dh_AD,xx)
d2h_ZD = xx->Zygote.hessian(dh_ZD,xx)

# currently needs to allocate new variable.
# I am here.
# do one variable at a time f(x[i]' rest consts'). This avoids array allocation.
# use persist to update the consts for each function.

println("AD: dh(x0) = ", dh_x0_AD)
println("ZD: dh(x0) = ", dh_x0_ZD)
println()


d2h_x0_AD = d2h_AD(x0) # ForwardDiff
d2h_x0_ZD = d2h_ZD(x0) # Zygote

println("AD: d2h(x0) = ", d2h_x0_AD)
println("ZD: d2h(x0) = ", d2h_x0_ZD)
println()
