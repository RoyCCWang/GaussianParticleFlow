# test exact Gaussian flow.


using Distributed

import HCubature
import Utilities
import Printf
import PyPlot
import Random

using LinearAlgebra
import Interpolations

using FFTW

import Statistics

import Distributions

import Calculus

import ForwardDiff
import StatsFuns

using Test
using BenchmarkTools

import VisualizationTools

#import Seaborn
include("../src/misc/declarations.jl")

include("../src/SDE/Brownian.jl")
include("../src/flows/approx_flow.jl")
include("../tests/routines/sde.jl")
include("../src/flows/moments.jl")
include("../src/flows/derivatives.jl")
include("../src/misc/utilities.jl")
include("../src/misc/utilities2.jl")


### adaptive kernel.
include("../fresh/adaptive_kernel/scalar.jl")


Random.seed!(25)

D_x = 6

# test ln pdf.
σ² = rand()
σ²_vec = [ σ² ]
R = ones(Float64, 1, 1)
R[1] = σ²

P_0 = diagm( σ² .* ones(D_x) )
m_0 = randn(D_x)

A = randn(D_x, D_x)
ψ = xx->[ sinc(dot(xx,A*xx)) ]
y = randn(1)

prior_dist = Distributions.MvNormal(m_0, P_0)

ln_prior_pdf_func = xx->Distributions.logpdf(prior_dist, xx)
ln_likelihood_func = xx->evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R)

f = xx->evallnproductnormal(xx, m_0, σ²_vec)
g = xx->evallnproductnormallikelihood(xx, y, ψ, σ²_vec)


x0 = randn(D_x)
println("f(x0) - ln_prior_pdf_func(x0) = ", f(x0) - ln_prior_pdf_func(x0))
println("g(x0) - ln_likelihood_func(x0) = ", g(x0) - ln_likelihood_func(x0))
println()
