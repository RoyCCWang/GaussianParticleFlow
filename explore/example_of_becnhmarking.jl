using Zygote, BenchmarkTools
using Zygote:@adjoint

ρfun(r) = exp(-3*(r+1))
dρfun(r) = -3 * ρfun(r)

function eam(x)
   ρ = sum(ρfun, x)
   return ρ^2 + ρ^3 + 0.25 * ρ^4
end

function deam!(g, x)
   fill!(g, 0.0)
   N = length(x)
   ρ = sum(ρfun, x)
   dF = 2*ρ + 3*ρ^2 + ρ^3
   for n = 1:N
      g[n] = dρfun(x[n]) * dF
   end
   return g
end

# --------------------------------------------------
# Using the workaround from
#   https://github.com/FluxML/Zygote.jl/issues/292
function sum2(op,arr)
	return sum(op,arr)
end
function sum2adj( Δ, op, arr )
	n = length(arr)
	g = x->Δ*Zygote.gradient(op,x)[1]
	return ( nothing, map(g,arr))
end
@adjoint function sum2(op,arr)
	return sum2(op,arr),Δ->sum2adj(Δ,op,arr)
end
function eam2(x)
   ρ = sum2(ρfun, x)
   return ρ^2 + ρ^3 + 0.25 * ρ^4
end
# --------------------------------------------------


# benchmark script
# ----------------

deam(x) = deam!(zeros(length(x)), x)
zeam(x) = gradient( eam, x )[1]
zeam2(x) = gradient( eam2, x )[1]

x = rand(100)
g = rand(100)
@show sqrt(sum((zeam(x) - deam(x)).^2))

@btime eam($x);
@btime deam($x);
@btime deam!($g, $x);
@btime zeam($x);
@btime zeam2($x);
