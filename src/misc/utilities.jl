
function evalGaussiancopula(  x,
                            R::Matrix{T},
                            marginal_dists) where T <: Real
    D = length(x)

    PDF_eval_x = collect( Distributions.pdf(marginal_dists[d], x[d]) for d = 1:D )
    probit_CDF_eval_x = collect( probit( Distributions.cdf(marginal_dists[d], x[d]) ) for d = 1:D )
    CDF_eval_x = collect( Distributions.cdf(marginal_dists[d], x[d]) for d = 1:D )


    #println(probit_CDF_eval_x)
    out = Gaussiancopula(probit_CDF_eval_x,R)*prod(PDF_eval_x)

    if isfinite(out) != true
        println(prod(PDF_eval_x))
        println(probit_CDF_eval_x)
        println(CDF_eval_x)
        println(x)
    end
    @assert isfinite(out) == true
    return out
end


#### lorentz GP



###


# M is multiple for envelop.
# N is number of proposal draws.
# uniform proposal.
function rejectionsampler(pdf::Function,
                            limit_a::Vector{T},
                            limit_b::Vector{T},
                            N::Int,
                            M::T) where T <: Real
    #
    D = length(limit_a)
    @assert length(limit_b) == D


    # set up normalizing constant for the proposal.
    q_x = one(T)/prod( limit_b[d]-limit_a[d] for d = 1:D )

    # draw.
    X = Vector{Vector{T}}(undef, N)

    k = 0
    for n = 1:N
        xp = collect( Utilities.convertcompactdomain(rand(), 0.0, 1.0, limit_a[d], limit_b[d]) for d = 1:D )

        u = rand()
        if u < pdf(xp)/(M*q_x)
            k += 1

            X[k] = xp
        end
    end
    resize!(X, k)

    return X
end



function batcharraygivenbatchsize(N::Int, batch_size::Int)::Vector{Int}

    N_batches = cld(N,batch_size)

    N_for_each_batch = Vector{Int}(undef, N_batches)
    fill!(N_for_each_batch, batch_size)

    if mod(N,batch_size) != 0
        N_for_each_batch[end] = mod(N, batch_size)
    end

    return N_for_each_batch
end

# return in BigFloat
function RKHSquerytopdf(c::Vector{T},
                        X::Vector{Vector{T}},
                        a_SqExp::T,
                        ϕ::Function) where T <: Real
    #
    σ = 1/sqrt(2*a_SqExp)
    N = length(X)

    # mixture weights.
    C = sum(c)
    ln_w = collect( log(c[n]) -log(C) for n = 1:N )

    Σ = Matrix{T}(LinearAlgebra.I, D+1, D+1) .* σ^2

    ln_pdf = xx->evaladaptivequerylnpdf(xx, Σ, ln_w, X, ϕ)

    return ln_pdf
end



function runGFISonadaptivekernel(z::Vector{T},
                            ϕ::Function,
                            γ::T,
                            N_particles::Int,
                            N_batches::Int,
                            σ::T,
                            P_0::Matrix{T},
                            R::Matrix{T},
                            ψ::Function,
                            get𝐻func::Function,
                            get∂𝐻tfunc::Function,
                            N_discretizations::Int) where T <: Real

    # set up prior.
    D_x = length(z)

    drawxfunc = xx->Utilities.drawnormal(z, σ)
    m_0 = z
    P_0 = Matrix{T}(LinearAlgebra.I, D_x, D_x) .* σ^2

    # set up observation.
    y = [ ϕ(z) ]

    ln_prior_pdf_func = xx->evallnMVN(xx, m_0, P_0)
    ln_likelihood_func = xx->evallnMVNlikelihood(xx, y, m_0, P_0, ψ, R)

    ## traverse SDE.
    return paralleltraverseSDEs(drawxfunc,
                                N_discretizations,
                                γ,
                                m_0,
                                P_0,
                                ψ,
                                R,
                                y,
                                get𝐻func,
                                get∂𝐻tfunc,
                                ln_prior_pdf_func,
                                ln_likelihood_func,
                                N_particles,
                                N_batches)

end

function setupGFISadaptivekernel(   D_x::Int,
                                    ϕ::Function,
                                    dϕ::Function,
                                    d2ϕ::Function,
                                    N_discretizations::Int,
                                    a_SqExp::T) where T <: Real

    σ = 1/sqrt(2*a_SqExp)
    #σ = 1/sqrt(a_SqExp)

    P_0 = Matrix{T}(LinearAlgebra.I, D_x, D_x) .* σ^2

    # set up likelihood.
    R = Matrix{T}(undef,1,1)
    R[1,1] = σ^2 # single warpmap, single dimension for each y.

    ψ = xx->[ ϕ(xx) ]
    #get𝐻func = xx->convertcolvectorowvec(dϕ(xx))
    #get∂𝐻tfunc = xx->convertmatrixtonestedcolmats(d2ϕ(xx))
    D_y = 1
    get𝐻func = xx->Calculus.jacobian(ψ, xx, :central)
    hessian_funcs = gethessianfuncsND(ψ, D_y)
    get∂𝐻tfunc = xx->compute∂𝐻tover∂x(hessian_funcs, xx, D_y)

    # set up SDE.
    λ_array, Bλ_array = drawBrownianmotiontrajectorieswithoutstart(N_discretizations, D_x)

    return  σ, P_0, R, ψ, get𝐻func, get∂𝐻tfunc, λ_array, Bλ_array
end

function mergeclosepositions(X_in::Vector{Vector{T}}, ϵ::T)::Vector{Vector{T}} where T <: Real
    X = copy(X_in)
    N = length(X)
    D = length(X[1])
    n = 1

    while n < length(X)
        x = X[n]

        # false for merging.
        indicators = collect( norm(X[j]-x) > ϵ for j = 1:length(X) )

        if sum(indicators) < length(X)-1
            centroid = zeros(T,D)
            for i = 1:length(indicators)
                if !indicators[i]
                    centroid += X[i]
                end
            end
            X[n] = centroid ./ sum(indicators)

            X = X[indicators]
            n = 1
        else
            n += 1
        end
    end

    return X
end




function convertcolvectorowvec(x::Vector{T})::Matrix{T} where T <: Real
    out = Matrix{T}(undef, 1, length(x) )
    for i = 1:length(x)
        out[i] = x[i]
    end

    return out
end

function convertcolvectocolmat(x::Vector{T})::Matrix{T} where T <: Real
    out = Matrix{T}(undef, length(x), 1 )
    for i = 1:length(x)
        out[i] = x[i]
    end

    return out
end

function convertmatrixtonestedcolmats(A::Matrix{T})::Vector{Matrix{T}} where T <: Real
    out = Vector{Matrix{T}}(undef, size(A,1) )
    for i = 1:size(A,1)
        out[i] = convertcolvectocolmat(vec(A[i,:]))
    end

    return out
end


function packageupMambaMCMC(values::Array{T,Dp1})::Vector{Vector{T}} where {T,Dp1}
    N_per_chain = size(values,1)
    N_chains = size(values,Dp1)
    N = N_per_chain*N_chains

    D = Dp1 - 1

    out = Vector{Vector{T}}(undef, N)

    n = 1
    for j = 1:N_chains
        for i = 1:N_per_chain
            out[n] = vec(values[i, 1:D, j])
            n += 1
        end
    end

    return out
end

###
function evallnMVNlikelihood( x,
                            y::Vector{T},
                            m_x,
                            S_x,
                            ψ::Function,
                            S_y)::T where T <: Real

    #
    #term1 = evallnMVN(x, m_x, S_x)
    term1 = 0.0
    term2 = evallnMVN(y, ψ(x), S_y)

    return term1 + term2
end


function evallnMVN(x, μ::Vector{T}, Σ::Matrix{T})::T where T <: Real
    D = length(x)

    r = x-μ
    term1 = -0.5*dot(r,Σ\r)
    term2 = -D/2*log(2*π) -0.5*logdet(Σ)

    return term1 + term2
end

##### legacy?


function getguidesgmm(dist_array, N_b, N_a)
    K = length(dist_array)

    a_guide = Vector{LinRange}(undef,0)
    b_guide = Vector{LinRange}(undef,0)

    for k = 1:K
        μ = dist_array[k].μ
        σ = dist_array[k].σ
        push!(b_guide, LinRange(μ-2*σ,μ+2*σ,N_b))
        push!(a_guide, LinRange(0.01, 10, N_a))
    end

    return a_guide, b_guide
end

function initializeatomsgmm(a_guide,
                        b_guide, N)

    #
    K = length(a_guide)
    @assert length(b_guide) == K

    a_array = ones(Float64,N)
    b_array = Vector{Float64}(undef,N)
    counter = 1
    for r = 1:K
        for a_b in Iterators.product(a_guide[r],b_guide[r])

            b_array[counter] = a_b[2]
            a_array[counter] = a_b[1]

            counter += 1
        end
    end
    @assert counter-1 == N

    return a_array, b_array
end

#####

function initializeatoms(a_guide,
                        b_guide, N)

    #
    a_array = ones(Float64,N)
    b_array = Vector{Float64}(undef,N)
    counter = 1
    for r = 1:3
        for a_b in Iterators.product(a_guide[r],b_guide[r])

            b_array[counter] = a_b[2]
            a_array[counter] = a_b[1]

            counter += 1
        end
    end
    @assert counter-1 == N

    return a_array, b_array
end

function getcolorarray(N::Int)
    X = ["#00ffff";
        "#f0ffff";
        "#f5f5dc";
        "#000000";
        "#0000ff";
        "#a52a2a";
        "#00ffff";
        "#00008b";
        "#008b8b";
        "#a9a9a9";
        "#006400";
        "#bdb76b";
        "#8b008b";
        "#556b2f";
        "#ff8c00";
        "#9932cc";
        "#8b0000";
        "#e9967a";
        "#9400d3";
        "#ff00ff";
        "#ffd700";
        "#008000";
        "#4b0082";
        "#f0e68c";
        "#add8e6";
        "#e0ffff";
        "#90ee90";
        "#d3d3d3";
        "#ffb6c1";
        "#ffffe0";
        "#00ff00";
        "#ff00ff";
        "#800000";
        "#000080";
        "#808000";
        "#ffa500";
        "#ffc0cb";
        "#800080";
        "#800080";
        "#ff0000";
        "#c0c0c0";
        "#ffffff";
        "#ffff00"]

        #     aqua: "#00ffff",
        #     azure: "#f0ffff",
        #     beige: "#f5f5dc",
        #     black: "#000000",
        #     blue: "#0000ff",
        #     brown: "#a52a2a",
        #     cyan: "#00ffff",
        #     darkblue: "#00008b",
        #     darkcyan: "#008b8b",
        #     darkgrey: "#a9a9a9",
        #     darkgreen: "#006400",
        #     darkkhaki: "#bdb76b",
        #     darkmagenta: "#8b008b",
        #     darkolivegreen: "#556b2f",
        #     darkorange: "#ff8c00",
        #     darkorchid: "#9932cc",
        #     darkred: "#8b0000",
        #     darksalmon: "#e9967a",
        #     darkviolet: "#9400d3",
        #     fuchsia: "#ff00ff",
        #     gold: "#ffd700",
        #     green: "#008000",
        #     indigo: "#4b0082",
        #     khaki: "#f0e68c",
        #     lightblue: "#add8e6",
        #     lightcyan: "#e0ffff",
        #     lightgreen: "#90ee90",
        #     lightgrey: "#d3d3d3",
        #     lightpink: "#ffb6c1",
        #     lightyellow: "#ffffe0",
        #     lime: "#00ff00",
        #     magenta: "#ff00ff",
        #     maroon: "#800000",
        #     navy: "#000080",
        #     olive: "#808000",
        #     orange: "#ffa500",
        #     pink: "#ffc0cb",
        #     purple: "#800080",
        #     violet: "#800080",
        #     red: "#ff0000",
        #     silver: "#c0c0c0",
        #     white: "#ffffff",
        #     yellow: "#ffff00"
        # };

    return X[1:N]
end


function solvefora(y, x, b, c)

    tmp = y/c

    numerator = (1-tmp^2)*(x-b)^2
    denominator = tmp^2

    return numerator/denominator
end
# # test code
# a = abs(randn())
# b = randn()
# x = randn()
# y = KRTransportMap.algebraicatom(x,a,b)
# a_rec = solvefora(y,x,b, 1.0)

# c and a must both be positive.
function ℝtointerval(x::T, min_value::T, max_value::T)::T where T <: Real
    h = max_value - min_value
    c = h/2
    a = 25.0

    return c*x/sqrt(a+x^2) +c + min_value
end

# c and a must both be positive.
function intervaltoℝ(y::T, min_value::T, max_value::T)::T where T <: Real
    h = max_value - min_value
    c = h/2
    a = 25.0

    numerator = a*(y-c-min_value)^2
    denominator = c^2 - (y-c-min_value)^2
    x = sqrt(numerator/denominator)

    # ln_numerator = log(a) + 2*log(y-c-min_value)
    # ln_denominator = cannot do logsumexp for negative numbers..
    # x = sqrt(exp(ln_numerator-ln_denominator))

    return sign(y-c-min_value)*x
end
# # test code.
# min_value = 0.75
# max_value = 10.25
# x = 99999.9 # randn()
# u = ℝtointerval(x, min_value, max_value)
# x_rec = intervaltoℝ(u, min_value, max_value)
# # u should be near max_value.
# # to do: there is some round-off error in x_rec. figure out a log-space method for this.


# x here is a real-valued variable.
# A needs to be posdef.
function computemsqrtderivatives(   A::Matrix{T},
                                    ∂A_∂x::Matrix{T})::Matrix{T} where T <: Real

    #A_sqrt = naivesqrtpsdmatrix(A)
    A_sqrt = real.(LinearAlgebra.sqrt(A))
    ∂Asqrt_∂x = LinearAlgebra.sylvester(A_sqrt, A_sqrt, -∂A_∂x)

    return ∂Asqrt_∂x
end

# x here is a real-valued multivariate variable.
function computemsqrtderivatives(   A::Matrix{T},
                                    ∂A_∂x_array::Vector{Matrix{T}})::Vector{Matrix{T}} where T <: Real
    N = length(∂A_∂x_array)

    #A_sqrt = naivesqrtpsdmatrix(A)
    A_sqrt = real.(LinearAlgebra.sqrt(A))
    ∂Asqrt_∂x = collect( LinearAlgebra.sylvester(A_sqrt, A_sqrt, -∂A_∂x_array[j]) for j = 1:N )

    return ∂Asqrt_∂x
end


# w_array is assumed to be normalized such that it sums to 1.
function getcovmatfromparticles(x_array::Vector{Vector{T}},
                                m::Vector{T},
                                w_array::Vector{T}) where T <: Real

    @assert length(w_array) == length(x_array)

    D = length(x_array[1])
    @assert length(m) == D

    C = zeros(T,D,D)
    for i = 1:D
        for j = 1:D

            # add the contribution of each particle.
            for n = 1:length(x_array)
                contribution = (x_array[n][i] - m[i]) *(x_array[n][j] - m[j]) *w_array[n]
                C[i,j] += contribution
            end

        end
    end

    return C
end


function GaussianFlowSimpleBuffersType(D_y::Int, D_x::Int, val::T) where T <: Real

    return GaussianFlowSimpleBuffersType(
        ψ_eval = Vector{T}(undef, D_y),

        𝐻 = Matrix{T}(undef, D_y, D_x),
        ∂2ψ_eval = Vector{Vector{T}}(undef, D_y),

        𝑦 = Vector{T}(undef, D_y),

        # moments.
        𝑚_a = Vector{T}(undef, D_x),
        𝑃_a = Matrix{T}(undef, D_x, D_x),
        𝑚_b = Vector{T}(undef, D_x),
        𝑃_b = Matrix{T}(undef, D_x, D_x),

        # derivatives.
        ∂𝑚_a_∂x = Matrix{T}(undef, D_x, D_x),
        ∂𝑚_b_∂x = Matrix{T}(undef, D_x, D_x),

        ∂𝑃_b_∂x = Vector{Matrix{T}}(undef, D_x),
        ∂𝑃_b_inv𝑃_a_∂x = Vector{Matrix{T}}(undef, D_x),

        ∂𝑃_b_sqrt_∂x = Vector{Matrix{T}}(undef, D_x),
        ∂𝑃_b_inv𝑃_a_sqrt_∂x = Vector{Matrix{T}}(undef, D_x) )

end


function getHtmatrix(j::Int,
                    ∂2ψ_∂θ2::Vector{Vector{T}},
                    D::Int) where T

    #
    N = length(∂2ψ_∂θ2)

    H_j = Matrix{T}(undef, D, N)
    fill!(H_j, Inf) # debug.

    for k = 1:D
        for i = 1:N
            H_j[k,i] = Utilities.readsymmetric(
                    j, k, D, ∂2ψ_∂θ2[i])
        end
    end

    return H_j
end
