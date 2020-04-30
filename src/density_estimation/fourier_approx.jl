# routines that approximates the forward and inverse Fourier integral.



# for f(x) = 1/sqrt(det(Σ)) * 1/sqrt(2*π)^D * exp(-0.5*dot(x,Σ*x))
function computecontinuousFTGaussian(   𝑓_nD::Array{Vector{T},D},
                                        Σ::Matrix{T}) where {T,D}
    sz = size(𝑓_nD)

    FT_f_nD = Array{T,D}(undef,sz)
    for 𝑖 in CartesianIndices(sz)
        FT_f_nD[𝑖] = exp(-2*π^2*dot(𝑓_nD[𝑖],Σ*𝑓_nD[𝑖]))
    end

    return FT_f_nD
end

function approximatecharfunc(   N_array::Vector{Int},
                                𝑓_ranges::Vector)
    #
    D = length(N_array)
    fs = collect( 𝑓_ranges[d][end]*2 for d = 1:D )
    ts = fs ./ (-2*π)

    # 𝑡_ranges = collect( LinRange(-ts[d]/2, ts[d]/2, N_array[d]) for d = 1:D )
    # 𝑡 = collect( [𝑡_ranges[1][i]; 𝑡_ranges[2][j]] for j = 1:N_array[2] for i = 1:N_array[1] )
    # 𝑡2 = collect( 𝑓[i] ./ (-2*π) for i = 1:length(𝑓) ) # dummy index for characteristic function evaluation.

    #𝑥_ranges = collect( LinRange(x_a, x_b, N_array[d]) for d = 1:D )
    #𝑋 = collect( [𝑥_ranges[1][i]; 𝑥_ranges[2][j]] for j = 1:N_array[2] for i = 1:N_array[1] )

    #return 𝑡, 𝑡2
    𝑥_ranges = collect( x_ranges[d] ./ (-2*π) for d = 1:D )

    return 𝑥_ranges, ts
end

# use Monte-carlo to evaluate 𝔼[exp(-im*2*π*dot(r,w))] with respect to p(x).
# dist is the distribution of p(x), in Distributions.jl data type.
function evalMCFT(X::Vector{Vector{T}}, w::Vector{T})::Complex{T} where T <: Real
    N = length(X)
    return sum( exp(-im*2*π*dot(X[n],w))/N for n = 1:N )
end

function getMCFTarray(  X::Vector{Vector{T}},
                        𝑓_nD,
                        ::Val{D})::Array{Complex{T},D} where {T,D}

    FT_f_MC_nD = Array{Complex{T},D}(undef,size(𝑓_nD))
    @inbounds for 𝑖 in CartesianIndices(size(𝑓_nD))
        FT_f_MC_nD[𝑖] = evalMCFT(X, 𝑓_nD[𝑖])
        #println(𝑖)
    end

    return FT_f_MC_nD
end

function evalMCFT1D(X::Vector{T}, w::T)::Complex{T} where T <: Real
    N = length(X)
    return sum( exp(-im*2*π*X[n]*w)/N for n = 1:N )
end

function getMCFTarray1D(  X::Vector{T}, 𝑓)::Vector{Complex{T}} where T <: Real

    return collect( evalMCFT1D(X, 𝑓[i]) for i = 1:length(𝑓) )
end

function getMCFTarray1D(X::Vector{T},
                        𝑓_array::LinRange{T},
                        ln_w_array::Vector{T})::Vector{Complex{T}} where T <: Real

    return collect( evalMCFT1D(X, 𝑓_array[i], ln_w_array) for i = 1:length(𝑓_array) )
end


function evalMCFT1D(X::Vector{T},
                        𝑓::T,
                        ln_w_array::Vector{T})::Complex{T} where T <: Real

    N = length(X)
    ln_W = StatsFuns.logsumexp(ln_w_array)

    ln_out::Vector{Complex{T}} = collect( ln_w_array[n] +
                -im*2*π*X[n]*𝑓 for n = 1:N )
                # exp(-im*2*π*X[n]*𝑓)

    # division by the sum of weights.
    out = exp( Utilities.logsumexp(ln_out) - ln_W)

    return out
end

function get𝑓ranges(τ::T, N_samples_per_dim, ::Val{1}) where {T,D}

    # set up bounds.
    x_a = -τ
    x_b = τ
    t0 = τ # delay in space units.

    # set up number of samples.
    N_samples = N_samples_per_dim # number of samples in the freq domain.

    # set up sampling locations for both spatial and frequency domains.
    x_ranges = LinRange(x_a, x_b, N_samples)
    X = collect( x_ranges )

    fs = 1/(x_ranges[2]-x_ranges[1]) # sampling frequency in Hz.

    𝑓_ranges = LinRange(-fs/2, fs/2 * (π-2*π/N_samples)/π, N_samples)
    𝑓 = collect( 𝑓_ranges )
    𝑓_nD = collect( [𝑓[i]] for i = 1:length(𝑓) )

    return x_ranges, X, fs, 𝑓_ranges, 𝑓, 𝑓_nD, [N_samples], t0, N_samples
end

function get𝑓ranges(τ::T, N_samples_per_dim, ::Val{2}) where T

    # set up bounds.
    x_a = -τ
    x_b = τ
    t0 = τ .* ones(T,D) # delay in space units.

    # set up number of samples.
    N_array = N_samples_per_dim .*ones(Int, D) # number of samples in the freq domain.
    N_samples = prod(N_array)

    # set up sampling locations for both spatial and frequency domains.
    x_ranges = collect( LinRange(x_a, x_b, N_array[d]) for d = 1:D )
    X = collect( [x_ranges[1][i]; x_ranges[2][j]] for j = 1:N_array[2] for i = 1:N_array[1] )

    fs = collect( 1/(x_ranges[d][2]-x_ranges[d][1]) for d = 1:D ) # sampling frequency in Hz.

    # 𝑓_ranges = collect( LinRange(-fs[d]/2, fs[d]/2, N_array[d]) for d = 1:D )
    # 𝑓 = collect( [𝑓_ranges[1][i]; 𝑓_ranges[2][j]] for j = 1:N_array[2] for i = 1:N_array[1] )
    # 𝑓_nD = reshape(𝑓, N_array...)
    𝑓_ranges = collect( LinRange(-fs[d]/2, fs[d]/2 * (π-2*π/N_array[d])/π, N_array[d]) for d = 1:D )
    𝑓 = collect( [𝑓_ranges[1][i]; 𝑓_ranges[2][j]] for j = 1:N_array[2] for i = 1:N_array[1] )
    𝑓_nD = reshape(𝑓, N_array...)

    return x_ranges, X, fs, 𝑓_ranges, 𝑓, 𝑓_nD, N_array, t0, N_samples
end

# continuous Fourier transform approx from DFT for a time-limited function f,
#   where g(t[d]) = 0 if abs(t[d]) > τ, for each dimension index d.
function approximatecontinuousFT(   f::Function,
                                    τ::T,
                                    ::Val{D},
                                    N_samples_per_dim::Int) where {T <: Real,D}


    x_ranges, X, fs, 𝑓_ranges, 𝑓, 𝑓_nD, N_array, t0, N_samples = get𝑓ranges(τ, N_samples_per_dim, Val(D))

    # sample function, and compute DFT from samples.
    f_evals = collect( f(X[i]) for i = 1:N_samples )
    f_evals_nD = reshape(f_evals, N_array...)
    DFT_f = fftshift(fft(f_evals_nD))

    # compute continuous Fourier transform from DFT.
    FT_f_approx = collect( DFT_f[i]*exp(im*2*π*dot(𝑓[i],t0))/prod(fs) for i = 1:N_samples )
    FT_f_approx_nD = reshape(FT_f_approx, N_array...)

    return FT_f_approx_nD, f_evals_nD, fs, 𝑓_nD, x_ranges, 𝑓_ranges
end

function approximatecontinuousIFT(FT_f_approx_nD::Array{Complex{T},D}, fs::Vector{T})::Array{T,D} where {T <: Real,D}
    return ifftshift(ifft(FT_f_approx_nD.*prod(fs)))
end
