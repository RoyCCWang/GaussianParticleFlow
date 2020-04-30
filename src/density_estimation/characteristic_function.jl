# routines that estimates the characteristic function from samples.

# Spatial support is [-τ, τ].
# Uses Gaussian low-pass filters.
"""
    bar(x[, y])

For univariate distributions. Computes a grid-based interpolation
of an estimated unnormalized density function that governs Y.
"""
function estimatedensityviachar(  Y::Vector{T},
                                ln_w::Vector{T},
                                ::Val{1},
                                τ::T,
                                fs::T,
                                x_cut_off::T,
                                attenuation_factor_at_cut_off::T) where T <: Real

    #
    ### set up frequency and space locations.
    x_a = -τ
    x_b = τ
    x = collect( x_a:1/fs:x_b )
    N_DFT_samples = length(x)
    t0 = x_a

    𝑓 = LinRange(-fs/2,fs/2,N_DFT_samples)

    ### evaluate FT integral via Monte Carlo.
    FT_f_MC = Vector{Complex{T}}(undef, 0)
    if length(ln_w) == 0
        FT_f_MC = getMCFTarray1D(Y, 𝑓)
    else
        FT_f_MC = getMCFTarray1D(Y, 𝑓, ln_w)
    end


    # do not lowpass for now.
    # lowpass filter the estimated CFT, using DFT theory though.
    reciprocal_cut_off_percentage = 𝑓[end]/x_cut_off
    σ_c = π/(reciprocal_cut_off_percentage*sqrt(2*log(attenuation_factor_at_cut_off)))
    LP,HP = SignalTools.getGaussianfilters((length(FT_f_MC),),Val(1),σ_c)

    LP_shifted = fftshift(LP)

    # apply lowpass filter in Fourier domain.
    FT_f_MC_LP = LP_shifted.*FT_f_MC

    # inverse DFT, then take magnitude to ensure density is positive.
    f_evals_MC_LP = ifft(FT_f_MC_LP.*fs)
    f_evals_MC_LP_mag_rsp = fftshift(abs.(f_evals_MC_LP))

    return FT_f_MC, FT_f_MC_LP, f_evals_MC_LP_mag_rsp, x, 𝑓
end

### legacy.

function estimatedensityviaFT(  Y::Vector{T},
                                ::Val{1},
                                τ::T,
                                N_samples_per_dim::Int,
                                reciprocal_cut_off_percentage::T) where T <: Real

    #
    ### set up frequency and space locations.
    x_ranges, X, fs, 𝑓_ranges, 𝑓, 𝑓_nD, N_array, t0, N_samples = get𝑓ranges(τ, N_samples_per_dim, Val(1))

    ### Monte-carlo estimate of density's continuous Fourier transform.
    FT_f_MC = getMCFTarray1D(Y, 𝑓)


    # do not lowpass for now.
    # lowpass filter the estimated CFT, using DFT theory though.
    attenuation_factor_at_cut_off = 2
    σ_c = π/(reciprocal_cut_off_percentage*sqrt(2*log(attenuation_factor_at_cut_off)))
    LP,HP = SignalTools.getGaussianfilters((length(FT_f_MC),),Val(1),σ_c)

    LP_shifted = fftshift(LP)
    # apply lowpass filter in Fourier domain, then do inverse DFT.
    FT_f_MC_LP = LP_shifted.*FT_f_MC

    f_evals_MC_LP = ifftshift(ifft(FT_f_MC_LP.*fs))
    f_evals_MC_LP_mag_rsp = abs.(f_evals_MC_LP)

    return FT_f_MC, f_evals_MC_LP_mag_rsp, x_ranges, 𝑓_ranges
end

function estimatedensityviaFT(  Y::Vector{Vector{T}},
                                ::Val{D},
                                τ::T,
                                N_samples_per_dim::Int,
                                reciprocal_cut_off_percentage::T) where {T <: Real,D}

    ### set up frequency and space locations.
    x_ranges, X, fs, 𝑓_ranges, 𝑓, 𝑓_nD, N_array, t0, N_samples = get𝑓ranges(τ, N_samples_per_dim, Val(D))

    ### Monte-carlo estimate of density's continuous Fourier transform.
    FT_f_MC_nD = getMCFTarray(Y, 𝑓_nD, Val(D))

    ### estimate density samples.
    # lowpass filter the estimated CFT, using DFT theory though.
    attenuation_factor_at_cut_off = 2
    σ_c = π/(reciprocal_cut_off_percentage*sqrt(2*log(attenuation_factor_at_cut_off)))
    LP,HP = SignalTools.getGaussianfilters(tuple(N_array...),Val(D),σ_c)

    LP_shifted = fftshift(LP)

    # apply lowpass filter in Fourier domain, then do inverse DFT.
    FT_f_MC_nD_LP = LP_shifted.*FT_f_MC_nD

    f_evals_MC_LP = ifftshift(ifft(FT_f_MC_nD_LP.*prod(fs)))
    f_evals_MC_LP_mag_rsp = abs.(f_evals_MC_LP)

    return f_evals_MC_LP_mag_rsp, FT_f_MC_nD, FT_f_MC_nD_LP, x_ranges, 𝑓_ranges
end
