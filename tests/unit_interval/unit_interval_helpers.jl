function getconfigforunitintervaldensities(  f::Function,
                                    θ_canonical::KT,
                                    limit_a::Vector{T},
                                    limit_b::Vector{T}) where {T,KT}
    #
    zero_tol_RKHS = 1e-13
    prune_tol = 1.1*zero_tol_RKHS
    max_iters_RKHS = 50000
    σ² = 1e-9
    N_IS = 10000
    N_𝑖 = 500

    avg_num_clusters_func = xx->max(round(Int,xx/20), 3)
    N_iters_DPGMM = 1000
    N_candidates = 1000
    a_LP = 1.0
    a_HPc = 0.7 /2  # larget is sharper.
    multiplier_warpmap_BP = 180.0
    multiplier_warpmap_LP = 1/30 * 300.0
    multiplier_warpmap_HP  = 1/15 *300.0
    ϵ_warpmap = 1e-7
    warp_weights = ones(T, 2)  #.* 0.3
    self_gain = 1.0
    RQ_a_kDPP = 500.0 #1.0
    kDPP_warp_weights = ones(T, 2)

    ϵ = 1e-2
    N_kDPP_draws = 1
    N_𝑗 = 100
    N_kDPP_per_draw = 100

    @assert length(kDPP_warp_weights) == 2 == length(warp_weights)

    w_config = KRTransportMap.IrregularWarpMapConfigType(N_IS, N_𝑖, ϵ, avg_num_clusters_func,
                    N_iters_DPGMM, N_candidates, a_LP, a_HPc,
                    multiplier_warpmap_LP, multiplier_warpmap_HP,
                    multiplier_warpmap_BP, ϵ_warpmap, warp_weights,
                    N_kDPP_draws, N_kDPP_per_draw, self_gain, RQ_a_kDPP,
                    kDPP_warp_weights, N_𝑗, prune_tol)

    #
    return KRTransportMap.fitadaptiveRKHSunnormalizeddensityconfig(f,
                                        limit_a,
                                        limit_b,
                                        θ_canonical,
                                        w_config,
                                        max_iters_RKHS,
                                        σ²,
                                        zero_tol_RKHS,
                                        prune_tol)
end


function packageupunitinterval(  f::Function,
                                    limit_a::Vector{T},
                                    limit_b::Vector{T}) where T
    #
    zero_tol_RKHS = 1e-13
    prune_tol = 1.1*zero_tol_RKHS
    max_iters_RKHS = 50000
    σ² = 1e-9
    N_IS = 10000
    N_𝑖 = 500

    avg_num_clusters_func = xx->max(round(Int,xx/20), 3)
    N_iters_DPGMM = 1000
    N_candidates = 1000
    a_LP = 100.0
    a_HPc = 70.0   # larget is sharper.
    multiplier_warpmap_BP = 180.0
    multiplier_warpmap_LP = 1/30 * 300.0
    multiplier_warpmap_HP  = 1/15 *300.0
    ϵ_warpmap = 1e-7
    warp_weights = ones(T, 2)  #.* 0.3
    self_gain = 1.0
    RQ_a_kDPP = 500.0 #1.0
    kDPP_warp_weights = ones(T, 2)

    ϵ = 1e-2
    N_kDPP_draws = 1
    N_𝑗 = 100
    N_kDPP_per_draw = 100

    @assert length(kDPP_warp_weights) == 2 == length(warp_weights)

    w_config = KRTransportMap.IrregularWarpMapConfigType(N_IS, N_𝑖, ϵ, avg_num_clusters_func,
                    N_iters_DPGMM, N_candidates, a_LP, a_HPc,
                    multiplier_warpmap_LP, multiplier_warpmap_HP,
                    multiplier_warpmap_BP, ϵ_warpmap, warp_weights,
                    N_kDPP_draws, N_kDPP_per_draw, self_gain, RQ_a_kDPP,
                    kDPP_warp_weights, N_𝑗, prune_tol)

    #
    return w_config
end
