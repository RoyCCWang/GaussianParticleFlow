# to do: move to dedicated repo later.

function getmixture2Dbetacopula1( τ::T,
                                    N_array::Vector{Int}) where T
    #

    D = 2
    limit_a = [τ; τ]
    limit_b = [1-τ; 1-τ]

    x_ranges = collect( LinRange(limit_a[d], limit_b[d], N_array[d]) for d = 1:D )

    R_g = [   1.12  -0.24;
        -0.24  0.07]
    R_f = [   0.61829  0.674525;
        0.674525  0.759165]

    a_g = [1.2; 3.4]
    θ_g = [2.0; 3.0]
    gamma_dists = collect( Distributions.Beta(a_g[d], θ_g[d]) for d = 1:D )

    g = xx->evalGaussiancopula(xx, R_g, gamma_dists)

    M = 5
    a = Vector{Vector{Float64}}(undef, M)
    a[1] = [1.2, 3.4]
    a[2] = [8.137763709838943, 8.773653512648599]
    a[3] = [10.778052251461023, 7.174287054713922]
    a[4] = [4.4419645076864525, 14.540862450228454]
    a[5] = [6.582089073364336, 8.060857932040115]

    θ = Vector{Vector{Float64}}(undef, M)
    θ[1] = [2.0, 3.0]
    θ[2] = [7.9914398321997275, 7.804725310098352]
    θ[3] = [2.5364332327991757, 14.67358895901066]
    θ[4] = [0.6668795542283168, 8.625209709368946]
    θ[5] = [2.6212936024181754, 12.445626221379737]

    #γ = 1.4
    #mix_weights = stickyHDPHMM.drawDP(γ, M)
    mix_weights = [0.13680543732370892;
                     0.0019470978511141439;
                     0.8395383241234364;
                     0.018327175970185964;
                     0.003381964731554627 ]

    mix_dists = collect( Distributions.MixtureModel(Distributions.Beta[
    Distributions.Beta(a[1][d], θ[1][d]),
    Distributions.Beta(a[2][d], θ[2][d]),
    Distributions.Beta(a[3][d], θ[3][d]),
    Distributions.Beta(a[4][d], θ[4][d]),
    Distributions.Beta(a[5][d], θ[5][d])], mix_weights) for d = 1:D )

    f = xx->evalGaussiancopula(xx, R_f, mix_dists)

    #
    h = xx->(0.15*f(xx)+0.85*g(xx))

    return h, x_ranges
end
