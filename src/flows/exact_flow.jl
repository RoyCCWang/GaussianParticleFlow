### routines for exact flow.

# section 3.1 for Gaussian flow paper.
function getexactmoments(λ, m_0, Σ_0, H, R, y)
    Σ_λ = inv( inv(Σ_0) + λ*H'*inv(R)*H )
    Σ_λ = Utilities.forcesymmetric(Σ_λ)

    m_λ = Σ_λ*( inv(Σ_0)*m_0 + λ*H'*inv(R)*y )

    @assert isposdef(Σ_λ)

    return m_λ, Σ_λ
end

# based on the last equation in Appendix D.
function computeflowparticlestate(  x_λ,
                                    λ, δλ,
                                    m_0, Σ_0, H, R, y,
                                    ϵ_λ, ϵ_λpδλ, γ)

    m_λ, Σ_λ = getexactmoments(λ, m_0, Σ_0, H, R, y)

    term1 = Σ_λ*H'*inv(R)*( (y-H*m_λ) -0.5*H*(x_λ-m_λ) -0.5*γ*(x_λ-m_λ) )*δλ
    term2 = sqrt(γ)*Utilities.naivesqrtpsdmatrix(Σ_λ)*(ϵ_λpδλ - ϵ_λ)

    x_λpδλ = x_λ + term1 + term2

    return x_λpδλ
end



function traversesdepathexactflow(   x_0::Vector{T},
                            λ_array,
                            Bλ_array,
                            γ,
                            m_0, Σ_0, H, R, y) where T

    x = Vector{Vector{T}}(undef,length(λ_array))

    # to start.
    ϵ_λ = zeros(T,length(x_0))
    ϵ_λpδλ = Bλ_array[1]
    λ = zero(T)
    δλ = λ_array[1]-λ


    x[1] = computeflowparticlestate(x_0,
                                    λ, δλ,
                                    m_0, Σ_0, H, R, y,
                                    ϵ_λ, ϵ_λpδλ, γ)

    for n = 2:length(x)
        ϵ_λ = Bλ_array[n-1]
        ϵ_λpδλ = Bλ_array[n]
        λ = λ_array[n-1]
        δλ = λ_array[n]-λ

        x[n] = computeflowparticlestate(x[n-1],
                                        λ, δλ,
                                        m_0, Σ_0, H, R, y,
                                        ϵ_λ, ϵ_λpδλ, γ)

    end

    return x
end




# via equation 15.
function computeflowparticlestateeqn15(  x,
                                    λ_a, λ_b,
                                    m_0, Σ_0, H, R, y,
                                    ϵ_a, ϵ_b, γ)

    m_a, P_a = getexactmoments(λ_a, m_0, Σ_0, H, R, y)
    m_b, P_b = getexactmoments(λ_b, m_0, Σ_0, H, R, y)

    Δλ = λ_b - λ_a

    factor1 = real.(LinearAlgebra.sqrt(P_b*inv(P_a)))
    term2 = exp(-0.5*γ*Δλ)*factor1*(x-m_a)

    term3 = sqrt( (1-exp(-γ*Δλ))/Δλ )* real.(LinearAlgebra.sqrt(P_b)) *( ϵ_b - ϵ_a )

    out = m_b + term2 + term3

    return out
end

function traversesdepathexactflowviaeqn15(   x_0::Vector{T},
                                            λ_array,
                                            Bλ_array,
                                            γ,
                                            m_0, Σ_0, H, R, y) where T

    x = Vector{Vector{T}}(undef,length(λ_array))
    D_x = length(x_0)

    # to start.
    ϵ_a = zeros(T,D_x)
    ϵ_b = Bλ_array[1]
    λ_a = zero(T)
    λ_b = λ_array[1]


    x[1] = computeflowparticlestateeqn15(   x_0,
                                            λ_a, λ_b,
                                            m_0, Σ_0, H, R, y,
                                            ϵ_a, ϵ_b, γ)

    for n = 2:length(x)
        ϵ_a = Bλ_array[n-1]
        ϵ_b = Bλ_array[n]
        λ_a = λ_array[n-1]
        λ_b = λ_array[n]

        x[n] = computeflowparticlestateeqn15(   x[n-1],
                                                λ_a, λ_b,
                                                m_0, Σ_0, H, R, y,
                                                ϵ_a, ϵ_b, γ)

    end

    return x
end
