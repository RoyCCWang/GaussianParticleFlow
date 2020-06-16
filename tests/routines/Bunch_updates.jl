# state, weight, moment updates from Bunch's paper.

function compute𝑚Bunch(x, λ, ψ, m0, inv_P0, inv_R, y)

    H = ForwardDiff.jacobian(ψ, x)
    y_hat = y - ψ(x) + H*x

    inv_P_λ_x = inv_P0 + λ*H'*inv_R*H
    P_λ_x = inv(inv_P_λ_x)

    𝑚 = P_λ_x*( inv_P0*m0 + λ*H'*inv_R*y_hat )

    return 𝑚
end

function computestateupdateBunch(x, λ0, λ1, ψ, m0, inv_P0, inv_R, y, γ = 0.0)

    m_λ1 = compute𝑚Bunch(x, λ1, ψ, m0, inv_P0, inv_R, y)
    m_λ0 = compute𝑚Bunch(x, λ0, ψ, m0, inv_P0, inv_R, y)

    G = compute𝐺Bunch(x, λ0, λ1, ψ, inv_P0, inv_R)

    β_γ = exp(-0.5*γ*(λ1-λ0))

    return m_λ1 + β_γ .* (G*(x-m_λ0))
end



##### covariance-related updates.

# Let P := P_λ(x). This is the general case as reported by Bunch's eqn 18.
# Compute via eqn 18: P, inv(P), and sqrt(Q), where Q := P*inv(P).
function computePquantitiesBunch(x, λ, ψ, inv_P0, inv_R)

    H = ForwardDiff.jacobian(ψ, x)

    inv_P_λ_x = inv_P0 + λ*H'*inv_R*H

    P_λ_x = inv(inv_P_λ_x)

    return P_λ_x, inv_P_λ_x#, Q, sqrt_Q
end


# define 𝑃1 := sqrt( P_λ1*inv(P_λ0) ) at x_λ0.
# x is x_λ0 in this function.
# ψ: ℝᴰ → ℝᴸ here; a vector-valued function, even if L == 1.
function compute𝐺Bunch(x, λ0, λ1, ψ, inv_P0, inv_R)

    H = ForwardDiff.jacobian(ψ, x)

    inv_P_λ1_x = inv_P0 + λ1*H'*inv_R*H
    P_λ1_x = inv(inv_P_λ1_x)

    inv_P_λ0_x = inv_P0 + λ0*H'*inv_R*H

    # println(" P_λ1_x*inv_P_λ0_x = ",  P_λ1_x*inv_P_λ0_x)
    𝐺 = sqrt( P_λ1_x*inv_P_λ0_x )

    # return 𝐺

    𝐺_real = real.(𝐺)

    return 𝐺_real
end

# Use numerical differentiation to compute df, where f: ℝᴰ → ℝ^{MxN}.
function computegradientformatrixfunctionND(x::Vector{T}, f, M, N) where T
    D = length(x)

    df = Matrix{Vector{T}}(undef, M, N)

    for j = 1:N
        for i = 1:M

            h = xx->f(xx)[i,j]

            df[i,j] = Calculus.gradient(h, x)
        end
    end

    return df
end
