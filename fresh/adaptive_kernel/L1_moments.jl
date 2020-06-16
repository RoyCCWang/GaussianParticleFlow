

# Let P := P_λ(x). Consider the L = 1 case.
# Compute via my formulae: P, inv(P), and sqrt(Q), where Q := P*inv(P).
function computePquantitiesL1(x, λ, ψ, σ²)
    u = vec(ForwardDiff.jacobian(ψ, x)[1,:])
    c = σ²*λ/(1+λ*dot(u,u))

    P_λ_x = σ²*LinearAlgebra.I - c*u*u'

    inv_P_λ_x = (1/σ²)*LinearAlgebra.I + (λ/σ²)*u*u'

    # # WIP.
    # K = 1.234
    # Q = LinearAlgebra.I + K .* u*u'
    #
    # x = ( sqrt(1+dot(u,u)*K)-1 )/dot(u,u)
    # sqrt_Q = LinearAlgebra.I + x .* u*u'

    return P_λ_x, inv_P_λ_x#, Q, sqrt_Q
end

# define 𝑃1 := sqrt( P_λ1*inv(P_λ0) ) at x_λ0.
# x is x_λ0 in this function.
# ψ: ℝᴰ → ℝ here; a scalar-valued function.
function compute𝐺(x, λ0, λ1::T, ψ, σ²) where T
    D = length(x)

    dψ_x = ForwardDiff.gradient(ψ, x)
    d2ψ_x = ForwardDiff.hessian(ψ, x)

    u = dψ_x
    uᵀu = dot(u,u)

    denominator = 1 + λ1*uᵀu
    k = λ0 - ( 1 + λ0*uᵀu )*λ1/denominator

    #𝐺_naive = sqrt( LinearAlgebra.I + k .* u*u' ) # does sqrt(Q).
    𝐺 = LinearAlgebra.I + (( sqrt(1+uᵀu*k) - 1 )/uᵀu) .* u*u'

    ### ∂𝐺_∂x.
    z = 1 + uᵀu*k

    a = sqrt(z) - 1
    #da = 1/(2*sqrt( 1 + uᵀu * k )) .* dz

    #d_a_over_uᵀu = (da .* uᵀu - a .* d_uᵀu)/uᵀu^2

    ### put it together.
    b = uᵀu

    β_dk = ( λ1^2 - λ1*λ0*(1+λ1*uᵀu) + λ1^2*λ0*uᵀu )/((1+λ1*uᵀu)^2)

    β_da = (k+uᵀu*β_dk)/(2*sqrt(1+uᵀu*k))

    β_q = (β_da * uᵀu -a)/(uᵀu^2)

    d𝐺 = Matrix{Vector{T}}(undef, D, D)

    for j = 1:D
        for i = 1:D

            d𝐺[i,j] = Vector{T}(undef, D)
            for l = 1:D

                running_sum = zero(T)
                for m = 1:D
                    running_sum += dψ_x[m]*d2ψ_x[m,l]
                end

                term1 = 2*β_q*running_sum*dψ_x[i]*dψ_x[j]

                term2 = a/b * ( d2ψ_x[i,l]*dψ_x[j] + dψ_x[i]*d2ψ_x[j,l] )

                d𝐺[i,j][l] = term1 + term2
            end
        end
    end

    return 𝐺, d𝐺
end


# function compute𝑚old(x, z_input, λ::T, ψ, σ², y) where T
#     D = length(x)
#     @assert length(z_input) == D
#
#     dψ_x = ForwardDiff.gradient(ψ, x)
#     d2ψ_x = ForwardDiff.hessian(ψ, x)
#
#     y_hat = y - ψ(x) + dot(dψ_x, x)
#
#     # 𝑚.
#     u = dψ_x
#     uᵀu = dot(u,u)
#     denominator = 1 + λ*uᵀu
#
#     κ = λ*y_hat - λ*dot(u,z_input)/denominator - λ^2*y_hat*uᵀu/denominator
#
#     𝑚 = z_input + κ .* u
#
#     ### ∂𝑚_∂x.
#
#     # pre-compute common quantities.
#     w_v = y_hat - dot(u, z_input) - dot(u,u)*κ
#     β_v = λ*w_v
#
#     # compute derivative.
#     d𝑚 = Vector{Vector{T}}(undef, D)
#
#     for j = 1:D
#
#         # quantities related to v.
#         v = vec(d2ψ_x[j,:])
#         #v = vec(d2ψ_x[:,j])
#
#
#         w_u = dot(v, x-z_input-u)*κ
#
#         β_u = λ * ( w_u - λ/denominator *(w_u*uᵀu + w_v*dot(u,v)) )
#
#         #𝐺_naive = sqrt( LinearAlgebra.I + k .* u*u' ) # does sqrt(Q).
#         d𝑚[j] = β_u .* u + β_v .* v
#
#
#         # debug.
#         blah = z_input+κ .* u
#         P1 = σ²*LinearAlgebra.I - σ²*λ/(1+λ*dot(u,u)) .* u*u'
#
#         # inv_P = diagm(1/σ² * ones(Float64, D)) +
#         #             (λ/σ²)*dψ_x*dψ_x'
#         # P2 = inv(inv_P)
#         #
#         # println(" norm(P1-P2)  = ", norm(P1-P2))
#         # @assert 12==3
#         P = P1
#
#         #d𝑚[j] = P*λ/σ² *( v .* (y_hat - dot(u, blah)) + u .* dot(v, x-blah) )
#         d𝑚[j] = P*λ/σ² *( v .* (y_hat - dot(u, 𝑚)) + u .* dot(v, x-𝑚) )
#
#     end
#
#     return 𝑚, d𝑚
# end

function compute𝑚(x, z_input, λ, ψ, σ², y) where T
    D = length(x)
    @assert length(z_input) == D

    dψ_x = ForwardDiff.gradient(ψ, x)
    d2ψ_x = ForwardDiff.hessian(ψ, x)

    y_hat = y - ψ(x) + dot(dψ_x, x)

    # 𝑚.
    u = dψ_x
    uᵀu = dot(u,u)
    denominator = 1 + λ*uᵀu

    κ = λ*y_hat - λ*dot(u,z_input)/denominator - λ^2*y_hat*uᵀu/denominator
    #κ = computeκ(x, z_input, λ, ψ, σ², y)

    𝑚 = z_input + κ .* u

    # ### ∂𝑚_∂x.
    #
    # # pre-compute common quantities.
    # v = collect( d2ψ_x[:,j] for j = 1:D )
    #
    # # compute derivative.
    #
    # dκ = computedκ(x, z_input, λ, ψ, σ², y)
    # d𝑚 = collect( dκ[l] .* u + κ .* v[l] for l = 1:D )

    return 𝑚
end

function computed𝑚(x, z_input, λ, ψ, σ², y) where T
    D = length(x)
    @assert length(z_input) == D

    dψ_x = ForwardDiff.gradient(ψ, x)
    d2ψ_x = ForwardDiff.hessian(ψ, x)

    y_hat = y - ψ(x) + dot(dψ_x, x)

    # 𝑚.
    u = dψ_x

    # pre-compute common quantities.
    v = collect( d2ψ_x[:,j] for j = 1:D )

    # compute derivative.
    κ = computeκ(x, z_input, λ, ψ, σ², y)

    dκ = computedκ(x, z_input, λ, ψ, σ², y)
    d𝑚 = collect( dκ[l] .* u + κ .* v[l] for l = 1:D )
    # each element of d𝑚 is the D-dim outputs derivative wrt to the l-th component.
    # i.e., the l-th column of the Jacobian of 𝑚.

    return d𝑚
end

function convertnestedvector(X::Vector{Vector{T}})::Matrix{T} where T
    D_dest = length(X[1])
    D_src = length(X)

    out = Matrix{T}(undef, D_dest, D_src)
    for j = 1:D_src
        for i = 1:D_dest
            out[i,j] = X[j][i]
        end
    end

    return out
end

"""
u := ∂ψ/∂x, ψ: ℝᴰ → ℝ.
"""
function evalduuᵀ(dψ_x::Vector{T}, d2ψ_x::Matrix{T}, i, j, l)::T where T
    return d2ψ_x[i,l]*dψ_x[j] + dψ_x[i]*d2ψ_x[j,l]
end


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




####

# m0 is z_input in the GFAK setting.
function compute𝑚Bunch(x, λ, ψ, m0, inv_P0, inv_R, y)

    H = ForwardDiff.jacobian(ψ, x)
    y_hat = y - ψ(x) + H*x

    inv_P_λ_x = inv_P0 + λ*H'*inv_R*H
    P_λ_x = inv(inv_P_λ_x)

    𝑚 = P_λ_x*( inv_P0*m0 + λ*H'*inv_R*y_hat )

    return 𝑚
end

function computeyhatL1(x, ψ, y)

    H = ForwardDiff.jacobian(ψ, x)

    y_hat = y - ψ(x) + H*x

    return y_hat
end

function computeκ(x, z_input, λ::T, ψ, σ², y) where T
    D = length(x)
    @assert length(z_input) == D

    dψ_x = ForwardDiff.gradient(ψ, x)
    d2ψ_x = ForwardDiff.hessian(ψ, x)

    y_hat = y - ψ(x) + dot(dψ_x, x)

    # 𝑚.
    u = dψ_x
    uᵀu = dot(u,u)
    denominator = 1 + λ*uᵀu

    κ = λ*y_hat - λ*dot(u,z_input)/denominator - λ^2*y_hat*uᵀu/denominator

    return κ
end

function computedκ(x, z_input, λ::T, ψ, σ², y) where T
    D = length(x)
    @assert length(z_input) == D

    dψ_x = ForwardDiff.gradient(ψ, x)
    d2ψ_x = ForwardDiff.hessian(ψ, x)

    y_hat = y - ψ(x) + dot(dψ_x, x)

    # 𝑚.
    u = dψ_x
    uᵀu = dot(u,u)
    denominator = 1 + λ*uᵀu

    v = collect( d2ψ_x[:,j] for j = 1:D )
    dy_hat = collect( dot(v[j],x) for j = 1:D )

    B = 1+λ*uᵀu

    β_y_hat = λ-λ^2*uᵀu/B
    β_b = λ^2*( dot(u,z_input)/B^2 -y_hat/B + λ*(y_hat*uᵀu)/B^2 )

    d_uᵀu = Vector{T}(undef, D)
    for l = 1:D

        d_uᵀu[l] = zero(T)
        for m = 1:D
            d_uᵀu[l] += 2*dψ_x[m]*d2ψ_x[m,l]
        end
    end

    #dκ = collect( β_y_hat*dy_hat[l] - λ*dot(v[l],z_input)/B + β_b * d_uᵀu[l] for l = 1:D )

    dκ = collect( β_y_hat*dot(v[l],x) - λ*dot(v[l],z_input)/B + β_b * d_uᵀu[l] for l = 1:D )

    #β_v = (B-1-λ*uᵀu)*λ/B
    #β_b = λ^2/B * ( dot(u,z_input)/B + y_hat*(λ*uᵀu*/B-1) )
    #dκ = collect( β_v*dot(v[l],z_input-x) + β_b * d_uᵀu[l] for l = 1:D )

    # db = d_uᵀu
    # dyb = dy_hat*uᵀu + y_hat*db

    # dκ = Vector{T}(undef, D)
    # for l = 1:D
    #
    #     term1 = λ*dy_hat[l]
    #
    #     term2 = ( λ*dot(v[l],z_input) *B - db[l]*λ^2*dot(u,z_input) )/B^2
    #
    #     term3 = λ^2*( dyb[l]*B - y_hat*uᵀu*λ*db[l] )/B^2
    #
    #     dκ[l] = term1 -term2 -term3
    #
    # end

    return dκ
end

# function computed𝑚Bunch(x, λ::T, ψ, m0, inv_P0, inv_R, y) where T
#
#     H = ForwardDiff.jacobian(ψ, x)
#     y_hat = y - ψ(x) + H*x
#
#     inv_P_λ_x = inv_P0 + λ*H'*inv_R*H
#     P_λ_x = inv(inv_P_λ_x)
#
#     𝑚 = P_λ_x*( inv_P0*m0 + λ*H'*inv_R*y_hat )
#
#     # I am here. construct αH.
#     L = length(y)
#     D = length(x)
#
#     dH_x = collect( Matrix{T}(undef, D, D) for j = 1:D )
#
#
#     for i = 1:L
#
#         f = xx->ψ(xx)[i]
#         d2f_x = ForwardDiff.hessian(f, x)
#
#         for j = 1:D
#             for k = 1:D
#
#                 dH_x[j][i,k] = d2f_x[j,k]
#             end
#         end
#     end
#
#     for j = 1:D
#         ### I am here.
#         println("size(dH_x[j]) = ", size(dH_x[j]))
#
#         term1 = λ*P_λ_x*(dH_x[j]'*inv_R*(y_hat-H*𝑚))
#         term2 = H'*inv_R*dH_x[j]*(x-𝑚)
#
#         d𝑚[j] = term1 + term2
#     end
#
#     return d𝑚
# end
