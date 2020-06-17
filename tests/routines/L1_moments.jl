

# Let P := P_λ(x). Consider the L = 1 case.
# Compute via my formulae: P, inv(P), and sqrt(Q), where Q := P*inv(P).
function computePquantitiesL1(x, λ, ψ, σ²)
    u = vec(ForwardDiff.jacobian(ψ, x)[1,:])
    c = σ²*λ/(1+λ*dot(u,u))

    P_λ_x = σ²*LinearAlgebra.I - c*u*u'

    inv_P_λ_x = (1/σ²)*LinearAlgebra.I + (λ/σ²)*u*u'


    return P_λ_x, inv_P_λ_x#, Q, sqrt_Q
end

# define 𝑃1 := sqrt( P_λ1*inv(P_λ0) ) at x_λ0.
# x is x_λ0 in this function.
# ψ: ℝᴰ → ℝ here; a scalar-valued function.
function compute𝐺(x::Vector{T}, λ0, λ1, ψ, σ²) where T
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

function compute𝑚(x, z_input, λ, ψ, σ², y)
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



####

# m0 is z_input in the GFAK setting.


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

#### state update.

function computestateupdate(x, λ0, λ1, ψ, z_input, σ², y)

    m_λ1 = compute𝑚(x, z_input, λ1, ψ, σ², y)
    m_λ0 = compute𝑚(x, z_input, λ0, ψ, σ², y)

    G, dG = compute𝐺(x, λ0, λ1, ψ, σ²)

    #γ = 0
    #β_γ = exp(-0.5*γ*(λ1-λ0))

    return m_λ1 + G*(x-m_λ0)
end

function computedx1L1(x, λ0, λ1::T, ψ, z_input, σ², y) where T
    D = length(x)

    m_λ1 = compute𝑚(x, z_input, λ1, ψ, σ², y)
    m_λ0 = compute𝑚(x, z_input, λ0, ψ, σ², y)

    G, dG = compute𝐺(x, λ0, λ1, ψ, σ²)

    #γ = 0
    #β_γ = exp(-0.5*γ*(λ1-λ0))

    dm_λ0 = computed𝑚(x, z_input, λ0, ψ, σ², y)
    dm_λ1 = computed𝑚(x, z_input, λ1, ψ, σ², y)

    dm_λ0_mat = convertnestedvector(dm_λ0)
    A = G*( LinearAlgebra.I - dm_λ0_mat )

    C = Matrix{T}(undef, D, D)
    for l = 1:D
        for i = 1:D

            C[i,l] = zero(T)
            for j = 1:D
                C[i,l] += dG[i,j][l]*( x[j] - m_λ0[j] )
            end
        end
    end

    dx1 = Matrix{T}(undef, D, D)
    for l = 1:D
        for i = 1:D
            dx1[i,l] = dm_λ1[l][i] + A[i,l] + C[i,l]
        end
    end

    return dx1
    #return A # debug.
end
