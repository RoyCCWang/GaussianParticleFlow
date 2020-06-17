# routines related to moments, derivatives of moments.
# mode is a Symbol that specifies if L == 1 or larger.
# Equations taken from Bunch et al.'s Gaussian flow 2016 paper.

#### fast, for L == 1.

function computedG!(dG::Matrix{Vector{T}},
                    dψ_x::Vector{T},
                    d2ψ_x::Matrix{T},
                    a_div_b::T,
                    β_q::T) where T <: Real

    D = size(dG,1)
    @assert size(dG,2) == D
    @assert length(dG[1]) == D

    for j = 1:D
        for i = 1:D

            #dG[i,j] = Vector{T}(undef, D)
            for l = 1:D

                running_sum = zero(T)
                for m = 1:D
                    running_sum += dψ_x[m]*d2ψ_x[m,l]
                end

                term1 = 2*β_q*running_sum*dψ_x[i]*dψ_x[j]

                term2 = a_div_b * ( d2ψ_x[i,l]*dψ_x[j] + dψ_x[i]*d2ψ_x[j,l] )

                dG[i,j][l] = term1 + term2
            end
        end
    end

    return nothing
end

"""
State and weight update for the L = 1 case.
"""
function updatesL1(x, λ0, λ1::T, ψ, z_input, σ², y) where T
    D = length(x)

    ## state update, slow.
    # m_λ1 = compute𝑚(x, z_input, λ1, ψ, σ², y)
    m_λ0 = compute𝑚(x, z_input, λ0, ψ, σ², y)
    #
    # G, dG = compute𝐺(x, λ0, λ1, ψ, σ²)
    #
    # x1_old = m_λ1 + G*(x-m_λ0)


    dψ_x = ForwardDiff.gradient(ψ, x)
    d2ψ_x = ForwardDiff.hessian(ψ, x)

    u = dψ_x
    uᵀu = dot(u,u)

    k = λ0 - ( 1 + λ0*uᵀu )*λ1/(1 + λ1*uᵀu)
    a_div_b = (sqrt(1+uᵀu*k) - 1)/uᵀu

    # this is κ_λ1.
    κ = computeκ(x, z_input, λ1, ψ, σ², y)

    ### state update.
    #κ_λ0 = computeκ(x, z_input, λ0, ψ, σ², y)
    #r = x - z_input - κ_λ0*u
    r = x - m_λ0

    x1 = Vector{T}(undef, D)
    for i = 1:D
        x1[i] = z_input[i] + κ*u[i] + r[i] + a_div_b*u[i]*dot(u,r)
    end

    ## compute dG.
    β_dk = ( λ1^2 - λ1*λ0*(1+λ1*uᵀu) + λ1^2*λ0*uᵀu )/((1+λ1*uᵀu)^2)

    a_plus_1 = sqrt(1+uᵀu*k)
    β_da = (k+uᵀu*β_dk)/(2*a_plus_1)

    a = a_plus_1 -1
    β_q = (β_da * uᵀu -a)/(uᵀu^2)

    dG = Matrix{Vector{T}}(undef, D, D)
    for m = 1:length(dG)
        dG[m] = Vector{T}(undef, D)
    end
    computedG!( dG,
                dψ_x,
                d2ψ_x,
                a_div_b,
                β_q)

    # I am here.
    #check if further speed ups are possible with weight update.

    # right final implemented equation in notes.
    # stand-alone benchmark test for state and weight updates.

    # package into function, write GFAK.
    # visualize.

    #γ = 0
    #β_γ = exp(-0.5*γ*(λ1-λ0))

    dm_λ0 = computed𝑚(x, z_input, λ0, ψ, σ², y) # not pre-allocate.

    w_λ0 = collect( dot(u,dm_λ0[l]) for l = 1:D ) # not pre-allocate.

    #v = collect( d2ψ_x[:,j] for j = 1:D ) # not pre-allocate. only one use for this?!

    dκ = computedκ(x, z_input, λ1, ψ, σ², y) # merge with κ_λ0.

    #@assert size(dx,2) == D == size(dx,1)
    dx = Matrix{T}(undef, D, D)
    for l = 1:D
        for i = 1:D

            # compute C[i,l].
            C_il = zero(T)
            for j = 1:D
                #C_il += dG[i,j][l]*( x[j] - m_λ0[j] )
                C_il += dG[i,j][l]*( x[j] - m_λ0[j] )
            end

            # compute A[i,l]
            A_λ0_il = -dm_λ0[l][i] + a_div_b*( u[i]*u[l] - u[i]*w_λ0[l] )

            # first two terms.
            #term12 = dκ[l]*u[i] + κ*v[l][i]
            term12 = dκ[l]*u[i] + κ*d2ψ_x[i,l]

            # put it together.
            dx[i,l] = term12 + A_λ0_il + C_il

            # debug.
            #dx[i,l] = A_λ0_il
        end
    end

    # the kronecker delta term for A_ij.
    for l = 1:D
        #dx[l,l] += β_γ
        dx[l,l] += one(T) # when γ = 0.
    end

    return dx
end
