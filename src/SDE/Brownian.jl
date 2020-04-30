


# assume starts at t = 0, n_next = 1.
function getnextBrownianmotion(n_next, t, t_next, B_t)

    return getnextBrownianmotion(n_next, t, t_next, B_t, randn())
end

function getnextBrownianmotion(n_next, t, t_next, B_t::T, Z_t::T) where T
    if n_next == 1
        return sqrt(t_next)*Z_t
    end

    return B_t + sqrt(t_next-t)*Z_t
end

function getnextBrownianmotion(n_next, t, t_next, B_t::T, Z_t_array::Vector{T}) where T
    @assert n_next <= length(Z_t_array)

    return getnextBrownianmotion(n_next, t, t_next, B_t, Z_t_array[n_next])
end

# assume t_array is sorted in ascending order.
function getallBrownianmotions(t_array)::Tuple{Vector{Float64},Vector{Float64}}
    N = length(t_array)

    Z_array = randn(N)

    Bt_array = Vector{Float64}(undef,N)

    Bt_array[1] = sqrt(t_array[1])*Z_array[1]
    for k = 2:N

        Bt_array[k] = sum( sqrt(t_array[i]-t_array[i-1])*Z_array[i] for i = 2:k ) + Bt_array[1]
    end

    return Bt_array, Z_array
end


function drawBrownianmotiontrajectories(N_discretizations::Int, D::Int, λ_array)

    # set up Brownian motion.

    # B1λ_array, Z1_array = getallBrownianmotions(λ_array)
    # B2λ_array, Z2_array = getallBrownianmotions(λ_array)
    # Bλ_array = collect( [B1λ_array[i]; B2λ_array[i]] for i = 1:length(λ_array) )

    Bdλ_array = Matrix{Float64}(undef, N_discretizations, D)
    Z_array = Matrix{Float64}(undef, N_discretizations, D)

    for d = 1:D
        Bdλ_array[:,d], Z_array[:,d] = getallBrownianmotions(λ_array)
    end

    Bλ_array = collect( Bdλ_array[n,:] for n = 1:N_discretizations )

    return λ_array, Bλ_array
end


function drawBrownianmotiontrajectorieswithoutstart(N_discretizations::Int, D::Int)

    λ_array = LinRange(1/(N_discretizations-1),1.0,N_discretizations)

    return drawBrownianmotiontrajectories(N_discretizations, D, λ_array)
end

function drawBrownianmotiontrajectorieswithstart(N_discretizations::Int, D::Int)

    λ_array = LinRange(0.0, 1.0, N_discretizations)

    return drawBrownianmotiontrajectories(N_discretizations, D, λ_array)
end

function setupdrawBrownia(D::Int, dummy_val::T) where T <: Real

    Bλ::Vector{T} = zeros(T, D)

    drawfunc = dd->drawindptBrowniansonline!(Bλ, dd)

    return Bλ, drawfunc
end

# univariate.
# Δλ must be positive.
function drawBrownianonline(Bλ_prev::T, Δλ::T)::T where T <: Real
    return Bλ_prev + sqrt(Δλ)*randn()
end

function drawindptBrowniansonline!(Bλ::Vector{T}, Δλ::T) where T <: Real
    D = length(Bλ)

    for d = 1:D
        Bλ[d] = drawBrownianonline(Bλ[d], Δλ)
    end

    return nothing
end
