
# """
# X[d][n], where n is chain index, d is dimension index.
# """
# function evalexpectationMCMC(f::Function,
#                                 X::Vector{Vector{T}})::T where T <: Real
#     #
#     D = length(X)
#     return collect( sum(f(X[d]))/length(X[d]) for d = 1:D)
# end

# MCMC or IID.
function evalexpectation(f::Function,
                                X::Vector{Vector{T}})::T where T <: Real
    #
    return sum( f(X[n]) for n = 1:length(X))/length(X)
end

# importance sampling.
function evalexpectation(f::Function,
                                X::Vector{Vector{T}},
                                w::Vector{T})::T where T <: Real
    #
    return sum( w[n]*f(X[n]) for n = 1:length(X) )
end

# more accurate, uses log-space weights.
function evalexpectation2(f::Function,
                                x_array::Vector{Vector{T}},
                                ln_w_array::Vector{T}) where T <: Real

    N = length(x_array)
    ln_N = log(N)
    ln_W = StatsFuns.logsumexp(ln_w_array)

    ln_out_positive = zeros(T, N)
    ln_out_negative = zeros(T, N)

    kp = 0
    kn = 0
    for i = 1:N
        f_x = f(x_array[i])

        if f_x > zero(T)
            kp += 1
            ln_out_positive[kp] = ln_w_array[i] + log(f_x)
        else
            kn += 1
            ln_out_negative[kn] = ln_w_array[i] + log(abs(f_x))
        end
    end
    resize!(ln_out_positive, kp)
    resize!(ln_out_negative, kn)


    # division by the sum of weights.
    out_positive = exp(StatsFuns.logsumexp(ln_out_positive) - ln_W)
    out_negative = exp(StatsFuns.logsumexp(ln_out_negative) - ln_W)


    return out_positive - out_negative, ln_out_positive, ln_out_negative
end

function uniformsampling(   limit_a::Vector{T},
                            limit_b::Vector{T},
                            N::Int)::Vector{Vector{T}} where T <: Real
    #
    D = length(limit_a)

    return collect( rand(D) for n = 1:N )
end

function evalexpectation(f::Function,
                            p_y_given_x::Function,
                            p_x::Function,
                            limit_a::Vector{T},
                            limit_b::Vector{T},
                            max_integral_evals::Int,
                            initial_div::Int) where T <: Real

    # prepare posterior.
    p_tilde = xx->p_y_given_x(xx)*p_x(xx)

    # # better-condition p_tilde0
    # X = uniformsampling(limit_a, limit_b, N)
    # Y = p_tilde0.(X)
    # p_tilde = xx->p_tilde(xx)/maximum(Y)
    #p_tilde = p_tilde0

    return evalexpectation(f, p_tilde, limit_a, limit_b, max_integral_evals, initial_div)
end

function evalexpectation(f::Function,
                            p_tilde::Function,
                            limit_a::Vector{T},
                            limit_b::Vector{T},
                            max_integral_evals::Int,
                            initial_div::Int) where T <: Real

    # get normalizing constant.
    val_Z, err_Z = evalintegral(p_tilde, limit_a, limit_b, max_integral_evals, initial_div)

    # integrand.
    h = xx->p_tilde(xx)*f(xx)

    # integrate.
    val_h, err_h = evalintegral(h, limit_a, limit_b, max_integral_evals, initial_div)

    return val_h/val_Z, val_h, err_h, val_Z, err_Z
end

function evalintegral( f::Function,
                        limit_a::Vector{T},
                        limit_b::Vector{T},
                        max_integral_evals::Int,
                        initial_div::Int) where T <: Real
    #
    @assert length(limit_a) == length(limit_b)


    return val_Z, err_Z = HCubature.hcubature( f, limit_a, limit_b;
                                            norm = norm, rtol = sqrt(eps(T)),
                                            atol = 0,
                                            maxevals = max_integral_evals,
                                            initdiv = initial_div )
end

function evalintegral( f::Function,
                        limit_a::T,
                        limit_b::T;
                        max_integral_evals::Int = 10000,
                        initial_div::Int = 1) where T <: Real

    return val_Z, err_Z = HCubature.hquadrature( f, limit_a, limit_b;
                                            norm = norm, rtol = sqrt(eps(T)),
                                            atol = 0,
                                            maxevals = max_integral_evals,
                                            initdiv = initial_div )
end
