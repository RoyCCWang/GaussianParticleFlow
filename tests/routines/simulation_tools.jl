### routines for different ψ maps.

function exampleψfunc2Dto3D1(x::Vector{T})::Vector{T} where T <: Real
    @assert length(x) == 2

    out = Vector{T}(undef,3)

    out[1] = exp(-0.1*dot(x,x))*cos(x[1])*x[2] + x[2]^2 -x[1]
    out[2] = out[1]*x[1]^3
    out[3] = exp(-0.1*dot(x,x))*sin(x[2]*x[1]^4) - x[2]

    return out
end

function exampleψfunc2Dto2D1(x::Vector{T})::Vector{T} where T <: Real
    @assert length(x) == 2

    out = Vector{T}(undef,2)

    out[1] = exp(-0.1*dot(x,x))*cos(x[1])*x[2] + x[2]^2 -x[1]
    out[2] = out[1]*x[1]^3

    return out
end

function exampleψfunc2Dto2D1(x)
    @assert length(x) == 2

    out = Vector{Float64}(undef,2)

    out[1] = exp(-0.1*dot(x,x))*cos(x[1])*x[2] + x[2]^2 -x[1]
    out[2] = out[1]*x[1]^3

    return out
end

function exampleψfunc2Dto2D1Zygote(x::Vector{T})::Vector{T} where T <: Real
    @assert length(x) == 2
    #println("hi")
    out = Zygote.Buffer(x, 2)

    out[1] = exp(-0.1*dot(x,x))*cos(x[1])*x[2] + x[2]^2 -x[1]
    out[2] = out[1]*x[1]^3

    return Zygote.copy(out)
end

function exampleψfunc4Dto2D1(x::Vector{T})::Vector{T} where T <: Real
    @assert length(x) == 4

    out = Vector{T}(undef,2)

    out[1] = exp(-0.1*dot(x,x))*cos(x[1])*x[2] + x[2]^2 -x[1] + x[4]^3 + x[3]*x[2]
    out[2] = x[1]*x[2] + x[3]*x[4]

    return out
end

function exampleψfunc4Dto2D1Zygote(x::Vector{T})::Vector{T} where T <: Real
    @assert length(x) == 4

    out = Zygote.Buffer(x, 2)

    out[1] = exp(-0.1*dot(x,x))*cos(x[1])*x[2] + x[2]^2 -x[1] + x[4]^3 + x[3]*x[2]
    out[2] = x[1]*x[2] + x[3]*x[4]

    return Zygote.copy(out)
end

function examp2eψfunc2Dto1D1(x)
    @assert length(x) == 2

    out::Float64 = x[1]*x[2] -2*x[2] + x[1]^2

    return out
end
