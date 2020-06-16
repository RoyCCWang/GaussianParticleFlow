

## for adaptive kernel.
Base.@kwdef struct GFAKParamsType{T}
    z_persist::Vector{T} # m_0.
    σ²_persist::Vector{T} # P_0 and R.
    y::Vector{T}

    inv_σ²_persist::Vector{T} # for inv(R) and inv(P_0)
    inv_σ²_mul_z_persist::Vector{T} # for inv(P0)*m_0.
end

## adaptive kernel.
Base.@kwdef struct GaussianFlowMutatingMethodsType
    ψ::Function # debug.

    ψ!::Function
    ∂ψ!::Function
    ∂2ψ!::Function
    ln_prior_pdf_func::Function
    ln_likelihood_func::Function
end

## adaptive kernel.
Base.@kwdef struct GFAKBuffersType{T}

    ψ_eval::Vector{T}
    𝐻::Matrix{T}
    ∂2ψ_eval::Vector{Vector{T}}

    #∂𝐻_∂x::Vector{Matrix{T}}
    𝑦::Vector{T}

    # moments.
    𝑚_a::Vector{T}
    𝑃_a::Matrix{T}
    𝑚_b::Vector{T}
    𝑃_b::Matrix{T}

    # derivatives.
    ∂𝑚_a_∂x::Matrix{T}
    ∂𝑚_b_∂x::Matrix{T}

    ∂𝑃_b_∂x::Vector{Matrix{T}} # length D.
    ∂𝑃_b_inv𝑃_a_∂x::Vector{Matrix{T}} # length D.

    ∂𝑃_b_sqrt_∂x::Vector{Matrix{T}} # length D.
    ∂𝑃_b_inv𝑃_a_sqrt_∂x::Vector{Matrix{T}} # length D.
end

Base.@kwdef struct GFAKConfigType{T}
    γ::T
    mode::Symbol
end





abstract type GaussianFlowParamsType end

# conditional non-linear Gaussian system.
Base.@kwdef struct GaussianFlowSimpleParamsType{T} <: GaussianFlowParamsType
    m_0::Vector{T}
    P_0::Matrix{T}
    R::Matrix{T}
    y::Vector{T}

    inv_R::Matrix{T}
    inv_P0_mul_m0::Vector{T}
    inv_P0::Matrix{T}
end

# product likelihood.
Base.@kwdef struct GaussianFlowBlockDiagonalParamsType{T} <: GaussianFlowParamsType
    m_0::Vector{T}
    P_0::Matrix{T}
    R_set::Vector{Matrix{T}}
    y_set::Vector{Vector{T}}

    inv_R_set::Vector{Matrix{T}}
    inv_P0_mul_m0::Vector{T}
    inv_P0::Matrix{T}

    # intermediates.
end

Base.@kwdef struct GaussianFlowConfigType{T}
    γ::T
end

Base.@kwdef struct GaussianFlowMethodsType
    ψ::Function
    ∂ψ::Function
    ∂2ψ::Function
    ln_prior_pdf_func::Function
    ln_likelihood_func::Function
end

abstract type GaussianFlowBuffersType end

Base.@kwdef struct GaussianFlowSimpleBuffersType{T} <: GaussianFlowBuffersType

    ψ_eval::Vector{T}
    𝐻::Matrix{T}
    ∂2ψ_eval::Vector{Vector{T}}

    #∂𝐻_∂x::Vector{Matrix{T}}
    𝑦::Vector{T}

    # moments.
    𝑚_a::Vector{T}
    𝑃_a::Matrix{T}
    𝑚_b::Vector{T}
    𝑃_b::Matrix{T}

    # derivatives.
    ∂𝑚_a_∂x::Matrix{T}
    ∂𝑚_b_∂x::Matrix{T}

    ∂𝑃_b_∂x::Vector{Matrix{T}} # length D.
    ∂𝑃_b_inv𝑃_a_∂x::Vector{Matrix{T}} # length D.

    ∂𝑃_b_sqrt_∂x::Vector{Matrix{T}} # length D.
    ∂𝑃_b_inv𝑃_a_sqrt_∂x::Vector{Matrix{T}} # length D.
end

Base.@kwdef struct GaussianFlowBlockDiagonalBuffersType{T} <: GaussianFlowBuffersType
    ψ_eval::Vector{Vector{T}}
    ∂ψ_eval::Vector{Vector{Vector{T}}}
    ∂2ψ_eval::Vector{Vector{Vector{T}}}

    𝐻_set::Vector{Matrix{T}}
    #∂𝐻_j_set::Vector{Matrix{T}}
    𝑦_set::Vector{Vector{T}}

    # moments
    𝑚_a::Vector{T}
    𝑃_a::Matrix{T}
    𝑚_b::Vector{T}
    𝑃_b::Matrix{T}
end
