
#### general-purpose routines.
# Eventually move this to the Utilites package.

"""
    packagesym(v::Vector{T}, D::Int)
package the unique entries of a symmetric matrix into such matrix.
"""
function packagesym(v::Vector{T},
                    D::Int) where T
    #
    A = Matrix{T}(undef, D, D)
    for j = 1:D
        for i = 1:j
            A[i,j] = Utilities.readsymmetric(i, j, D, v)
        end
    end

    forcesymmetricviauppertriangular!(A)

    return A
end

"""

A must be square
"""
function forcesymmetricviauppertriangular!(A::Matrix{T}) where T
    M = size(A,2)
    @assert size(A,1) == M

    for i = 2:M
        for j = 1:(i-1)
            A[i,j] = A[j,i]
        end
    end

    return nothing
end


"""
Block diagonal version of H'*R*H.
"""
function applyHtRH( H_set::Vector{Matrix{T}},
                    R_set::Vector{Matrix{T}})::Matrix{T} where T
    #
    N, D = size(H_set[1])
    M = length(R_set)
    @assert length(H_set) == M

    return sum( H_set[m]'*R_set[m]*H_set[m] for m = 1:M )
end


"""
Block diagonal version of H'*R*y.
"""
function applyHtRy( H_set::Vector{Matrix{T}},
                    R_set::Vector{Matrix{T}},
                    y_set::Vector{Vector{T}})::Vector{T} where T
    #
    N, D = size(H_set[1])
    M = length(R_set)
    @assert length(H_set) == M

    return sum( H_set[m]'*R_set[m]*y_set[m] for m = 1:M )
end


"""
Matrix-vector multiplication of H*x, both in block form.

"""
function evalmatrixvectormultiply(  H_set::Vector{Matrix{T}},
                                    x_set::Vector{Vector{T}})::Vector{Vector{T}} where T
    #
    N, D = size(H_set[1])
    M = length(x_set)
    @assert length(H_set) == M

    return collect( H_set[m]*x_set[m] for m = 1:M )
end

function evalmatrixvectormultiply(  H_set::Vector{Matrix{T}},
                                    x::Vector{T})::Vector{Vector{T}} where T
    #
    N, D = size(H_set[1])
    M = length(H_set)
    @assert length(x) == D

    return collect( H_set[m]*x for m = 1:M )
end

"""
Example:
'''
M = 3
D = 2
H_set = collect( randn(N,D) for m = 1:M )
x_set = collect( randn(D) for d = 1:D )
H = [ H_set[1]; H_set[2]; H_set[3] ]
H2 = verticalblocktomatrix(H_set)
display(norm(H-H2))
```
"""
function verticalblocktomatrix(H_set::Vector{Matrix{T}}) where T
    N, D = size(H_set[1])
    M = length(H_set)

    H = Matrix{T}(undef, N*M, D)

    st = 0
    fin = 0

    for m = 1:M
        st = fin +1
        fin = st + N -1
        H[st:fin,:] = H_set[m]
    end

    return H
end


function ∂ψtoblockdiagmatrix(∂ψ_∂θ::Vector{Vector{Vector{T}}})::Vector{Matrix{T}} where T
    N = length(∂ψ_∂θ)
    M = length(∂ψ_∂θ[1])
    D = length(∂ψ_∂θ[1][1])

    # 𝐻 = Matrix{T}(undef, M*N, D)
    # i = 0
    # for m = 1:M
    #     for n = 1:N
    #         i += 1
    #
    #         𝐻[i,:] = ∂ψ_∂θ[n][m]
    #     end
    # end

    𝐻_set = Vector{Matrix{T}}(undef, M)
    for m = 1:M
        𝐻_set[m] = Matrix{T}(undef, N, D)

        for n = 1:N
            𝐻_set[m][n,:] = ∂ψ_∂θ[n][m]
        end
    end

    return 𝐻_set
end

function nestedvectorswapindices(y::Vector{Vector{T}})::Vector{Vector{T}} where T
    N = length(y)
    M = length(y[1])

    return collect( collect( y[n][m] for n = 1:N ) for m = 1:M )
end




### legacy Ht.
function gethessianfuncs(ψ::Function, D_y::Int)::Vector{Function}

    return collect( xx->ForwardDiff.hessian( yy->ψ(yy)[i], xx) for i = 1:D_y )
end

function gethessianfuncsND(ψ::Function, D_y::Int)::Vector{Function}

    return collect( xx->Calculus.hessian( yy->ψ(yy)[i], xx) for i = 1:D_y )
end

# From the text near equation 28.
function compute∂𝐻tover∂xj(hessian_array::Vector{Matrix{T}}, j::Int) where T <: Real
    D_y = length(hessian_array)
    D_x = size(hessian_array[1],1)

    ∂𝐻t_∂xj = Matrix{T}(undef,D_x,D_y)
    for i = 1:D_y
        ∂𝐻t_∂xj[:,i] = hessian_array[i][:,j]
    end

    return ∂𝐻t_∂xj
end

function compute∂𝐻tover∂x(  hessian_funcs::Vector{Function},
                            x::Vector{T},
                            D_y::Int) where T <: Real

    # get the derivatives.
    hessian_array = collect( hessian_funcs[i](x) for i = 1:D_y )

    return collect( compute∂𝐻tover∂xj(hessian_array, j) for j = 1:length(x) )
end




# numerical derivatives.
function eval∂2ψND( x::Vector{T}, ψ::Function, D_y::Int)::Vector{Vector{T}} where T

    D_x = length(x)

    # get component functions
    ψ_components = Vector{Function}(undef, D_y)
    for j = 1:D_y
        ψ_components[j] = xx->ψ(xx)[j]
    end

    # get second derivatives.
    ∂2ψ_x = Vector{Vector{T}}(undef, D_y)

    for j = 1:D_y
        A = Calculus.hessian(ψ_components[j], x)
        ∂2ψ_x[j] = Utilities.packageuppertriangle(A)
    end


    return ∂2ψ_x
end
