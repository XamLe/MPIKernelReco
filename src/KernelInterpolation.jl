using IterativeSolvers
using LinearAlgebra

"""
# KernelInterpolation

File containing functions for the kernel-based interpolation of functions.

Author: Max Lewerenz
Email: max.lewerenz@gmail.com
Date Created: 07.03.2024
"""

"""
    getKernelInterpolationCoefficients(f, x, ::Type{KernelType})

Interpolates a function ``f: \\mathbb{R}^d \\to \\mathbb{R}``, whose values are known in interpolation points ``x_1, \\ldots, x_n \\in \\mathbb{R}^d`` by applying a kernel-based approach.
``(f_1, \\ldots, f_n) := (f(x_1), \\ldots, f(x_n))``.
The interpolant has the form ``s(x) = \\sum_{i = 1}^{n} \\beta_i K(x, x_i)``.
``K(x,y): \\mathbb{R}^d \\times \\mathbb{R}^d \\to \\mathbb{R}`` is a symmetric, positive-definite kernel function.
``\\beta_1, \\ldots, \\beta_n \\in \\mathbb{R}`` are coefficients.
The interpolant satisfies ``s(x_i) = f_i`` For all ``i = 1, \\ldots, n``.

# Arguments
- `f`: Vector of length n containing the values of ``f_1, ..., f_n``
- `x`: Vector of length n containing the interpolation points ``x_1, ..., x_n``
- `::Type{KernelType}`: The type of kernel function to be used for computing the kernel values.
  It should be a subtype of `AbstractKernel`.


# Returns
- `beta`: Vector of length n containing the coefficients corresponding to the kernel centered at the respective interpolation point x_i
"""
function getKernelInterpolationCoefficients(A::Matrix{Float64}, f::Vector, x, epsilon, ::Type{KernelType}) where KernelType<:AbstractKernel
    # println("Assemble interpolation matrix")
    # TODO: Interpolation matrix stays the same as long as the centers of interpolation stay the same. Perform Cholzki decomposition to speed up computation. Decomposition has to be done only once.
    # A = assembleKernelInterpolationMatrix(x, epsilon, KernelType)
    # println("Solve linear system with backslash")
    # @time beta = A'A \ A'f
    # NOTE: The backslash operator is multiple times faster than the conjugate gradient method. (3sec vs 50sec)
    # However it leads to more "noisy" solutions.
    #println("Solve linear system with cg and preconditioner A'")
    beta = cg(A'A, A'*f, maxiter = 20) # find a norm minimal solution to the problem
    return beta
end

"""
    getKernelInterpolationCoefficients(f::Vector, A::Cholesky)

returns coefficients to matching interpolation centers on the basis of a cholesky decomposed vandermonde matrix.

# Arguments
- `f::Vector`: Vector of length n containing the values of ``f_1, ..., f_n``
- `A::Cholesky`: A Cholesky structure containing the previously factorized Reconstruction matrix
"""
function getKernelInterpolationCoefficients(f::Vector, A::Cholesky)
	return A \ f
end


"""
    assembleKernelInterpolationMatrix(x::Vector{T}, ::Type{K}) where {T, K<:AbstractKernel}

Assemble an interpolation matrix using a specified kernel function.

This function constructs a matrix where each entry `A[i, j]` represents the kernel value between
the `i`-th and `j`-th points in the input vector `x`, using the specified kernel function `K`.

# Arguments
- `x::Tuple{Vararg{Vector{T}}}`: A tuple containing an abitrary number of Vectors of type `T`. Each point should be represented
  as a vector containing the coordinates of the point.
- `::Type{KernelType}`: The type of kernel function to be used for computing the kernel values.
  It should be a subtype of `AbstractKernel`.

# Returns
- A matrix `A` of size `n x n`, where `n` is the length of vector `x`, containing the kernel
  values between each pair of points in `x`.

# Example
```julia
x = ([1.0, 2.0, 0.1], [3.0, 4.0, 0.2], [5.0, 6.0, 0.3])  # Example points
A = assembleKernelInterpolationMatrix(x, ExponentialKernel)
```
"""
function assembleKernelInterpolationMatrix(x, epsilon, ::Type{KernelType}) where {KernelType<:AbstractKernel}
    n = length(x)
    A = [kernel(x[i], x[j], epsilon, KernelType) for i in 1:n, j in 1:n]
    return A
end


"""
    kernelInterpolant(evaluationPoint, interpolationPoints, coefficients, ::Type{KernelType})

computes the interpolant at a given point ``\\text{evaluationPoint} = x`` using a Kernel function with nodes ``\\text{interpolationPoints}``.
The interpolant is of the form ``s(x) = \\sum_{i = 1}^n \beta_i K(x, x_i)

# Arguments
- `evalutationPoint`: Point of evaluation ``x``
- `interpolationPoints`: Vector containing the interpolation points ``(x_1, \\ldots, x_n)``
- `coefficients`: Vector of coefficients ``\\beta_1, \\ldots, \\beta_n``
- `::Type{KernelType}`: Type hint for the kernel function `K`.

# Returns
- The value of the interpolant at `evaluationPoint` ``x``.
"""
function kernelInterpolant(evaluationPoint, interpolationPoints, coefficients, epsilon, ::Type{KernelType}) where {KernelType<:AbstractKernel}
    kernelValues = kernel.(Ref(evaluationPoint), interpolationPoints, epsilon, Ref(KernelType))
    return dot(coefficients, kernelValues)
end

"""

"""
function choleskyDecomposeInterpolationMatrix(x, epsilon, ::Type{KernelType}) where {KernelType<:AbstractKernel}
    A = assembleKernelInterpolationMatrix(x, epsilon, KernelType)
    smallest_eigenvalue = minimum(eigvals(A))
    # TODO: If smaller then a threshold and negative, else error or without addition of smallest eigenvalue
    println("Warning: diagonal of interpolation matrix is added with smallest eigenvalue: ", smallest_eigenvalue)
    A = cholesky(A + 2 * smallest_eigenvalue * I)
    # A = cholesky(A)
    return A
end
