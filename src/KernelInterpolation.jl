"""
# KernelInterpolation

File containing functions for the kernel-based interpolation of functions.

Author: Max Lewerenz
Email: max.lewerenz@gmail.com
Date Created: 07.03.2024
"""

"""
    kernelInterpolate1D(f,x)

Interpolates a function \(f: \mathbb{R}^d \to \mathbb{R}\), whose values are known in interpolation points \(x_1, \ldots, x_n \in \mathbb{R}^d\) by applying a kernel-based approach.
\((f_1, \ldots, f_n) := (f(x_1), \ldots, f(x_n))\).
The interpolant has the form \(s(x) = \sum_{i = 1}^{n} \beta_i K(x, x_i)\).
\(K(x,y): \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}\) is a symmetric, positive-definite kernel function.
\(\beta_1, \ldots, \beta_n \in \mathbb{R}\) are coefficients.
The interpolant satisfies \(s(x_i) = f_i\) For all \(i = 1, \ldots, n\).

# Arguments
- `f`: Vector of length n containing the values of f_1, ..., f_n
- `x`: Vector of length n containing the interpolation points x_1, ..., x_n
- `::Type{KernelType}`: The type of kernel function to be used for computing the kernel values.
  It should be a subtype of `AbstractKernel`.


# Returns
- `beta`: Vector of length n containing the coefficients corresponding to the kernel centered at the respective interpolation point x_i
"""
function kernelInterpolate1D(f::Vector{Float64}, x::Tuple{Vararg{Vector{T}}}, ::Type{KernelType}) where {T, KernelType<:AbstractKernel}
    A = assembleKernelInterpolationMatrix(x, KernelType)
    beta = A \ f
    return beta
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
x = ([1.0, 2.0, 0.1], [3.0, 4.0, 0.2], [5.0, 6.0, 0.3])  # Example points with epsilon values
A = assembleKernelInterpolationMatrix(x, ExponentialKernel)
```
"""
function assembleKernelInterpolationMatrix(x::Tuple{Vararg{Vector{T}}}, ::Type{KernelType}) where {T,KernelType<:AbstractKernel}
    n = length(x)
    A = [kernel(x[i],x[j],KernelType) for i in 1:n, j in 1:n]
    return A
end
