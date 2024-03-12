"""
# Kernels

File containing kernel functions.

Author: Max Lewerenz
Email: max.lewerenz@gmail.com
Date Created: 07.03.2024
"""

using LinearAlgebra

# Define an abstract type AbstractKernel
abstract type AbstractKernel end

# Define a concrete subtype ExponentialKernel of AbstractKernel
struct ExponentialKernel <: AbstractKernel end

# Define a concrete subtype MultiquadricKernel of AbstractKernel
struct MultiquadricKernel <: AbstractKernel end

"""
    kernel(x, y, epsilon, k::Type{Exponential Kernel})

Exponential kernel function

# Arguments
- `x`: First argument to insert into the kernel function
- `y`: Second argument to insert into the kernel function k
- `k::Type{ExponentialKernel}`: Type hint for dispatching to the  ExponentialKernel method.

# Returns
- The kernel value between `x` and `y` using the Exponential Kernel. \( K(x,y) = \exp(-\varepsilon \|x - y\|^2) \)
"""
function kernel(x::Vector{T}, y::Vector{T}, ::Type{ExponentialKernel}) where T
    return exp(- norm(x - y)^2)
end

