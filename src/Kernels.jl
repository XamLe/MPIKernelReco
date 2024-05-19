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

# Define an abstract subtype AbstractSeparableKernel of AbstractKernel
abstract type AbstractSeperableKernel <: AbstractKernel end

# Define a concrete subtype ExponentialKernel of AbstractSeperableKernel
struct ExponentialKernel <: AbstractSeperableKernel end

# Define a concrete subtype MultiquadricKernel of AbstractKernel
struct MultiquadricKernel <: AbstractKernel end

# Define a concrete subtype InverseMultiquadricKernel of AbstractKernel
struct InverseMultiquadricKernel <: AbstractKernel end

# Define a concrete subtype Wendland0Kernel of AbstractKernel
struct Wendland0Kernel <: AbstractKernel end

"""
    kernel(x, y, k::Type{ExponentialKernel})

Exponential kernel function

# Arguments
- `x`: First argument to insert into the kernel function
- `y`: Second argument to insert into the kernel function k
- `k::Type{ExponentialKernel}`: Type hint for dispatching to the  ExponentialKernel method.

# Returns
- The kernel value between `x` and `y` using the Exponential Kernel. ``K(x,y) = \\exp(-\\|x - y\\|^2)``
"""
function kernel(x, y, epsilon ,::Type{ExponentialKernel})
    return exp(- epsilon * norm(x .- y)^2)
end


"""
    kernel(x, y, k::Type{MultiquadricKernel})

Multiquadric kernel function

# Arguments
- `x`: First argument to insert into the kernel function
- `y`: Second argument to insert into the kernel function k
- `k::Type{MultiquadricKernel}`: Type hint for dispatching to the MultiquadricKernel method.

# Returns
- The kernel value between `x` and `y` using the Multiquadrtic Kernel. ``K(x,y) = \\sqrt(1 + (\\varepsilon \\|x - y \\|)^2)``
"""
function kernel(x, y, ::Type{MultiquadricKernel})
    return sqrt(1 + norm(x - y)^2)
end


"""
    kernel(x, y, k::Type{InverseMultiquadricKernel})

Inverse Multiquadric kernel function

# Arguments
- `x`: First argument to insert into the kernel function
- `y`: Second argument to insert into the kernel function k
- `k::Type{InverseMultiquadricKernel}`: Type hint for dispatching to the InverseMultiquadricKernel method.

# Returns
- The kernel value between `x` and `y` using the Inverse Multiquadrtic Kernel. ``K(x,y) = 1 / \\sqrt(1 + (\\varepsilon \\|x - y \\|)^2)``
"""
function kernel(x, y, ::Type{InverseMultiquadricKernel})
    return 1 / kernel(x, y, MultiquadricKernel)
end

"""
    kernel(x, y, k::Type{Wendland0Kernel})

Wendland kernel of order 0.

# Arguments
- `x`: First argument to insert into the kernel function
- `y`: Second argument to insert into the kernel function k
- `k::Type{Wendland0Kernel}`: Type hint for dispatching to the Wendland0Kernel method.

# Returns
- The kernel value between `x` and `y` using the Wendland Kernel of order 0. ``K(x,y) = \\max(0, 1 - \\|x - y \\|)``
"""
function kernel(x, y, epsilon::Float64, ::Type{Wendland0Kernel})
    return max(0, 1 - norm(x - y))
end
