"""
# Kernels

File containing kernel functions.

Author: Max Lewerenz
Email: max.lewerenz@gmail.com
Date Created: 07.03.2024
"""

using LinearAlgebra

"""
    kernel(x, y, k::Type{GaussianKernel})

Gaussian kernel function

# Arguments
- `x`: First argument to insert into the kernel function
- `y`: Second argument to insert into the kernel function k
- `k::Type{GaussianKernel}`: Type hint for dispatching to the  GaussianKernel method.

# Returns
- The kernel value between `x` and `y` using the Gaussian Kernel. ``K(x,y) = \\exp(-\\|x - y\\|^2)``
"""
function kernel(x, y, epsilon ,::Type{GaussianKernel})
    value = exp(- epsilon * norm(x .- y)^2)
        return value
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
function kernel(x, y, epsilon, ::Type{MultiquadricKernel})
    return sqrt(1 + epsilon * norm(x - y)^2)
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
function kernel(x, y, epsilon, ::Type{InverseMultiquadricKernel})
    return 1 / kernel(x, y, epsilon, MultiquadricKernel)
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
function kernel(x::Float64, y::Float64, epsilon::Float64, ::Type{Wendland0Kernel})
    return max(0, 1 - epsilon * norm(x - y))
end

"""
    kernel(x::Vector{Float64}, y::Vector{Float64}, epsilon::Float64, ::Type{TensorWendland0Kernel})

Computes the product of the `Wendland0Kernel` evaluated at each corresponding pair of elements from vectors `x` and `y`. This implements a **tensor product kernel** using the `Wendland0Kernel` for each dimension.

# Arguments
- `x::Vector{Float64}`: The first input vector (e.g., point in multi-dimensional space).
- `y::Vector{Float64}`: The second input vector, of the same length as `x`.
- `epsilon::Float64`: A shape parameter for the kernel function.
- `::Type{TensorWendland0Kernel}`: Specifies the type of kernel being used (here, the `TensorWendland0Kernel`).

# Returns
- `value::Float64`: The product of the `Wendland0Kernel` values for each dimension.

# Notes
- The function assumes that both `x` and `y` are vectors of the same length.
- The kernel function used in the product is the `Wendland0Kernel`, which should be defined elsewhere in your code.
"""
function kernel(x::Vector{Float64}, y::Vector{Float64}, epsilon::Float64, ::Type{TensorWendland0Kernel})
    value = 1
    for i in 1:size(x,1)
        value *= kernel(x[i], y[i], epsilon, Wendland0Kernel)
    end
    return value
end

"""
    kernel(x::Vector{Float64}, y::Vector{Float64}, epsilon::Float64, ::Type{AnisotropicGaussianKernel})

Computes an **anisotropic Gaussian kernel** between two vectors `x` and `y`. This kernel applies the `GaussianKernel` separately to the first two dimensions and the third dimension, scaling the third dimension by `2 * epsilon`.

# Arguments
- `x::Vector{Float64}`: The first input vector, assumed to have at least 3 elements.
- `y::Vector{Float64}`: The second input vector, also assumed to have at least 3 elements.
- `epsilon::Float64`: A shape parameter for the kernel function.
- `::Type{AnisotropicGaussianKernel}`: Specifies that the kernel is an `AnisotropicGaussianKernel`.

# Returns
- `value::Float64`: The product of the Gaussian kernels applied to the first two dimensions of `x` and `y` and to the third dimension, with a modified `epsilon` for the third dimension.

# Notes
- The function assumes that the vectors `x` and `y` have at least 3 elements. It applies the `GaussianKernel` on the first two dimensions of the input vectors with the standard `epsilon` and on the third dimension with `2 * epsilon`, effectively scaling the kernel anisotropically.
"""
function kernel(x::Vector{Float64}, y::Vector{Float64}, epsilon::Float64, ::Type{AnisotropicGaussianKernel})
    return kernel(x[1:2], y[1:2], epsilon, GaussianKernel) * kernel(x[3], y[3], 2 * epsilon, GaussianKernel)
end

function kernel(x::Vector{Float64}, y::Vector{Float64}, epsilon::Float64, ::Type{PolynomialKernel})
    c = 1
    d = 2
    return (epsilon * transpose(x) * y + c).^d
end

"""
    computeKernelMatrix(grid1, grid2, epsilon::Float64, ::Type{KernelType}) where {KernelType <: AbstractKernel}

Computes the **kernel matrix** between two grids of points `grid1` and `grid2` using a specified kernel type `KernelType`. The kernel matrix contains the pairwise evaluations of the kernel function applied to each point in `grid1` and each point in `grid2`.

# Arguments
- `grid1`: A collection of points (e.g., vectors) where each element represents a point in space.
- `grid2`: Another collection of points, of the same type as `grid1`.
- `epsilon::Float64`: A shape parameter for the kernel function.
- `::Type{KernelType}`: The type of kernel to be used, which must be a subtype of `AbstractKernel`. This type determines which specific kernel function (e.g., `GaussianKernel`, `WendlandKernel`, etc.) will be used in the pairwise evaluations.

# Returns
- `matrix::Matrix{Float64}`: A matrix where each entry `matrix[i, j]` is the result of applying the kernel function to the points `grid1[i]` and `grid2[j]`.

# Notes
- The kernel function is applied element-wise between all pairs of points from `grid1` and `grid2`. Each point is assumed to be a vector or similar data structure that can be passed to the specified kernel.
- The function `kernel` must be defined for the specified `KernelType`, and it should accept the same arguments (e.g., two points and `epsilon`).
"""
function computeKernelMatrix(grid1, grid2, epsilon::Float64, ::Type{KernelType}) where {KernelType <: AbstractKernel}
    return [kernel(r1, r2, epsilon, KernelType) for r1 in grid1, r2 in grid2]
end

# Computes the evaluations of the Riesz representers on a given evaluation grid.
#
# This function applies the Riesz representer associated with a linear functional,
# represented in the form of the sampling matrix `SM`, to a kernel Gram matrix.
# Specifically, it computes the product of the transpose of `SM` with the Gram
# matrix constructed between `gridEval` (evaluation points) and `gridInherent`
# (kernel centers), using a kernel of type `KernelType` with shape parameter `epsilon`.
#
# Arguments:
# - gridEval     : Points at which to evaluate the Riesz representers.
# - gridInherent : Points representing the centers of the system matrix measurements.
# - SM           : System matrix representing the linear functionals.
# - epsilon      : Shape parameter for the kernel function.
# - KernelType   : A subtype of `AbstractKernel` specifying the kernel to use.
#
# Returns:
# - A matrix of Riesz representer evaluations at the points in `gridEval`.
#
# Note: !!!!!!!
# = Duplicate of the computeBasiFunctions function, these can be combined
function computeRieszRepresenterEvaluations(gridEval::Vector{<:AbstractVector}, problemData::ProblemData{K}) where {K <: AbstractKernel}
    gramMatrix = computeKernelMatrix(gridEval, problemData.inherentGrid, problemData.epsilon, typeof(problemData.reconstructionKernel)) # TODO: instanz statt typ uebergeben
    return problemData.voxelVolume * transpose(gramMatrix) * problemData.systemMatrix
end
