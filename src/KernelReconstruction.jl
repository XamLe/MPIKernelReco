"""
# KernelReconstruction

File containing functions for the kernel-based reconstruction.
"""

"""
    kernelInterpolate1D(f,x)

Interpolates a function \(f: \mathbb{R}^d \to \mathbb{R}\), whose values are known in interpolation points \(x_1, \ldots, x_n \in \mathbb{R}^d\) by applying a kernel-based approach.
\((f_1, \ldots, f_n) := (f(x_1), \ldots, f(x_n))\).
The interpolant has the form \(s(x) = \sum_{i = 1}^{n} \beta_i K(x, x_i)\).
\(K(x,y): \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}\) is a symmetric, positive-definite kernel function.
\(\beta_1, \ldots, \beta_n \in \mathbb{R}\) are coefficients.
The interpolant satisfies \(s(x_i) = f_i\) for all \(i = 1, \ldots, n\).

# Arguments
- `f`: Vector of length n containing the values of f_1, ..., f_n
- `x`: Vector of length n containing the interpolation points x_1, ..., x_n

# Returns
- `beta`: Vector of length n containing the coefficients corresponding to the kernel centered at the respective interpolation point x_i
"""
function kernelInterpolate1D(f::Vector{Float64}, x::Vector{Tuple{Float64, Float64, Float64}})

end


"""
    assembleKernelInterpolationMatrix(x)

Assembles the interpolation matrix in the context of kernel-based interpolation.

# Arguments
- `x`: Vector of length n containing the interpolation points x_1, ..., x_n

# Returns
- `A`: Matrix with entries a_{ij} = K(x_i, x_j)
"""
function assembleKernelInterpolationMatrix(x::Vector{Tuple{Float64, Float64, Float64}})

end
