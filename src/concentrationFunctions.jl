using Integrals
using LinearAlgebra

"""
# ConcentrationFunctions.jl

This file contains methods for computing the actual concentration using interpolation and reconstruction kernels.
"""

"""
    concentration(x1_interpol, x2_interpol, x3_interpol, x1_eval, x2_eval, x3_eval, ALPHA, beta, ::Type{ReconstructionKernel}, epsilon_reco, ::Type{InterpolationKernel}, epsilon_interpol)

Compute the concentration given interpolation points, evaluation points, interpolation coefficients, and reconstruction parameters.

# Arguments
- `x1_interpol::Vector{Float64}`: Vector of x-coordinates for interpolation points.
- `x2_interpol::Vector{Float64}`: Vector of y-coordinates for interpolation points.
- `x3_interpol::Vector{Float64}`: Vector of z-coordinates for interpolation points.
- `x1_eval::Vector{Float64}`: Vector of x-coordinates for evaluation points.
- `x2_eval::Vector{Float64}`: Vector of y-coordinates for evaluation points.
- `x3_eval::Vector{Float64}`: Vector of z-coordinates for evaluation points.
- `ALPHA::Matrix{Float64}`: `n × m` Matrix containing the interpolation coefficients of the system matrices, where `n` is the number of grid points and `m` is the number of forward operators.
- `beta::Vector{Float64}`: Vector of length `m`, containing the computed coefficients for the basis functions in the kernel-based reconstruction.
- `::Type{ReconstructionKernel}`: The type of kernel function to be used for reconstruction (`k^{\text{reco}}`).
- `epsilon_reco::Float64`: Shape parameter for the reconstruction kernel.
- `::Type{InterpolationKernel}`: The type of kernel function to be used for interpolation (`k^{\text{interpol}}`).
- `epsilon_interpol::Float64`: Shape parameter for the interpolation kernel.

# Returns
- `concentration::Vector{Float64}`: The computed concentration values at the evaluation points.
"""
function concentration(x1_interpol::Vector{Float64}, x2_interpol::Vector{Float64}, x3_interpol::Vector{Float64}, x1_eval::Vector{Float64}, x2_eval::Vector{Float64}, x3_eval::Vector{Float64}, ALPHA::Matrix{Float64}, beta::Vector{Float64},::Type{ReconstructionKernel}, epsilon_reco::Float64, ::Type{InterpolationKernel}, epsilon_interpol::Float64) where {ReconstructionKernel<:AbstractKernel, InterpolationKernel<:AbstractKernel}
    K_conv = computeConvolutionalKernel(x1_interpol, x2_interpol, x3_interpol, x1_eval, x2_eval, x3_eval, ReconstructionKernel, epsilon_reco, InterpolationKernel, epsilon_interpol)
    return (ALPHA' * K_conv)' * beta
end

"""
    computeConvolutionalKernel(x_interpol, x_eval, ::Type{ReconstructionKernel}, epsilon_reco::Float64, ::Type{InterpolationKernel}, epsilon_interpol::Float64)

Compute the convolutional kernel matrix given interpolation and evaluation points, and kernel parameters.

# Arguments
- `x_interpol::Vector{Vector{Float64}}`: Nested vector where each sub-vector contains coordinates for interpolation points.
- `x_eval::Vector{Vector{Float64}}`: Nested vector where each sub-vector contains coordinates for evaluation points.
- `::Type{ReconstructionKernel}`: The type of kernel function to be used for reconstruction (`k^{\text{reco}}`).
- `epsilon_reco::Float64`: Shape parameter for the reconstruction kernel.
- `::Type{InterpolationKernel}`: The type of kernel function to be used for interpolation (`k^{\text{interpol}}`).
- `epsilon_interpol::Float64`: Shape parameter for the interpolation kernel.

# Returns
- `K_conv::Matrix{Float64}`: The computed convolutional kernel matrix.
"""
function computeConvolutionalKernel(x_interpol, x_eval, ::Type{ReconstructionKernel}, epsilon_reco::Float64, ::Type{InterpolationKernel}, epsilon_interpol::Float64) where {ReconstructionKernel<:AbstractKernel, InterpolationKernel<:AbstractKernel}
    integral_dimension = length(x_interpol[1])
    k(u,p) = [kernel(u, x_interpol[i], epsilon_interpol, InterpolationKernel) * kernel(x_eval[j], u, epsilon_reco, ReconstructionKernel) for i in eachindex(x_interpol), j in eachindex(x_eval)]
    domain = (-ones(integral_dimension), ones(integral_dimension))
    prob = IntegralProblem(k, domain)
    @time sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
    return sol.u
end

"""
    computeConvolutionalKernel(x1_interpol::Vector{Float64}, x2_interpol::Vector{Float64}, x3_interpol::Vector{Float64}, x1_eval::Vector{Float64}, x2_eval::Vector{Float64}, x3_eval::Vector{Float64}, ::Type{ReconstructionKernel}, epsilon_reco::Float64, ::Type{InterpolationKernel}, epsilon_interpol::Float64)

Compute the convolutional kernel matrix for 3D interpolation and evaluation points with separable kernels.

# Arguments
- `x1_interpol::Vector{Float64}`: Vector of x-coordinates for interpolation points.
- `x2_interpol::Vector{Float64}`: Vector of y-coordinates for interpolation points.
- `x3_interpol::Vector{Float64}`: Vector of z-coordinates for interpolation points.
- `x1_eval::Vector{Float64}`: Vector of x-coordinates for evaluation points.
- `x2_eval::Vector{Float64}`: Vector of y-coordinates for evaluation points.
- `x3_eval::Vector{Float64}`: Vector of z-coordinates for evaluation points.
- `::Type{ReconstructionKernel}`: The type of kernel function to be used for reconstruction (`k^{\text{reco}}`).
- `epsilon_reco::Float64`: Shape parameter for the reconstruction kernel.
- `::Type{InterpolationKernel}`: The type of kernel function to be used for interpolation (`k^{\text{interpol}}`).
- `epsilon_interpol::Float64`: Shape parameter for the interpolation kernel.

# Returns
- `K_conv::Matrix{Float64}`: The computed convolutional kernel matrix for the 3D case.
"""
function computeConvolutionalKernel(x1_interpol::Vector{Float64}, x2_interpol::Vector{Float64}, x3_interpol::Vector{Float64}, x1_eval::Vector{Float64}, x2_eval::Vector{Float64}, x3_eval::Vector{Float64}, ::Type{ReconstructionKernel}, epsilon_reco::Float64, ::Type{InterpolationKernel}, epsilon_interpol::Float64) where {ReconstructionKernel<:AbstractSeperableKernel, InterpolationKernel<:AbstractSeperableKernel}
    K1 = computeConvolutionalKernel(x1_interpol, x1_eval, ReconstructionKernel, epsilon_reco, InterpolationKernel, epsilon_interpol)
    K2 = computeConvolutionalKernel(x2_interpol, x2_eval, ReconstructionKernel, epsilon_reco, InterpolationKernel, epsilon_interpol)
    K3 = computeConvolutionalKernel(x3_interpol, x3_eval, ReconstructionKernel, epsilon_reco, InterpolationKernel, epsilon_interpol)

    K = [K1[i1, j1] * K2[i2, j2] * K3[i3, j3]
         for i1 in eachindex(x1_interpol),
         i2 in eachindex(x2_interpol),
         i3 in eachindex(x3_interpol),
         j1 in eachindex(x1_eval),
         j2 in eachindex(x2_eval),
         j3 in eachindex(x3_eval)]
    return reshape(K, length(x1_interpol) * length(x2_interpol) * length(x3_interpol), length(x1_eval) * length(x2_eval) * length(x3_eval))
end
