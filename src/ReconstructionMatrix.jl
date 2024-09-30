using Integrals
using LinearAlgebra

"""
 ReconstructionMatrix

File to contain functions for assembling the Reconstruction matrix.

Author: Max Lewerenz
Email: max.lewerenz@gmail.com
Date Created: 07.03.2024
"""

"""
    assembleReconstructionMatrix(ALPHA::Matrix{Float64}, x1::Vector{Float64}, x2::Vector{Float64}, x3::Vector{Float64},
                                 ::Type{ReconstructionKernel}, epsilon_reco::Float64,
                                 ::Type{InterpolationKernel}, epsilon_interpol::Float64)
                                 -> Matrix{Float64}

Assembles the reconstruction matrix using the interpolation and reconstruction kernels.

# Arguments
- `ALPHA::Matrix{Float64}`: The computed interpolation coefficients matrix.
- `x1::Vector{Float64}`: The x-coordinates for the grid points.
- `x2::Vector{Float64}`: The y-coordinates for the grid points.
- `x3::Vector{Float64}`: The z-coordinates for the grid points.
- `ReconstructionKernel::Type{<:AbstractKernel}`: The type of kernel used for reconstruction.
- `epsilon_reco::Float64`: The shape parameter for the reconstruction kernel.
- `InterpolationKernel::Type{<:AbstractKernel}`: The type of kernel used for interpolation.
- `epsilon_interpol::Float64`: The shape parameter for the interpolation kernel.

# Returns
- `Matrix{Float64}`: The assembled reconstruction matrix.
"""
function assembleReconstructionMatrix(ALPHA::Matrix{Float64}, x1::Vector{Float64}, x2::Vector{Float64}, x3::Vector{Float64}, ::Type{ReconstructionKernel}, epsilon_reco::Float64, ::Type{InterpolationKernel}, epsilon_interpol::Float64) where {ReconstructionKernel<:AbstractKernel, InterpolationKernel<:AbstractKernel}
    K = computeConvolvedConvolutionalKernel(x1, x2, x3, ReconstructionKernel, epsilon_reco, InterpolationKernel, epsilon_interpol)
    return ALPHA'*K*ALPHA
end

"""
    computeConvolvedConvolutionalKernel(xi, xj, reconstructionKernel, interpolationKernel)

computes the integral ``k_{ij} = \\int_{[-1,1]^3} \\int_{[-1,1]^3} k^{\\text{reco}}(x', x) k^{\\text{interpol}}(x,x_{j}) k^{\\text{interpol}}(x',x_i) dx dx' ``
This integral is needed in the computation of the reconstruction matrix entries.

# Arguments
- `xi`: ``x_{i}`` representing a point in space, in particular an interpolation point of the system functions.
- `xj`: ``x_{j}`` representing a point in space, in particular an interpolation point of the system functions.
- `::Type{ReconstructionKernel}`: The type of kernel function to be used for ``k^{\\text{reco}}``
- `epsilon_reco`: Shape parameter for the reconstruction kernel
- `::Type{InterpolationKernel}`: The type of kernel function to be used for ``k^{\\text{interpol}}``
- `epsilon_interpol`: Shape parameter for the interpolation kernel

# Returns
- `k_{ij}`: The value of the integral
"""
function computeConvolvedConvolutionalKernel(xi, xj, ::Type{ReconstructionKernel}, epsilon_reco::Float64, ::Type{InterpolationKernel}, epsilon_interpol::Float64) where  {ReconstructionKernel<:AbstractKernel, InterpolationKernel<:AbstractKernel}
    # This assumes a 3Dimensional domain [-1,1]^3
    k(u,p) = kernel(u[1:3], u[4:6], epsilon_reco, ReconstructionKernel) * kernel(u[1:3], xj, epsilon_interpol, InterpolationKernel) * kernel(u[4:6], xi, epsilon_interpol, InterpolationKernel)
    domain = (-ones(6), ones(6))
    prob = IntegralProblem(k, domain)
    sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
	return sol
end


"""
    computeConvolvedConvolutionalKernel(x::Vector{Vector{Float64}}, ::Type{ReconstructionKernel}, epsilon_reco, ::Type{InterpolationKernel}, epsilon_interpol)

computes the integral ``k_{ij} = \\int_{[-1,1]^3} \\int_{[-1,1]^3} k^{\\text{reco}}(x', x) k^{\\text{interpol}}(x,x_{j}) k^{\\text{interpol}}(x',x_i) dx dx' `` for all given ``x = (x_{1}, \\ldots, x_{n}) \\subset \\mathbb{R}^{3}``
This integral is needed in the computation of the reconstruction matrix entries.

# Arguments
- `x::Vector{Vector{Float64}}`: ``x = (x_{1}, \\ldots, x_{n}) \\subset \\mathbb{R}^{3}`` points in which the integral is to be evaluated
- `::Type{ReconstructionKernel}`: The type of kernel function to be used for ``k^{\\text{reco}}``
- `epsilon_reco`: Shape parameter for the reconstruction kernel
- `::Type{InterpolationKernel}`: The type of kernel function to be used for ``k^{\\text{interpol}}``
- `epsilon_interpol`: Shape parameter for the interpolation kernel

# Returns
"""
function computeConvolvedConvolutionalKernel(x::Vector{Vector{Float64}}, ::Type{ReconstructionKernel}, epsilon_reco::Float64, ::Type{InterpolationKernel}, epsilon_interpol::Float64, domain) where {ReconstructionKernel<:AbstractKernel, InterpolationKernel<:AbstractKernel}
    # As the convolvedConvolutionalKernel is symmetric, we only need to compute the lower triagonal matrix.
    k(u,p) = [i >= j ? kernel(u[1:3], u[4:6], epsilon_reco, ReconstructionKernel) * kernel(u[1:3], x[j], epsilon_interpol, InterpolationKernel) * kernel(u[4:6], x[i], epsilon_interpol, InterpolationKernel) : 0.0 for i in eachindex(x), j in eachindex(x)]
	# domain = (-ones(6), ones(6))
	prob = IntegralProblem(k, domain)
	# @time sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
	@time sol = solve(prob, HCubatureJL(), reltol = 1e-3)
	return sol.u' + sol.u - Diagonal(diag(sol.u))
end

"""
    computeConvolvedConvolutionalKernel(x::Vector{Float64}, ::Type{ReconstructionKernel}, epsilon_reco, ::Type{InterpolationKernel}, epsilon_interpol)

computes the integral ``k_{ij} = \\int_{[-1,1]^3} \\int_{[-1,1]^3} k^{\\text{reco}}(x', x) k^{\\text{interpol}}(x,x_{j}) k^{\\text{interpol}}(x',x_i) dx dx' `` for all given ``x = (x_{1}, \\ldots, x_{n}) \\subset \\mathbb{R}^{3}``
This integral is needed in the computation of the reconstruction matrix entries.

# Arguments
- `x::Vector{Float64}`: ``x = (x_{1}, \\ldots, x_{n}) \\subset \\mathbb{R}`` points in which the integral is to be evaluated
- `::Type{ReconstructionKernel}`: The type of kernel function to be used for ``k^{\\text{reco}}``
- `epsilon_reco`: Shape parameter for the reconstruction kernel
- `::Type{InterpolationKernel}`: The type of kernel function to be used for ``k^{\\text{interpol}}``
- `epsilon_interpol`: Shape parameter for the interpolation kernel

# Returns
"""
function computeConvolvedConvolutionalKernel(x::Vector{Float64}, ::Type{ReconstructionKernel}, epsilon_reco::Float64, ::Type{InterpolationKernel}, epsilon_interpol::Float64) where {ReconstructionKernel<:AbstractSeparableKernel, InterpolationKernel<:AbstractSeparableKernel}
    # As the convolvedConvolutionalKernel is symmetric, we only need to compute the lower triagonal matrix.
    k(u,p) = [i >= j ? kernel(u[1], u[2], epsilon_reco, ReconstructionKernel) * kernel(u[1], x[j], epsilon_interpol, InterpolationKernel) * kernel(u[2], x[i], epsilon_interpol, InterpolationKernel) : 0.0 for i in eachindex(x), j in eachindex(x)]
    # domain = (-ones(2), ones(2))
	domain = ([x[1], x[1]], [x[end], x[end]])
	prob = IntegralProblem(k, domain)
	@time sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
	return sol.u' + sol.u - Diagonal(diag(sol.u))
end

"""
    function computeConvolvedConvolutionalKernel(x1::Vector{Float64}, x2::Vector{Float64}, x3::Vector{Float64}, ::Type{ReconstructionKernel}, epsilon_reco::Float64, ::Type{InterpolationKernel}, epsilon_interpol::Float64) where {ReconstructionKernel<:AbstractSeparableKernel, InterpolationKernel<:AbstractSeparableKernel}

computes the integral
"""
function computeConvolvedConvolutionalKernel(x1::Vector{Float64}, x2::Vector{Float64}, x3::Vector{Float64}, ::Type{ReconstructionKernel}, epsilon_reco::Float64, ::Type{InterpolationKernel}, epsilon_interpol::Float64) where {ReconstructionKernel<:AbstractSeparableKernel, InterpolationKernel<:AbstractSeparableKernel}
    if (x1 == x2 && x2 == x3)
        K1 = computeConvolvedConvolutionalKernel(x1, ReconstructionKernel, epsilon_reco, InterpolationKernel, epsilon_interpol)
        K2 = K1
        K3 = K1
    else
        K1 = computeConvolvedConvolutionalKernel(x1, ReconstructionKernel, epsilon_reco, InterpolationKernel, epsilon_interpol)
        K2 = computeConvolvedConvolutionalKernel(x2, ReconstructionKernel, epsilon_reco, InterpolationKernel, epsilon_interpol)
        K3 = computeConvolvedConvolutionalKernel(x3, ReconstructionKernel, epsilon_reco, InterpolationKernel, epsilon_interpol)
    end

    K = [K1[i1, j1] * K2[i2,j2] * K3[i3,j3] for i1 in eachindex(x1), i2 in eachindex(x2), i3 in eachindex(x3), j1 in eachindex(x1), j2 in eachindex(x2), j3 in eachindex(x3)]
    return reshape(K, length(x1)*length(x2)*length(x3), length(x1)*length(x2)*length(x3))
end

"""
    computeConvolvedConvolutionalKernel(x, ::Type{Wendland0Kernel}, epsilon_reco::Float64, ::Type{Wendland0Kernel}, epsilon_interpol::Float64)

computes the integral ``k_{ij} = \\int_{[-1,1]^3} \\int_{[-1,1]^3} k^{\\text{reco}}(x', x) k^{\\text{interpol}}(x,x_{j}) k^{\\text{interpol}}(x',x_i) dx dx' `` for all given ``x = (x_{1}, \\ldots, x_{n}) \\subset \\mathbb{R}^{3}``,
where ``k^{\\text{reco}} (x,y) = k^{\\text{interpol}}(x,y) = \\max(0, 1 - \\|x - y \\|)`` is the Wendland Kernel.
"""
function computeConvolvedConvolutionalKernel(x, ::Type{ReconstructionKernel}, epsilon_reco::Float64, ::Type{InterpolationKernel}, epsilon_interpol::Float64) where {ReconstructionKernel<:Wendland0Kernel,InterpolationKernel<:Wendland0Kernel}
    println("This function was trigerred")
    return [(-1 * (x[i] .^ 3 .* x[j] .^ 2) ./ 3 .+ x[i] .^ 3 ./ 4 .+ (x[i] .^ 2 .* x[j] .^ 3) ./ 3 .+ x[i] .^ 2 .* x[j] .^ 2 - (x[i] .^ 2 .* x[j]) ./ 4 - 3 .* x[i] .^ 2 ./ 4 - (x[i] .* x[j] .^ 4) ./ 6 - (x[i] .* x[j] .^ 3) ./ 3 .+ (x[i] .* x[j]) ./ 4 .+ 5 .* x[i] ./ 96 .+ (x[j] .^ 5) ./ 30 .+ (x[j] .^ 4) ./ 6 - (7 .* x[j] .^ 2) ./ 12 - x[j] ./ 96 .+ 37 ./ 96) for i in eachindex(x), j in eachindex(x)]
end
