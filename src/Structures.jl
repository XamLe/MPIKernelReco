struct ExperimentData
    systemMatrixPath::String
    systemMatrixGrid::Vector{Vector{Float64}}
    kernelType::String
    shapeParameter::Float64
    kernelMatrixSystemGrid::Matrix{Float64}
    convKernel
    convConvKernel
    systemMatrixLinearlyIndependent::Matrix{Float32}
    indicesOfLinearlyIndependentRows
    basisFunctions::Matrix{Float64}
    reconstructionMatrix::Matrix{Float64}
end


# Define an abstract type AbstractKernel
abstract type AbstractKernel end

# Define an abstract subtype AbstractSeparableKernel of AbstractKernel
abstract type AbstractSeparableKernel <: AbstractKernel end

# Define a concrete subtype GaussianKernel of AbstractSeparableKernel
struct GaussianKernel <: AbstractSeparableKernel end

# Define a concrete subtype MultiquadricKernel of AbstractKernel
struct MultiquadricKernel <: AbstractKernel end

# Define a concrete subtype InverseMultiquadricKernel of AbstractKernel
struct InverseMultiquadricKernel <: AbstractKernel end

# Define a concrete subtype Wendland0Kernel of AbstractKernel
struct Wendland0Kernel <: AbstractSeparableKernel end

# Define a concrete subtype TensorWendland0Kernel of AbstractSeparableKernel
struct TensorWendland0Kernel <: AbstractSeparableKernel end

# Define a concrete subtype AnisotropicGaussianKernel of Abstract Kernel
struct AnisotropicGaussianKernel <: AbstractKernel end

struct PolynomialKernel <: AbstractKernel end

mutable struct ProblemData{K1<:AbstractKernel}
    systemMatrix::AbstractMatrix
    inherentGrid::Vector{<:AbstractVector}
    voxelVolume::Float64
    voxelSize::Array{Float64,1}#computed from FOV and numVoxels
    evaluationGrid::Vector{<:AbstractVector}
    epsilon::Float64
    reconstructionKernel::K1
    u::Vector{<:Union{Float64,ComplexF64}}
    FOV::Array{Float64,2}#computed from fov and fovcenter
    rX::Array{Float64,1}#x components of center points of all voxels
    rY::Array{Float64,1}
    rZ::Array{Float64,1}
    weightNormalization::Bool
    isWeightNormalized::Bool
end

struct ReconstructionResult
    concentration::Array{Float64,1}
    rieszRepresenters::Array{Float64,2} # Basis Functions evaluated in the evaluation Grid
    coefB::Array{Float64,1} # Coefficient Vector with weights of the Riesz Representers / Basis Functions
    A::Array{Float64,2} # Gram Matrix / Reconstruction Matrix
    residual::Array{Float64,1}
    interpolantNorm::Float64
end
