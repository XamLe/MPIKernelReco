using OpenMPIData
using OffsetArrays
using ImageFiltering
using IterativeSolvers
using Preconditioners

include("kaczmarzReconstruction.jl")

function mainStraightforwardKernelReco(patch = 10,
                                       numPatches = 19,
                                       filenameSM = joinpath(OpenMPIData.basedir(), "data", "calibrations", "2.mdf"),
                                       filenameMeas = joinpath(OpenMPIData.basedir(), "data", "measurements", "shapePhantom", "2.mdf"),
                                       ::Type{Kernel} = GaussianKernel,
                                       saveData::Bool = false;
                                       epsilon = 20000.0,
                                       lambda = 1e8,
                                       solver::String = "CGNormal",
                                       saveFigure = false,
                                       saveDataPath="./tmp/") where {Kernel <: AbstractKernel}

    """
    Read system matrix and measurements
    """
    #load the data and parameters
    u, measP, mfP, tPaC, tP = readParams(filenameMeas, "Measurement", numPatches=numPatches)
    measuredSM, smP = readParams(filenameSM, "SM", numPatches=numPatches)

    #process the data for reconstruction
    u, goodFreqIndex = MPIsimTools.processMeasurement(u, measP, tP, mfP)
    measuredSM = MPIsimTools.processSM(measuredSM, smP, tP, goodFreqIndex)

    u = reshape(u[:, 1:3, patch, :], length(goodFreqIndex) * 3)
    SM = convert(Array{Complex{Float32},2}, reshape(measuredSM[:, :, 1:3, :],
        smP.numVoxels[1] * smP.numVoxels[2] * smP.numVoxels[3], length(goodFreqIndex) * 3))

    """
    Perform reconstruction
    """
    return mainStraightforwardKernelReco(SM, u, smP, Kernel, epsilon, lambda, saveData; solver = solver, patch = patch, saveFigure = saveFigure, saveDataPath = saveDataPath)
end

"""
`mainStraightforwardKernelReco` Function

#Description:
The `mainStraightforwardKernelReco` function performs kernel-based reconstruction of an image from a system matrix and measurements. It reads the required data, processes it, assembles the reconstruction matrices, performs the reconstruction, and computes the final results.

#Arguments:
- `patch::Int` (default `10`): The patch index to use for the reconstruction.
- `numPatches::Int` (default `19`): The number of patches to read from the files.
- `filenameSM::String` (default `joinpath(OpenMPIData.basedir(), "data", "calibrations", "2.mdf")`): The file path for the system matrix.
- `filenameMeas::String` (default `joinpath(OpenMPIData.basedir(), "data", "measurements", "shapePhantom", "2.mdf")`): The file path for the measurements.
- `::Type{Kernel}` (default `GaussianKernel`): The kernel function to use for basis function computation.
- `epsilon::Float64` (default `60.0`): A parameter for the kernel function.

#Returns:
- `c_real::Array{Float64, 1}`: The reconstructed concentration values using the real part of the system matrix.
- `c_imag::Array{Float64, 1}`: The reconstructed concentration values using the imaginary part of the system matrix.
- `c_real_imag::Array{Float64, 1}`: The reconstructed concentration values using both the real and imaginary parts of the system matrix.

"""
function mainStraightforwardKernelReco(SM::Matrix,
                                       u::Vector,
                                       smP,
                                       ::Type{Kernel},
                                       epsilon::Float64,
                                       lambda::Float64,
                                       saveData::Bool;
                                       solver::String = "CGNormal",
                                       patch::Int = 10,
                                       saveFigure = false,
                                       saveDataPath = "./tmp") where {Kernel <: AbstractKernel}


    """
    0. Define Variables
    """
    println("0. Define Variables")

    voxelVolume = prod(smP.voxelSize)

    SM = hcat(real.(SM), imag.(SM))
    SM /= voxelVolume
    u = vcat(real.(u), imag.(u))

    SM, u = findLinearlyIndependent(SM, u, 1e-5)


    # voxelVolume = 1 # voxelVolume is aready contained in the Systemmatrix
    grid = [[x, y, z] for x in smP.rX, y in smP.rY, z in smP.rZ]
    grid = reshape(grid, length(smP.rX) * length(smP.rY) * length(smP.rZ))
    # grid *= 100
    # voxelVolume *= 100^3
    # x1 = [(2 * i - (smP.FOV[1, 1] + smP.FOV[1, 2])) / ((smP.FOV[1, 2] - smP.FOV[1, 1])) for i in smP.rX]
    # x2 = [(2 * i - (smP.FOV[2, 1] + smP.FOV[2, 2])) / ((smP.FOV[2, 2] - smP.FOV[2, 1])) for i in smP.rY]
    # x3 = [(i - (smP.FOV[3, 1] + smP.FOV[3, 2])) / ((smP.FOV[3, 2] - smP.FOV[3, 1])) for i in smP.rZ]
    # grid = [[x1[i], x2[j], x3[k]] for i in eachindex(smP.rX), j in eachindex(smP.rY), k in eachindex(smP.rZ)]
    # grid = reshape(grid, length(smP.rX) * length(smP.rY) * length(smP.rZ))

    kernelMatrix = computeKernelMatrix(grid, grid, epsilon, Kernel)

    """
    1. Assemble reconstruction matrix
    """
    println("1. Assemble reconstruction matrix")

    B = computeBasisFunctions(SM, kernelMatrix, voxelVolume)

    # A = voxelVolume * B * SM
    A = voxelVolume.^2 .* transpose(SM) * kernelMatrix * SM
    A = (transpose(A) + A) ./2

    println("Reconstruction Matrix assembled")

    # Save data to file
    if (false)
        writeKernelBasedRecoExperiment(joinpath(pwd(), "../data/exponential20000.h5"),
            "calibrations2.mdf",
            hcat(grid...),
            string(Kernel),
            epsilon,
            kernelMatrix,
            [1, 2, 3],
            [1, 2, 3],
            SM,
            [1, 2, 3],
            B,
            A)
    end

    """
    2. Perform the reconstruction
    """
    println("2. Perform the reconstruction")

    coefB = solveLinearSystem(A, u, solver; lambda = lambda)

    """
    3. Evaluate the reconstruction
    """
    println("3. Evaluate the reconstruction")

    # c = (transpose(B) * coefB)

    length_eval = [19, 19, 19]
    x1_eval = collect(range(smP.FOV[1, 1], smP.FOV[1, 2], length=length_eval[1])) # * 100
    x2_eval = collect(range(smP.FOV[2, 1], smP.FOV[2, 2], length=length_eval[2])) # * 100
    x3_eval = collect(range(smP.FOV[3, 1], smP.FOV[3, 2], length=length_eval[3])) # * 100
    # x3_eval = smP.rZ
    eval_grid = [[i, j, k] for i in x1_eval, j in x2_eval, k in x3_eval]
    eval_grid = reshape(eval_grid, prod(length_eval))
    KRecoEval = computeKernelMatrix(grid, eval_grid, epsilon, Kernel)
    c = transpose(voxelVolume * transpose(SM) * KRecoEval) * coefB
    println("Concentration computed")

    """
    4. Visualization
    """
    println("4. Visualization")
    p, plotCoefficients, plotConcentration, plotConcentrationPositive = plotCoefficientsAndConcentration(coefB, c, length(x1_eval), length(x2_eval), length(x3_eval), 10)
    if (saveFigure)
        println("Speichere die Plots")
        savefig(plotCoefficients, saveDataPath * "coefficients_kernel=$(Kernel)_epsilon=$(epsilon)_solver=$(solver)_lambda=$(lambda).png")
        savefig(plotConcentration, saveDataPath * "concentration_kernel=$(Kernel)_epsilon=$(epsilon)_solver=$(solver)_lambda=$(lambda).png")
        savefig(plotConcentrationPositive, saveDataPath * "concentrationPositive_kernel=$(Kernel)_epsilon=$(epsilon)_solver=$(solver)_lambda=$(lambda).png")
    end

    return c, B, coefB, A, u, SM
end

function mainStraightforwardKernelReco(experimentHDFPath::String, u::Vector{Float64})
    ExperimentData = readKernelBasedRecoExperiment(experimentHDFPath)

    reconstructionMatrix = ExperimentData.reconstructionMatrix
    basisFunctions = ExperimentData.basisFunctions

    # coefB = cg(reconstructionMatrix, u)
    coefB = cg(reconstructionMatrix' * reconstructionMatrix, reconstructionMatrix' * u)
    # coefB = reconstruction(transpose(reconstructionMatrix) * reconstructionMatrix, transpose(reconstructionMatrix) * u)

    concentration = basisFunctions' * coefB

    return concentration, basisFunctions, coefB, reconstructionMatrix
end

"""
# `computeBasisFunctions` Function

## Description
The `computeBasisFunctions` function computes basis functions for a given system matrix `SM`. It leverages a kernel function to evaluate interactions between grid points and uses this information to transform the system matrix.

## Arguments
- `SM::Array{T, 2}`: A 2D array representing the system matrix. Each column corresponds to different frequency components.
- `grid::Array{T, 1}`: A 1D array containing the grid points' coordinates (assumed to be in a common space).
- `delta::T`: A scalar representing the spacing between grid points.
- `::Type{KernelType}`: The kernel function to use, which must be a subtype of `AbstractKernel`.
- `epsilon::T`: A parameter for the kernel function, often controlling the spread or shape of the kernel.

## Returns
- `g::Array{Float64, 2}`: A 2D array where each column corresponds to the computed basis functions for each frequency component in `SM`.
"""
function computeBasisFunctions(SM, grid, delta, ::Type{KernelType}, epsilon) where {KernelType<:AbstractKernel}
    K = [kernel(r1, r2, epsilon, KernelType) for r1 in grid, r2 in grid]
    return delta .* transpose(SM) * transpose(K)
end

function computeBasisFunctions(SM, K, delta)
    return delta .* transpose(SM) * transpose(K)
end
