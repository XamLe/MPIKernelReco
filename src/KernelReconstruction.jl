using MPIsimTools
using OpenMPIData
using IterativeSolvers
using Plots

"""
# KernelReconstruction

File containing functions for the kernel-based reconstruction.
"""

"""
    mainKernelReco(patch::Int, numPatches::Int, filenameSM::String, filenameMeas::String,
                   ::Type{ReconstructionKernel} = ExponentialKernel, epsilon_reco::Float64 = 50.0,
                   ::Type{InterpolationKernel} = ExponentialKernel, epsilon_interpol::Float64 = 50.0)

Triggers the kernel interpolation and starts the following processes:

1. Reading the system matrix and measurements.
2. Interpolating the system functions.
3. Assembling the reconstruction matrix.
4. Performing the reconstruction.
5. Evaluating and visualizing the concentration.

# Arguments
- `patch::Int`: Number that specifies the patch that should be reconstructed.
- `numPatches::Int`: Total amount of patches.
- `filenameSM::String`: Path to the system matrix `.mdf` file.
- `filenameMeas::String`: Path to the measurement results `.mdf` file.
- `ReconstructionKernel::Type{<:AbstractKernel}`: The type of kernel to use for reconstruction (default: `ExponentialKernel`).
- `epsilon_reco::Float64`: Parameter for the reconstruction kernel (default: 50.0).
- `InterpolationKernel::Type{<:AbstractKernel}`: The type of kernel to use for interpolation (default: `ExponentialKernel`).
- `epsilon_interpol::Float64`: Parameter for the interpolation kernel (default: 50.0).

# Returns
A tuple containing:
- `c`: The concentration array.
- `interpolation_coefficients`: The coefficients required for interpolation.
- `A`: The reconstruction matrix.
"""
function mainKernelReco(patch=10,
    numPatches=19,
    filenameSM=joinpath(OpenMPIData.basedir(), "data", "calibrations", "2.mdf"),
    filenameMeas=joinpath(OpenMPIData.basedir(), "data", "measurements", "shapePhantom", "2.mdf"),
    ::Type{ReconstructionKernel}=ExponentialKernel, epsilon_reco=50.0,
    ::Type{InterpolationKernel}=ExponentialKernel, epsilon_interpol=50.0) where {ReconstructionKernel<:AbstractKernel, InterpolationKernel<:AbstractKernel}

    """
    1. Read system matrix and measurements
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
    2. Interpolate system functions
    """
    # hcat real and imaginary values resulting in a matrix of dimensions num_gridpoints x (num_freq_components * 2)
    # TODO: handle the general case, by case distinction of SM in time or frequency space
    SM_real_imag = hcat(real.(SM), imag.(SM))
    u_real_imag = vcat(real.(u), imag.(u))

    # The centers of the systemMatrix evaluation points are saved in the arrays smP.rX, smP.rY, smP.rZ
    # Scale the grid to [-1,1]^3
    x1 = [(2 * i - (smP.FOV[1, 1] + smP.FOV[1, 2])) / ((smP.FOV[1, 2] - smP.FOV[1, 1])) for i in smP.rX]
    x2 = [(2 * i - (smP.FOV[2, 1] + smP.FOV[2, 2])) / ((smP.FOV[2, 2] - smP.FOV[2, 1])) for i in smP.rY]
    x3 = [(2 * i - (smP.FOV[3, 1] + smP.FOV[3, 2])) / ((smP.FOV[3, 2] - smP.FOV[3, 1])) for i in smP.rZ]
    interpolation_grid = [[x1[i], x2[j], x3[k]] for i in eachindex(smP.rX), j in eachindex(smP.rY), k in eachindex(smP.rZ)]
    interpolation_grid = reshape(interpolation_grid, length(smP.rX) * length(smP.rY) * length(smP.rZ))

    interpolation_matrix = choleskyDecomposeInterpolationMatrix(interpolation_grid, epsilon_interpol, InterpolationKernel)
    # kernel_interpolation_matrix = assembleKernelInterpolationMatrix(interpolation_grid, epsilon_interpol, InterpolationKernel)

    number_of_interpolation_operators = size(SM_real_imag)[2]
    interpolation_coefficients = zeros(size(SM_real_imag))
    for interpolation_operator in 1:number_of_interpolation_operators # for 2 * frequency components (real and imaginary part separately)
        SM_values = SM_real_imag[:, interpolation_operator]
        # interpolation_coefficients[:,interpolation_operator] = getKernelInterpolationCoefficients(kernel_interpolation_matrix, SM_values, interpolation_grid, epsilon_interpol, ExponentialKernel)
        interpolation_coefficients[:, interpolation_operator] = getKernelInterpolationCoefficients(SM_values, interpolation_matrix)

        if floor.(interpolation_operator % (number_of_interpolation_operators / 10)) == 0
            println(interpolation_operator / number_of_interpolation_operators * 100, "% of frequency components processed.")
        end
    end

    # 2.1 store data to a hdf5 file
    writeInterpolationCoefficients(interpolation_coefficients, InterpolationKernel, epsilon_interpol, interpolation_grid)

    """
    3. Assemble reconstruction matrix
    """
    A = assembleReconstructionMatrix(interpolation_coefficients, x1, x2, x3, ReconstructionKernel, epsilon_reco, InterpolationKernel, epsilon_interpol)
    beta = cg(A, u_real_imag)

    """
    4. Perform the reconstruction
    """
    c = concentration(x1, x2, x3, x1, x2, x3, interpolation_coefficients, beta, ReconstructionKernel, epsilon_reco, InterpolationKernel, epsilon_interpol)


    """
    5. Plot the concentration for the given patch
    """
    plot1 = plot(layout = (1,1))
    heatmap!(plot1,reshape(c, (smP.numVoxels[1], smP.numVoxels[2], smP.numVoxels[3]))[:,:,patch], aspect_ratio=:equal, title = "Heatmap of the concentration, patch = $patch")

    return c, interpolation_coefficients, A
end
