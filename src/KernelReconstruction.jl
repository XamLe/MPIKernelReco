using MPIsimTools
using OpenMPIData

"""
# KernelReconstruction

File containing functions for the kernel-based reconstruction.
"""

"""
    mainKernelReco(patch, numPatches, filenameMeas, filenameSM)

Triggers the kernel interpolation and starts the processes of
1. reading system matrix and measurements
2. interpolating the system functions
3. assembling the reconstruction matrix
4. performing the reconstruction
5. evaluating and visualizing the concentration

# Arguments
- `patch`: Number that specifies the patch that should be reconstructed
- `numPatches`: Total amount of patches
- `filenameSM`: Path to the system matrix `.mdf` file
- `filenameMeas`: Path to the measurement results `.mdf` file
"""

function mainKernelReco(patch=10,
    numPatches=19,
    filenameSM=joinpath(OpenMPIData.basedir(), "data", "calibrations", "2.mdf"),
    filenameMeas=joinpath(OpenMPIData.basedir(), "data", "measurements", "shapePhantom", "2.mdf"),
    epsilon=50000)


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

    # The centers of the systemMatrix evaluation points are saved in the arrays smP.rX, smP.rY, smP.rZ
    # Scale the grid to [-1,1]^3
    interpolation_grid = vec([[(2 * i - (smP.FOV[1, 1] + smP.FOV[1, 2])) / ((smP.FOV[1, 2] - smP.FOV[1, 1])),
                               (2 * j - (smP.FOV[2, 1] + smP.FOV[2, 2])) / ((smP.FOV[2, 2] - smP.FOV[2, 1])),
                               (2 * k - (smP.FOV[3, 1] + smP.FOV[3, 2])) / ((smP.FOV[3, 2] - smP.FOV[3, 1]))]
                              for i in smP.rX, j in smP.rY, k in smP.rZ])

    interpolation_matrix = choleskyDecomposeInterpolationMatrix(interpolation_grid, epsilon, ExponentialKernel)

    number_of_interpolation_operators = size(SM_real_imag)[2]
    coefficients = zeros(size(SM_real_imag))
    for interpolation_operator in 1:number_of_interpolation_operators # for 2 * frequency components (real and imaginary part separately)
        SM_values = SM_real_imag[:, interpolation_operator]
        # coefficients[:,interpolation_operator] = getKernelInterpolationCoefficients(SM_values, interpolation_grid, epsilon, ExponentialKernel)
        coefficients[:, interpolation_operator] = getKernelInterpolationCoefficients(SM_values, interpolation_matrix)
        println(interpolation_operator / number_of_interpolation_operators * 100, "% of frequency components processed.")
    end
end
