using LinearAlgebra
using IterativeSolvers
using OpenMPIData
using MPIsimTools

filenameSM = joinpath(OpenMPIData.basedir(), "data", "calibrations", "2.mdf")
filenameMeas = joinpath(OpenMPIData.basedir(), "data", "measurements", "shapePhantom", "2.mdf")
numPatches = 19
patch = 10

u, measP, mfP, tPaC, tP = readParams(filenameMeas, "Measurement", numPatches=numPatches)
measuredSM, smP = readParams(filenameSM, "SM", numPatches=numPatches)

#process the data for reconstruction
u, goodFreqIndex = MPIsimTools.processMeasurement(u, measP, tP, mfP)
measuredSM = MPIsimTools.processSM(measuredSM, smP, tP, goodFreqIndex)

u = reshape(u[:, 1:3, patch, :], length(goodFreqIndex) * 3)
SM = convert(Array{Complex{Float32},2}, reshape(measuredSM[:, :, 1:3, :],
    smP.numVoxels[1] * smP.numVoxels[2] * smP.numVoxels[3], length(goodFreqIndex) * 3))

voxelVolume = prod(smP.voxelSize) # * 100 * 100 * 100
# voxelVolume = 2/19 * 2/19 * 1/19

# hcat real and imaginary values resulting in a matrix of dimensions num_gridpoints x (num_freq_components * 2)
SM = hcat(real.(SM), imag.(SM))
SM /= voxelVolume
u = vcat(real.(u), imag.(u))

# The centers of the systemMatrix evaluation points are saved in the arrays smP.rX, smP.rY, smP.rZ
x1 = smP.rX # * 100
x2 = smP.rY # * 100
x3 = smP.rZ # * 100
interpolation_grid = [[i, j, k] for i in x1, j in x2, k in x3]
interpolation_grid = reshape(interpolation_grid, length(x1) * length(x2) * length(x3))

InterpolationKernel = MPIKernelReco.GaussianKernel
# epsilons = [10.0, 30.0, 50.0, 70.0]
epsilons = [100000.0]
SMInterpol = zeros(size(SM))

for epsilon in epsilons
    interpolation_matrix = MPIKernelReco.choleskyDecomposeInterpolationMatrix(interpolation_grid[1:2:end], epsilon, InterpolationKernel)
    # kernel_interpolation_matrix = MPIKernelReco.assembleKernelInterpolationMatrix(interpolation_grid[1:2:end], epsilon, InterpolationKernel)

    number_of_interpolation_operators = size(SM)[2]
    interpolation_coefficients = zeros(size(SM[1:2:size(SM,1), :]))
    for interpolation_operator in 1:number_of_interpolation_operators # for 2 * frequency components (real and imaginary part separately)
        SM_values = SM[1:2:size(SM,1), interpolation_operator]
        # interpolation_coefficients[:,interpolation_operator] = getKernelInterpolationCoefficients(kernel_interpolation_matrix, SM_values, interpolation_grid, epsilon_interpol, GaussianKernel)
        interpolation_coefficients[:, interpolation_operator] = MPIKernelReco.getKernelInterpolationCoefficients(SM_values, interpolation_matrix)
        if floor.(interpolation_operator % (number_of_interpolation_operators / 10)) == 0
            println(interpolation_operator / number_of_interpolation_operators * 100, "% of frequency components processed.")
        end
    end

    KInterpol = MPIKernelReco.computeKernelMatrix(interpolation_grid, interpolation_grid[1:2:end], epsilon, InterpolationKernel)

    global SMInterpol =  KInterpol * interpolation_coefficients

    SMError = norm(SM - SMInterpol, 2)
    println("Error for epsilon $(epsilon): ", SMError)
    MPIKernelReco.surfacePlotInterpolatedSystemFunction(interpolation_grid[1:2:end], interpolation_coefficients[:,1], InterpolationKernel, epsilon, 10, smP.FOV)
end
