function initializeProblem(
    filenameSM=joinpath(OpenMPIData.basedir(), "data", "calibrations", "2.mdf"),
    filenameMeas=joinpath(OpenMPIData.basedir(), "data", "measurements", "shapePhantom", "2.mdf"),
    patch=10,
    numPatches=19;
    kernel=TensorWendland0Kernel(),
    epsilon=100.0,
    evaluationGrid::Union{Nothing,Vector{<:AbstractVector}}=nothing,
)
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

    voxelVolume = prod(smP.voxelSize)
    SM = hcat(real.(SM), imag.(SM))
    SM /= voxelVolume
    u = vcat(real.(u), imag.(u))

    grid = [[x, y, z] for x in smP.rX, y in smP.rY, z in smP.rZ]
    grid = reshape(grid, length(smP.rX) * length(smP.rY) * length(smP.rZ))

    if evaluationGrid === nothing
        evaluationGrid = grid
    end

    problemData = ProblemData(SM, grid, voxelVolume, smP.voxelSize, evaluationGrid, epsilon, kernel, u, smP.FOV, smP.rX, smP.rY, smP.rZ, true, false)

    return problemData
end

function initializePhantomProblem(kernel=GaussianKernel(),
                                  epsilon=20000.0)
    u, SM, smP, phant = MPIsimTools.measuredSystemMatrixAndPhantom(noiseLevel=0.01)
    # u = u - real.(u)
    # u = vcat(real.(u), imag.(u))
    u = imag.(u)
    u = u[2:length(u)-1]
    # SM = SM - real.(SM)
    #SM = hcat(real.(SM), imag.(SM))
    SM = imag.(SM)
    SM = SM[:,2:size(SM,2)-1]
    # SM /= voxelVolume
    phant = phant[:, 1, 1]
    voxelVolume = prod(smP.voxelSize)
    grid = [[x, y, z] for x in smP.rX, y in smP.rY, z in smP.rZ]
    grid = reshape(grid, length(smP.rX) * length(smP.rY) * length(smP.rZ))
    return ProblemData(SM, grid, voxelVolume, smP.voxelSize, grid, epsilon, kernel, u, smP.FOV, smP.rX, smP.rY, smP.rZ, true, false)
end
