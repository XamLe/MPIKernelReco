include("../../mpisimulation/MPIsimTools/src/MPIsimTools.jl")
include("../src/MPIKernelReco.jl")
using OpenMPIData
using Plots
using LinearAlgebra
pyplot()

patch = 10
numPatches = 19
filenameSM = joinpath(OpenMPIData.basedir(), "data", "calibrations", "2.mdf")
filenameMeas = joinpath(OpenMPIData.basedir(), "data", "measurements", "shapePhantom", "2.mdf")
saveDataPath = "/Users/maxlewerenz/Documents/Uni/Master/Masterarbeit/tmp/thesisGraphics/interpolatedKernelReco/measuredShapePhantom/"

"""
1. Read system matrix and measurements
"""
#load the data and parameters
u, measP, mfP, tPaC, tP = MPIsimTools.readParams(filenameMeas, "Measurement", numPatches=numPatches)
measuredSM, smP = MPIsimTools.readParams(filenameSM, "SM", numPatches=numPatches)

#process the data for reconstruction
u, goodFreqIndex = MPIsimTools.processMeasurement(u, measP, tP, mfP)
measuredSM = MPIsimTools.processSM(measuredSM, smP, tP, goodFreqIndex)

u = reshape(u[:, 1:3, patch, :], length(goodFreqIndex) * 3)
SM = convert(Array{Complex{Float32},2}, reshape(measuredSM[:, :, 1:3, :],
        smP.numVoxels[1] * smP.numVoxels[2] * smP.numVoxels[3], length(goodFreqIndex) * 3))

ReconstructionKernel = MPIKernelReco.TensorWendland0Kernel
epsilon_reco = 100.0
open(saveDataPath * "experimentProperties_RecoKernel=$(ReconstructionKernel)_epsilon=$(epsilon_reco).txt", "w") do file
        global c, interpolation_coefficients, A, SMInterpol, system_grid = MPIKernelReco.mainInterpolatedKernelReco(SM,
                u,
                smP,
                MPIKernelReco.GaussianKernel,
                100000.0,
                ReconstructionKernel,
                epsilon_reco,
                0.0,
                "CGNormal",
                patch,
                true,
                saveDataPath)
        eigvalsA = eigvals(A)
        minEigval = eigvalsA[1]
        maxEigval = eigvalsA[end]
        condA = cond(A)
        write(file, "Kleinster Eigenwert: " * "$(minEigval) \n")
        write(file, "Größter Eigenwert: " * "$(maxEigval) \n")
        write(file, "Konditionszahl: " * "$(condA) \n")
        close(file)
end

SM = hcat(real.(SM), imag.(SM))
voxelVolume = prod(smP.voxelSize)
system_grid_z0 = filter(v -> v == 0, system_grid)
x_system_z0, y_system_z0= [p[1] for p in system_grid_z0], [p[2] for p in system_grid_z0]

for k in [1, 2, 3, 4]
        pSurfaceSM = plot(layout=(1, 1), legend=:none)
        pHeatmapSM = plot(layout=(1, 1))
        pSurfaceInterpolatedSM = plot(layout=(1, 1), legend=:none)
        pHeatmapInterpolatedSM = plot(layout=(1, 1))
        surface!(pSurfaceSM, smP.rX, smP.rY, reshape(SM[:, k] / voxelVolume, smP.numVoxels[1], smP.numVoxels[2], smP.numVoxels[3])[:, :, 10], legend=:none)
        heatmap!(pHeatmapSM, smP.rX, smP.rY, reshape(SM[:, k] / voxelVolume, 19, 19, 19)[:, :, 10])
        # scatter!(pHeatmapSM, x_coords, y_coords; color=:black, marker=:x, legend=false)
        MPIKernelReco.surfacePlotInterpolatedSystemFunction!(pSurfaceInterpolatedSM, system_grid, interpolation_coefficients[:, k], MPIKernelReco.GaussianKernel, 100000.0, patch, smP.FOV)
        MPIKernelReco.heatmapPlotInterpolatedSystemFunction!(pHeatmapInterpolatedSM, system_grid, interpolation_coefficients[:, k], MPIKernelReco.GaussianKernel, 100000.0, patch, smP.FOV)
        # scatter!(pHeatmapInterpolatedSM, x_coords, y_coords; color=:black, marker=:x, legend=false)
        savefig(pSurfaceSM, saveDataPath * "pSurfaceSM$k.png")
        savefig(pHeatmapSM, saveDataPath * "pHeatmapSM$k.png")
        savefig(pSurfaceInterpolatedSM, saveDataPath * "pSurfaceInterpolatedSM$k")
        savefig(pHeatmapInterpolatedSM, saveDataPath * "pHeatmapInterpolatedSM$k")
end
