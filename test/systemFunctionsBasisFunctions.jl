using OpenMPIData
using LinearAlgebra
using Plots
include("../src/MPIKernelReco.jl")
pyplot()

patch = 10
numPatches = 19
filenameSM = joinpath(OpenMPIData.basedir(), "data", "calibrations", "2.mdf")
filenameMeas = joinpath(OpenMPIData.basedir(), "data", "measurements", "shapePhantom", "2.mdf")
saveDataPath = "/Users/maxlewerenz/Documents/Uni/Master/Masterarbeit/tmp/systemFunctionsBasisFunctions/"

epsilon = 100000.0

c, B, coefB, A, uLinearlyIndependent, SMLinearlyIndependent = MPIKernelReco.mainStraightforwardKernelReco(;epsilon = epsilon, solver = "CGNormal", lambda = 0.0)

B /= voxelVolume
for i in 1:3
    heatmapSystem = plot(layout = (1,1))
    heatmap!(heatmapSystem, reshape(SMLinearlyIndependent[:,i], 19,19,19)[:,:,10])
    savefig(heatmapSystem, saveDataPath * "heatmapSystem$i")
    surfaceSystem = plot(layout = (1,1), legend=:none)
    surface!(surfaceSystem, reshape(SMLinearlyIndependent[:,i], 19,19,19)[:,:,10], legend=:none)
    savefig(surfaceSystem, saveDataPath * "surfaceSystem$i")
    heatmapBasis = plot(layout = (1,1))
    heatmap!(heatmapBasis, reshape(B[i,:], 19,19,19)[:,:,10])
    savefig(heatmapBasis, saveDataPath * "heatmapBasis$i")
    surfaceBasis = plot(layout = (1,1), legend=:none)
    surface!(surfaceBasis, reshape(B[i,:], 19,19,19)[:,:,10], legend=:none)
    savefig(surfaceSystem, saveDataPath * "surfaceBasis$i")
end
