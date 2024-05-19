"""
    initialize.jl file for loading relevant variables in the Julia REPL
"""

using MPIsimTools
using OpenMPIData

# Configuration
filenameSM = joinpath(OpenMPIData.basedir(), "data", "calibrations", "2.mdf")
filenameMeas = joinpath(OpenMPIData.basedir(), "data", "measurements", "shapePhantom", "2.mdf")
numPatches = 19
patch = 10
u, measP, mfP, tPaC, tP = MPIsimTools.readParams(filenameMeas, "Measurement", numPatches=numPatches)
measuredSM, smP = MPIsimTools.readParams(filenameSM, "SM", numPatches=numPatches)

#process the data for reconstruction
u, goodFreqIndex = MPIsimTools.processMeasurement(u, measP, tP, mfP)
measuredSM = MPIsimTools.processSM(measuredSM, smP, tP, goodFreqIndex)

#Dimensions of SMs: numGridPoints x tP.numFreq x 3 x periodsPerPatch (1)
u = reshape(u[:, 1:3, patch, :], length(goodFreqIndex) * 3)
SM = convert(Array{Complex{Float32},2}, reshape(measuredSM[:, :, 1:3, :],
                                                    smP.numVoxels[1] * smP.numVoxels[2] * smP.numVoxels[3], length(goodFreqIndex) * 3))
println("Done")

interpolation_grid = vec([[(2*i - (smP.FOV[1,1] + smP.FOV[1,2]))/((smP.FOV[1,2] - smP.FOV[1,1])),
                           (2*j - (smP.FOV[2,1] + smP.FOV[2,2]))/((smP.FOV[2,2] - smP.FOV[2,1])),
                           (2*k - (smP.FOV[3,1] + smP.FOV[3,2]))/((smP.FOV[3,2] - smP.FOV[3,1]))]
                          for i in smP.rX, j in smP.rY, k in smP.rZ])

x1 = [(2 * i - (smP.FOV[1, 1] + smP.FOV[1, 2])) / ((smP.FOV[1, 2] - smP.FOV[1, 1])) for i in smP.rX]
x2 = [(2 * i - (smP.FOV[2, 1] + smP.FOV[2, 2])) / ((smP.FOV[2, 2] - smP.FOV[2, 1])) for i in smP.rY]
x3 = [(2 * i - (smP.FOV[3, 1] + smP.FOV[3, 2])) / ((smP.FOV[3, 2] - smP.FOV[3, 1])) for i in smP.rZ]
