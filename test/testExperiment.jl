using OpenMPIData
using LinearAlgebra
using Plots
pyplot()

patch = 10
numPatches = 19
filenameSM = joinpath(OpenMPIData.basedir(), "data", "calibrations", "2.mdf")
filenameMeas = joinpath(OpenMPIData.basedir(), "data", "measurements", "shapePhantom", "2.mdf")
saveDataPath = "/Users/maxlewerenz/Documents/Uni/Master/Masterarbeit/tmp/thesisGraphics/"

# Gaussian Kernel & Epsilon = 20000.0
kernel = MPIKernelReco.TensorWendland0Kernel
epsilon = 100.0
# solvers = ["CG", "regularizedCG", "CGNormal", "regularizedKaczmarz", "backslash", "regularizedBackslash", "regularizedCholesky"]
solvers = ["CGNormal"]
lambda = 1e8

c, B, coefB, A, u, SM = 0, 0, 0, 0, 0, 0
for solver in solvers
        global c, B, coefB, A, u, SM = MPIKernelReco.mainStraightforwardKernelReco(patch,
                numPatches,
                filenameSM,
                filenameMeas,
                kernel,
                false;
                epsilon=epsilon,
                lambda=lambda,
                solver=solver,
                saveFigure=true,
                saveDataPath=saveDataPath)
end
eigvalsA = eigvals(A)
minEigval = eigvalsA[1]
maxEigval = eigvalsA[end]
condA = cond(A)
open(saveDataPath * "experimentProperties_kernel=$(kernel)_epsilon=$(epsilon).txt", "w") do file
    write(file, "Kleinster Eigenwert: " * "$(minEigval) \n")
    write(file, "Größter Eigenwert: " * "$(maxEigval) \n")
    write(file, "Konditionszahl: " * "$(condA) \n")
    close(file)
end
