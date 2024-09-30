include("../../mpisimulation/MPIsimTools/src/MPIsimTools.jl")
include("../src/MPIKernelReco.jl")
using Plots
pyplot()

saveDataPath = "/Users/maxlewerenz/Documents/Uni/Master/Masterarbeit/tmp/thesisGraphics/straightforwardBoxPhantom/"

u, SM, smP, phant = MPIsimTools.measuredSystemMatrixAndPhantom(noiseLevel = 0.01)
u = u - real.(u)
u = vcat(real.(u), imag.(u))
SM = SM - real.(SM)
SM = hcat(real.(SM), imag.(SM))

phant = phant[:, 1, 1]

patch = 1
numPatches = 1
Kernels = [MPIKernelReco.GaussianKernel, MPIKernelReco.InverseMultiquadricKernel, MPIKernelReco.TensorWendland0Kernel]
epsilons = [20000.0, 100000.0, 100.0, 20000.0]
solvers = ["CG", "regularizedCG", "CGNormal", "regularizedKaczmarz", "backslash", "regularizedBackslash", "regularizedCholesky"]
lambda = 1e-2

c, B, coefB, A, uLinearlyIndependent, SMLinearlyIndependent = 0, 0, 0, 0, 0, 0
for i in 1:size(Kernels, 1)
    open(saveDataPath * "experimentProperties_kernel=$(Kernels[i])_epsilon=$(epsilons[i]).txt", "w") do file
        for solver in solvers
            global c, B, coefB, A, uLinearlyIndependent, SMLinearlyIndependent = MPIKernelReco.mainStraightforwardKernelReco(SM,
                u,
                smP,
                Kernels[i],
                epsilons[i],
                lambda,
                false;
                solver=solver,
                patch=patch,
                saveFigure=true,
                saveDataPath=saveDataPath)
            L2Error = sqrt(sum((phant - c) .^ 2))
            L2ErrorPositiveConcentration = sqrt(sum((phant - max.(c, 0)) .^ 2))
            write(file, "$(solver) " * "L2 Error / sqrt(voxelVolume): " * "$(L2Error) \n")
            write(file, "$(solver) " * "L2 Error compared with positive concentration / sqrt(voxelVolume): " * "$(L2ErrorPositiveConcentration) \n")
        end
    eigvalsA = eigvals(A)
    minEigval = eigvalsA[1]
    maxEigval = eigvalsA[end]
    condA = cond(A)
    # open(saveDataPath * "experimentProperties_kernel=$(kernel)_epsilon=$(epsilon).txt", "w") do file
    write(file, "Voxel Volume: " * "$(prod(smP.voxelSize)) \n")
    write(file, "Kleinster Eigenwert: " * "$(minEigval) \n")
    write(file, "Größter Eigenwert: " * "$(maxEigval) \n")
    write(file, "Konditionszahl: " * "$(condA) \n")
    close(file)
    end
end
