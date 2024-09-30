include("../../mpisimulation/MPIsimTools/src/MPIsimTools.jl")
include("../src/MPIKernelReco.jl")
using Plots
pyplot()

saveDataPath = "/Users/maxlewerenz/Documents/Uni/Master/Masterarbeit/tmp/thesisGraphics/interpolatedKernelReco/modeledBoxPhantom/"

u, SM, smP, phant = MPIsimTools.measuredSystemMatrixAndPhantom(noiseLevel=0.01)
u = u - real.(u)
u = vcat(real.(u), imag.(u))
SM = SM - real.(SM)
SM = hcat(real.(SM), imag.(SM))

phant = phant[:, 1, 1]
patch = 1
numPatches = 1

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
    L2ErrorPositiveConcentration = sqrt(sum((phant - max.(c, 0)) .^ 2))
    write(file, "Kleinster Eigenwert: " * "$(minEigval) \n")
    write(file, "Größter Eigenwert: " * "$(maxEigval) \n")
    write(file, "Konditionszahl: " * "$(condA) \n")
    write(file, "L2 Error compared with positive concentration / sqrt(voxelVolume): " * "$(L2ErrorPositiveConcentration) \n")
    close(file)
end
