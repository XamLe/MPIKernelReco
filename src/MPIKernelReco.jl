module MPIKernelReco

include("Structures.jl")
include("Kernels.jl")
include("KernelInterpolation.jl")
include("KernelReconstruction.jl")
include("ReconstructionMatrix.jl")
include("ConcentrationFunctions.jl")
include("Visualization.jl")
include("readWriteHDF5.jl")
include("straightforwardKernelReco.jl")
include("phantoms.jl")
include("kaczmarzReconstruction.jl")
include("utils.jl")

export kernel, GaussianKernel, AbstractKernel, MultiquadricKernel, InverseMultiquadricKernel, Wendland0Kernel, computeKernelMatrix
export getKernelInterpolationCoefficients, assembleKernelInterpolationMatrix, kernelInterpolant, choleskyDecomposeInterpolationMatrix
export mainInterpolatedKernelReco
export assembleReconstructionMatrix, computeConvolvedConvolutionalKernel
export computeConvolutionalKernel
export heatmapPlotSystemFunction, surfacePlotSystemFunction, surfacePlotInterpolatedSystemFunction, heatmapPlotInterpolatedSystemFunction, compareInterpolation, heatmapPlotConcentration
export writeInterpolationCoefficients, writeKernelBasedRecoExperiment, readKernelBasedRecoExperiment
export concentration
export mainStraightforwardKernelReco, computeBasisFunctions, findLinearlyIndependent
export getBallPhantom
export kaczmarzReg, calculateTraceOfNormalMatrix, reconstruction
export scaleLinearSystemByRow, scaleLinearSystemByMaximum

# To start the kernel based reconstruction, use the function mainStraightforwardKernelReco
#
# To start the interpolated kernel based reconsruction, use the function mainInterpolatedKernelReco

end # module MPIKernelReco
