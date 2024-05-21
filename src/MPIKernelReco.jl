module MPIKernelReco

include("Kernels.jl")
include("KernelInterpolation.jl")
include("KernelReconstruction.jl")
include("ReconstructionMatrix.jl")
include("ConcentrationFunctions.jl")
include("Visualization.jl")
include("readWriteHDF5.jl")

export kernel, ExponentialKernel, AbstractKernel, MultiquadricKernel, InverseMultiquadricKernel, Wendland0Kernel
export getKernelInterpolationCoefficients, assembleKernelInterpolationMatrix, kernelInterpolant, choleskyDecomposeInterpolationMatrix
export mainKernelReco
export assembleReconstructionMatrix, computeConvolvedConvolutionalKernel
export computeConvolutionalKernel
export heatmapPlotSystemFunction, surfacePlotSystemFunction, surfacePlotInterpolatedSystemFunction, heatmapPlotInterpolatedSystemFunction, compareInterpolation
export writeInterpolationCoefficients
export concentration

end # module MPIKernelReco
