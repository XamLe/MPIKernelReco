module MPIKernelReco

include("Kernels.jl")
include("KernelInterpolation.jl")
include("KernelReconstruction.jl")
include("ReconstructionMatrix.jl")
include("ConcentrationFunctions.jl")
include("Visualization.jl")
include("readWriteHDF5.jl")

export kernel, ExponentialKernel, AbstractKernel, MultiquadricKernel, InverseMultiquadricKernel, Wendland0Kernel
export getKernelInterpolationCoefficients, kernelInterpolant, choleskyDecomposeInterpolationMatrix
export mainKernelReco
export computeConvolvedConvolutionalKernel
export computeConvolutionalKernel
export heatmapPlotSystemFunction, surfacePlotSystemFuction, surfacePlotInterpolatedSystemFunction, heatmapPlotInterpolatedSystemFunction, compareInterpolation
export writeInterpolationCoefficients
export concentration

end # module MPIKernelReco
