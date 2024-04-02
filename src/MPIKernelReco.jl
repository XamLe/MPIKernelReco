module MPIKernelReco

include("Kernels.jl")
include("KernelInterpolation.jl")
include("KernelReconstruction.jl")
include("ReconstructionMatrix.jl")

export kernel, ExponentialKernel, AbstractKernel, MultiquadricKernel, InverseMultiquadricKernel, Wendland0Kernel
export getKernelInterpolationCoefficients, kernelInterpolant, choleskyDecomposeInterpolationMatrix
export mainKernelReco
export computeConvolutedConvolutionalKernel

end # module MPIKernelReco
