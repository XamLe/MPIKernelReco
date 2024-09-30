"""
    File containing functions for reading and writing data to a hdf5 file
"""

using HDF5
using Dates

"""
    writeInterpolationCoefficients(B::Array, ::Type{InterpolationKernel}, epsilon::Float64, interpolation_grid)

Writes the interpolation coefficients and associated metadata to an HDF5 file. The file is named with the current date and stored in the `../data` directory.

# Arguments
- `B::Array`: The interpolation coefficients to be written.
- `InterpolationKernel::Type{<:AbstractKernel}`: The type of interpolation kernel used.
- `epsilon::Float64`: The shape parameter for the interpolation kernel.
- `interpolation_grid`: The grid points used for interpolation (currently not written to file).
"""
function writeInterpolationCoefficients(B, ::Type{InterpolationKernel}, epsilon, interpolation_grid) where {InterpolationKernel<:AbstractKernel}
    filename = joinpath(pwd(), "../data", string(Dates.today()) * ".h5")
    h5file = h5open(filename, "w")
    write(h5file, "InterpolationCoefficients", B)
    write(h5file, "/Kernel/InterpolationKernel", string(InterpolationKernel))
    write(h5file, "/Kernel/ShapeParameter", epsilon)
    # write(h5file, "InterpolationGrid", interpolation_grid)

    timestamp = Dates.now()
    write(h5file, "timestamp", string(timestamp))

    close(h5file)
end

function writeKernelBasedRecoExperiment(filename::String,
                                        systemMatrixPath::String,
                                        systemMatrixGrid::Matrix{Float64},
                                        kernelType::String,
                                        shapeParameter::Float64,
                                        kernelMatrixSystemGrid::Matrix{Float64},
                                        convKernel,
                                        convConvKernel,
                                        systemMatrixLinearlyIndependent::Matrix{Float32},
                                        indicesOfLinearlyIndependentRows,
                                        basisFunctions::Matrix{Float64},
                                        reconstructionMatrix::Matrix{Float64})
    h5file = h5open(filename, "w")
    write(h5file, "/Properties/SystemMatrixPath", systemMatrixPath)
    write(h5file, "/Properties/SystemMatrixGrid", systemMatrixGrid)

    write(h5file, "/Kernel/KernelType", kernelType)
    write(h5file, "/Kernel/ShapeParameter", shapeParameter)
    write(h5file, "/Kernel/KernelMatrixSystemGrid", kernelMatrixSystemGrid)
    write(h5file, "/Kernel/ConvKernel", convKernel)
    write(h5file, "/Kernel/ConvConvKernel", convConvKernel)

    write(h5file, "/SystemMatrix/SystemMatrixLinearlyIndependent", systemMatrixLinearlyIndependent)
    write(h5file, "/SystemMatrix/IndicesOfLinearlyIndependentRows", indicesOfLinearlyIndependentRows)

    write(h5file, "BasisFunctions", basisFunctions)

    write(h5file, "ReconstructionMatrix", reconstructionMatrix)
    close(h5file)
end

function readKernelBasedRecoExperiment(filename)
    h5file = h5open(filename, "r")

    systemMatrixPath = read(h5file, "/Properties/SystemMatrixPath")
    systemMatrixGrid = read(h5file, "/Properties/SystemMatrixGrid")
    systemMatrixGrid = [collect(col) for col in eachcol(systemMatrixGrid)]

    kernelType = read(h5file, "/Kernel/KernelType")
    shapeParameter = read(h5file, "/Kernel/ShapeParameter")
    kernelMatrixSystemGrid = read(h5file, "/Kernel/KernelMatrixSystemGrid")
    convKernel = read(h5file, "/Kernel/ConvKernel")
    convConvKernel = read(h5file, "/Kernel/ConvConvKernel")

    systemMatrixLinearlyIndepent = read(h5file, "/SystemMatrix/SystemMatrixLinearlyIndependent")
    indicesOfLinearlyIndependentRows = read(h5file, "/SystemMatrix/IndicesOfLinearlyIndependentRows")

    basisFunctions = read(h5file, "BasisFunctions")

    reconstructionMatrix = read(h5file, "ReconstructionMatrix")

    close(h5file)

    return ExperimentData(systemMatrixPath,
        systemMatrixGrid,
        kernelType,
        shapeParameter,
        kernelMatrixSystemGrid,
        convKernel,
        convConvKernel,
        systemMatrixLinearlyIndepent,
        indicesOfLinearlyIndependentRows,
        basisFunctions,
        reconstructionMatrix)
end
