"""
    File containing functions for reading and writing data to a hdf5 file
"""

using HDF5
using Dates

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
