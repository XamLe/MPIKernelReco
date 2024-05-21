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
