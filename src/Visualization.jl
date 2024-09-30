using Plots
using LaTeXStrings
using Printf


"""
    Visualization.jl

file containing functions for visualizing concentrations or system functions.
"""

"""
    heatmapPlotSystemFunction!(p, values::Vector{Float32}, patch::Int64 = 10)

Generates a heatmap of the system function for the specified patch and adds it to the given plot.

# Arguments
- `p`: The plot object to which the heatmap will be added.
- `values::Vector{Float32}`: A vector of system function values that will be reshaped into a 3D array with dimensions (19, 19, 19).
- `patch::Int64`: The patch index for which the system function should be visualized (default: 10).
"""
function heatmapPlotSystemFunction!(p, values::Vector, patch=10::Int64)
    values = reshape(values, (19,19,19))

    heatmap!(p, values[:,:,patch], title = "System Function, for patch $patch")
end

"""
    heatmapPlotSystemFunction!(p, values::Vector{ComplexF32}, patch::Int64 = 10)

Generates heatmaps of the real and imaginary parts of the system function for the specified patch and adds these heatmaps to the given plot.

# Arguments
- `p`: The plot object to which the heatmaps will be added.
- `values::Vector{ComplexF32}`: A vector of complex system function values that will be reshaped into a 3D array with dimensions (19, 19, 19).
- `patch::Int64`: The patch index for which the system function should be visualized (default: 10).
"""
function heatmapPlotSystemFunction!(p, values::Vector{ComplexF32}, patch=10::Int64)
    values = reshape(values, (19,19,19))

    p1 = Plots.plot(layout=(1,1))
    p2 = Plots.plot(layout=(1,1))

    p1 = heatmap!(p1, real.(values[:,:,patch]), aspect_ratio=:equal, title = "Real Part, patch $patch")

    p2 = heatmap!(p2, imag.(values[:,:,patch]), aspect_ratio=:equal, title = "Imaginary Part, patch $patch")

    Plots.plot!(p, p1, p2, layout = (1,2))
end

"""
    heatmapPlotSystemFunction(values::Vector{ComplexF32}, patch::Int64 = 10) -> Plots.Plot

Generates a plot with heatmaps of the real and imaginary parts of the system function for the specified patch.

This function creates a new plot and then calls `heatmapPlotSystemFunction!` to add the heatmaps to the plot.

# Arguments
- `values::Vector{ComplexF32}`: A vector of complex system function values that will be reshaped into a 3D array with dimensions (19, 19, 19).
- `patch::Int64`: The patch index for which the system function should be visualized (default: 10).

# Returns
- `Plots.Plot`: The generated plot object with the heatmaps.
"""
function heatmapPlotSystemFunction(values, patch=10::Int64)
    p = Plots.plot(layout=(1,1))
    heatmapPlotSystemFunction!(p, values, patch)
end

"""
    surfacePlotSystemFunction!(p, values::Vector{Float32}, patch::Int64 = 10)

Generates a surface plot of the system function for the specified patch and adds it to the given plot.

# Arguments
- `p`: The plot object to which the surface plot will be added.
- `values::Vector{Float32}`: A vector of system function values that will be reshaped into a 3D array with dimensions (19, 19, 19).
- `patch::Int64`: The patch index for which the system function should be visualized (default: 10).
"""
function surfacePlotSystemFuction!(p, values::Vector, patch=10::Int64)
	values = reshape(values, (19,19,19))
    surface!(p, values[:,:,patch])
end

"""
    surfacePlotSystemFunction!(p, values::Vector{ComplexF32}, patch::Int64 = 10)

Generates surface plots of the real and imaginary parts of the system function for the specified patch and adds these plots to the given plot.

# Arguments
- `p`: The plot object to which the surface plots will be added.
- `values::Vector{ComplexF32}`: A vector of complex system function values that will be reshaped into a 3D array with dimensions (19, 19, 19).
- `patch::Int64`: The patch index for which the system function should be visualized (default: 10).
"""
function surfacePlotSystemFuction!(p, values::Vector{ComplexF32}, patch=10::Int64)
	values = reshape(values, (19,19,19))

    p1 = Plots.plot(layout=(1,1))
    p2 = Plots.plot(layout=(1,1))

    p1 = surface!(p1, real.(values[:,:,patch]), title = "Real Part, patch $patch")
    p2 = surface!(p2, imag.(values[:,:,patch]), title = "Imaginary Part, patch $patch")

    Plots.plot!(p, p1,p2, layout = (1,2))
end

"""
    surfacePlotSystemFunction(values::Vector{ComplexF32}, patch::Int64 = 10) -> Plots.Plot

Generates a plot with surface plots of the real and imaginary parts of the system function for the specified patch.

This function creates a new plot and then calls `surfacePlotSystemFunction!` to add the surface plots to the plot.

# Arguments
- `values::Vector{ComplexF32}`: A vector of complex system function values that will be reshaped into a 3D array with dimensions (19, 19, 19).
- `patch::Int64`: The patch index for which the system function should be visualized (default: 10).

# Returns
- `Plots.Plot`: The generated plot object with the surface plots.
"""
function surfacePlotSystemFunction(values, patch=10::Int64)
    p = Plots.plot(layout=(1,1))
    surfacePlotSystemFuction!(p, values, patch)
end

"""
    surfacePlotInterpolatedSystemFunction!(p, interpolation_grid::Array, coefficients::Array,
                                           ::Type{InterpolationKernel}, epsilon::Float64, patch::Int64)
                                           where {InterpolationKernel <: AbstractKernel}

Generates a surface plot of the interpolated system function for the specified patch and adds it to the given plot.

# Arguments
- `p`: The plot object to which the surface plot will be added.
- `interpolation_grid::Array`: The grid points used for interpolation.
- `coefficients::Array`: The coefficients required for interpolation.
- `InterpolationKernel::Type{<:AbstractKernel}`: The type of kernel used for interpolation.
- `epsilon::Float64`: The shape parameter for the interpolation kernel.
- `patch::Int64`: The patch index for which the system function should be visualized.
"""
function surfacePlotInterpolatedSystemFunction!(p, interpolation_grid, coefficients, ::Type{InterpolationKernel}, epsilon, patch, FOV) where {InterpolationKernel<:AbstractKernel}
    x_range, y_range = range(FOV[1,1], FOV[1,2], length=50), range(FOV[2,1], FOV[2,2], length=50)
    z = range(FOV[3,1], FOV[3,2], length=19)[patch]
    f(x,y) = kernelInterpolant([y,x,z], interpolation_grid, coefficients, epsilon, InterpolationKernel)
    Plots.plot!(p, x_range, y_range, f, st=[:surface], legend=:none)
end

"""
    surfacePlotInterpolatedSystemFunction(interpolation_grid::Array, coefficients::Array,
                                          ::Type{InterpolationKernel}, epsilon::Float64, patch::Int64)

Generates a plot with a surface plot of the interpolated system function for the specified patch.

This function creates a new plot and then calls `surfacePlotInterpolatedSystemFunction!` to add the surface plot to the plot.

# Arguments
- `interpolation_grid::Array`: The grid points used for interpolation.
- `coefficients::Array`: The coefficients required for interpolation.
- `InterpolationKernel::Type{<:AbstractKernel}`: The type of kernel used for interpolation.
- `epsilon::Float64`: The shape parameter for the interpolation kernel.
- `patch::Int64`: The patch index for which the system function should be visualized.
"""
function surfacePlotInterpolatedSystemFunction(interpolation_grid, coefficients, ::Type{InterpolationKernel}, epsilon, patch, FOV) where {InterpolationKernel<:AbstractKernel}
    p = Plots.plot(layout=(1,1))
    surfacePlotInterpolatedSystemFunction!(p, interpolation_grid, coefficients, InterpolationKernel, epsilon, patch, FOV)
    display(p)
    return p
end

"""
    heatmapPlotInterpolatedSystemFunction!(p, interpolation_grid::Array, coefficients::Array,
                                           ::Type{InterpolationKernel}, epsilon::Float64, patch::Int64)
                                           where {InterpolationKernel<:AbstractKernel}

Generates a heatmap of the interpolated system function for the specified patch and adds it to the given plot.

# Arguments
- `p`: The plot object to which the heatmap will be added.
- `interpolation_grid::Array`: The grid points used for interpolation.
- `coefficients::Array`: The coefficients required for interpolation.
- `InterpolationKernel::Type{<:AbstractKernel}`: The type of kernel used for interpolation.
- `epsilon::Float64`: The shape parameter for the interpolation kernel.
- `patch::Int64`: The patch index for which the system function should be visualized.
"""
function heatmapPlotInterpolatedSystemFunction!(p, interpolation_grid, coefficients, ::Type{InterpolationKernel}, epsilon, patch, FOV) where {InterpolationKernel<:AbstractKernel}
    x_range, y_range = range(FOV[1,1], FOV[1,2], length=50), range(FOV[2,1], FOV[2,2], length=50)
    z = range(FOV[3,1], FOV[3,2], length=19)[patch]
    values = [kernelInterpolant([x, y, z], interpolation_grid, coefficients, epsilon, InterpolationKernel) for x in x_range, y in y_range]
    values = reshape(values, length(x_range), length(y_range))

    p = heatmap!(p, x_range, y_range, values)
end

"""
    heatmapPlotInterpolatedSystemFunction(interpolation_grid::Array, coefficients::Array,
                                          ::Type{InterpolationKernel}, epsilon::Float64, patch::Int64)
                                          where {InterpolationKernel<:AbstractKernel}

Generates a plot with a heatmap of the interpolated system function for the specified patch.

This function creates a new plot and then calls `heatmapPlotInterpolatedSystemFunction!` to add the heatmap to the plot.

# Arguments
- `interpolation_grid::Array`: The grid points used for interpolation.
- `coefficients::Array`: The coefficients required for interpolation.
- `InterpolationKernel::Type{<:AbstractKernel}`: The type of kernel used for interpolation.
- `epsilon::Float64`: The shape parameter for the interpolation kernel.
- `patch::Int64`: The patch index for which the system function should be visualized.

# Returns
- `Plots.Plot`: The generated plot object with the heatmap.
"""
function heatmapPlotInterpolatedSystemFunction(interpolation_grid, coefficients, ::Type{InterpolationKernel}, epsilon, patch, FOV) where {InterpolationKernel<:AbstractKernel}
    p = Plots.plot(layout = (1,1))
    heatmapPlotInterpolatedSystemFunction!(p, interpolation_grid, coefficients, InterpolationKernel, epsilon, patch, FOV)
    return p
end

"""
    compareInterpolation(values::Vector{ComplexF32}, interpolation_grid::Array, coefficients::Array,
                         ::Type{InterpolationKernel}, epsilon::Float64, patch::Int64)
                         where {InterpolationKernel<:AbstractKernel}

Generates a comparison plot that includes heatmaps and surface plots of the actual system function and its interpolated version for the specified patch.

This function creates subplots to visualize both the actual and interpolated system function using heatmaps and surface plots.

# Arguments
- `values::Vector{ComplexF32}`: A vector of complex system function values that will be reshaped into a 3D array.
- `interpolation_grid::Array`: The grid points used for interpolation.
- `coefficients::Array`: The coefficients required for interpolation.
- `InterpolationKernel::Type{<:AbstractKernel}`: The type of kernel used for interpolation.
- `epsilon::Float64`: The shape parameter for the interpolation kernel.
- `patch::Int64`: The patch index for which the system function should be visualized.
"""
function compareInterpolation(values, interpolation_grid, coefficients, ::Type{InterpolationKernel}, epsilon, patch) where {InterpolationKernel<:AbstractKernel}
    p1 = plot(layout = (1,1), title="Heatmap, actual SM")
    p2 = plot(layout = (1,1), title="Surface, actual SM")
    p3 = plot(layout = (1,1), title="Heatmap, interpolated SM")
    p4 = plot(layout = (1,1), title="Surface, interpolated SM")

    p1 = heatmapPlotSystemFunction!(p1, values, patch)
    p2 = surfacePlotSystemFuction!(p2, values, patch)
    p3 = heatmapPlotInterpolatedSystemFunction!(p3, interpolation_grid, coefficients, InterpolationKernel, epsilon, patch)
    p4 = surfacePlotInterpolatedSystemFunction!(p4, interpolation_grid, coefficients, InterpolationKernel, epsilon, patch)

    p = plot(p1, p2, p3, p4, layout = (2,2))
end

function heatmapPlotConcentration!(p, concentration::Vector, x::Int, y::Int, z::Int, patch::Int; positive::Bool=false)
    maximumConcentration = findmax(concentration)[1]
    concentration = reshape(concentration, x, y, z)
    concentration = concentration[:,:, patch]

    if (positive)
        heatmap!(p, concentration, clims=(0, maximumConcentration), aspect_ratio=:equal, xlabel=L"x", ylabel=L"y", title="")
    else
        heatmap!(p, concentration, aspect_ratio=:equal, xlabel=L"x", ylabel=L"y", title="")
    end
end

function heatmapPlotConcentration(concentration::Vector, x::Int, y::Int, z::Int, patch::Int; positive::Bool=false)
    p = Plots.plot(layout=(1, 1))
    heatmapPlotConcentration!(p, concentration, x, y, z, patch; positive)
    display(p)
end

function plotCoefficientsAndConcentration(coefficients, concentration::Vector, x::Int, y::Int, z::Int, patch::Int)
    pyplot()
    normCoefficients = @sprintf("%.3e", norm(coefficients, 2))
    p1 = plot(layout=(1, 1))
    p3 = plot(layout=(1, 1))
    p4 = plot(layout=(1, 1))

    p1 = plot!(p1, coefficients, label=L"\alpha", xlabel=L"k")
    p3 = heatmapPlotConcentration!(p3, concentration, x, y, z, patch)
    p4 = heatmapPlotConcentration!(p4, concentration, x, y, z, patch; positive=true)

    p2 = plot(p3, p4, layout=(1, 2), plot_title="Reconstructed Particle concentration")

    p = plot(p1, p2, layout=(2, 1))
    # pCopy = deepcopy(p) # Workouround necessary
    display(p)
    return p, p1, p3, p4
end

