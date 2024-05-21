using Plots

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
function heatmapPlotSystemFunction!(p, values::Vector{Float32}, patch=10::Int64)
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
function surfacePlotSystemFuction!(p, values::Vector{Float32}, patch=10::Int64)
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
function surfacePlotInterpolatedSystemFunction!(p, interpolation_grid, coefficients, ::Type{InterpolationKernel}, epsilon, patch) where {InterpolationKernel<:AbstractKernel}
    x_range, y_range = range(-1, 1, length=19), range(-1, 1, length=19)
    z = range(-1, 1, length=19)[patch]
    f(x,y) = kernelInterpolant([y,x,z], interpolation_grid, coefficients, epsilon, InterpolationKernel)
    Plots.plot!(p, x_range, y_range, f, st=:surface)
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
function surfacePlotInterpolatedSystemFunction(interpolation_grid, coefficients, ::Type{InterpolationKernel}, epsilon, patch) where {InterpolationKernel<:AbstractKernel}
    p = Plots.plot(layout=(1,1))
    surfacePlotInterpolatedSystemFunction!(p, interpolation_grid, coefficients, InterpolationKernel, epsilon, patch)
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
function heatmapPlotInterpolatedSystemFunction!(p, interpolation_grid, coefficients, ::Type{InterpolationKernel}, epsilon, patch) where {InterpolationKernel<:AbstractKernel}
    x_range, y_range = range(-1, 1, length=19), range(-1, 1, length=19)
    z = range(-1, 1, length=19)[patch]
    values = [kernelInterpolant([x, y, z], interpolation_grid, coefficients, epsilon, InterpolationKernel) for x in x_range, y in y_range]
    values = reshape(values, length(x_range), length(y_range))

    p = heatmap!(p, values)
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
function heatmapPlotInterpolatedSystemFunction(interpolation_grid, coefficients, ::Type{InterpolationKernel}, epsilon, patch) where {InterpolationKernel<:AbstractKernel}
    p = Plots.plot(layout = (1,1))
    heatmapPlotInterpolatedSystemFunction!(p, interpolation_grid, coefficients, InterpolationKernel, epsilon, patch)
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
