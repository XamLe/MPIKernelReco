using Plots

"""
    Visualization.jl

file containing functions for visualizing concentrations or system functions.
"""

function heatmapPlotSystemFunction!(p, values::Vector{Float32}, patch=10::Int64)
    values = reshape(values, (19,19,19))

    heatmap!(p, values[:,:,patch], title = "System Function, for patch $patch")
end

function heatmapPlotSystemFunction!(p, values::Vector{ComplexF32}, patch=10::Int64)
    values = reshape(values, (19,19,19))

    p1 = Plots.plot(layout=(1,1))
    p2 = Plots.plot(layout=(1,1))

    p1 = heatmap!(p1, real.(values[:,:,patch]), aspect_ratio=:equal, title = "Real Part, patch $patch")

    p2 = heatmap!(p2, imag.(values[:,:,patch]), aspect_ratio=:equal, title = "Imaginary Part, patch $patch")

    Plots.plot!(p, p1, p2, layout = (1,2))
end

function heatmapPlotSystemFunction(values, patch=10::Int64)
    p = Plots.plot(layout=(1,1))
    heatmapPlotSystemFunction!(p, values, patch)
end

function surfacePlotSystemFuction!(p, values::Vector{Float32}, patch=10::Int64)
	values = reshape(values, (19,19,19))
    surface!(p, values[:,:,patch])
end

function surfacePlotSystemFuction!(p, values::Vector{ComplexF32}, patch=10::Int64)
	values = reshape(values, (19,19,19))

    p1 = Plots.plot(layout=(1,1))
    p2 = Plots.plot(layout=(1,1))

    p1 = surface!(p1, real.(values[:,:,patch]), title = "Real Part, patch $patch")
    p2 = surface!(p2, imag.(values[:,:,patch]), title = "Imaginary Part, patch $patch")

    Plots.plot!(p, p1,p2, layout = (1,2))
end

function surfacePlotSystemFunction(values, patch=10::Int64)
    p = Plots.plot(layout=(1,1))
    surfacePlotSystemFuction!(p, values, patch)
end

function surfacePlotInterpolatedSystemFunction!(p, interpolation_grid, coefficients, ::Type{InterpolationKernel}, epsilon, patch) where {InterpolationKernel<:AbstractKernel}
    x_range, y_range = range(-1, 1, length=19), range(-1, 1, length=19)
    z = range(-1, 1, length=19)[patch]
    f(x,y) = kernelInterpolant([y,x,z], interpolation_grid, coefficients, epsilon, InterpolationKernel)
    Plots.plot!(p, x_range, y_range, f, st=:surface)
end

function surfacePlotInterpolatedSystemFunction(interpolation_grid, coefficients, ::Type{InterpolationKernel}, epsilon, patch) where {InterpolationKernel<:AbstractKernel}
    p = Plots.plot(layout=(1,1))
    surfacePlotInterpolatedSystemFunction!(p, interpolation_grid, coefficients, InterpolationKernel, epsilon, patch)
end

function heatmapPlotInterpolatedSystemFunction!(p, interpolation_grid, coefficients, ::Type{InterpolationKernel}, epsilon, patch) where {InterpolationKernel<:AbstractKernel}
    x_range, y_range = range(-1, 1, length=19), range(-1, 1, length=19)
    z = range(-1, 1, length=19)[patch]
    values = [kernelInterpolant([x, y, z], interpolation_grid, coefficients, epsilon, InterpolationKernel) for x in x_range, y in y_range]
    values = reshape(values, length(x_range), length(y_range))

    p = heatmap!(p, values)
end

function heatmapPlotInterpolatedSystemFunction(interpolation_grid, coefficients, ::Type{InterpolationKernel}, epsilon, patch) where {InterpolationKernel<:AbstractKernel}
    p = Plots.plot(layout = (1,1))
    heatmapPlotInterpolatedSystemFunction!(p, interpolation_grid, coefficients, InterpolationKernel, epsilon, patch)
end

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
