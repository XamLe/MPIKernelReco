# These methods were copied from the MPIsimTools package, for reference see the readme.org file.
using LinearAlgebra

function ballPhantomConcentration(r::Vector, radius::Float64)
    if(norm(r) <= radius)
        return 1
    else
        return 0
    end
end

function getBallPhantom(regularGrid::Array{Vector{Float64}}, radius::Float64)
    return ballPhantomConcentration.(regularGrid, radius)
end
