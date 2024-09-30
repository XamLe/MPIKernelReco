struct ExperimentData
    systemMatrixPath::String
    systemMatrixGrid::Vector{Vector{Float64}}
    kernelType::String
    shapeParameter::Float64
    kernelMatrixSystemGrid::Matrix{Float64}
    convKernel
    convConvKernel
    systemMatrixLinearlyIndependent::Matrix{Float32}
    indicesOfLinearlyIndependentRows
    basisFunctions::Matrix{Float64}
    reconstructionMatrix::Matrix{Float64}
end
