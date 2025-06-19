problemData = MPIKernelReco.initializeProblem(kernel=MPIKernelReco.PolynomialKernel(), epsilon=10000.0)

residualNorms = []
interpolantNorms = []

λ = 1e-9
for ε in [1e8, 1e9, 1e10]
    problemData.epsilon = ε
    reconstructionResult = MPIKernelReco.mainStraightforwardKernelReco(problemData; lambda = λ, solver="regularizedCG")
    push!(residualNorms, norm(reconstructionResult.residual))
    push!(interpolantNorms, reconstructionResult.interpolantNorm)
end
