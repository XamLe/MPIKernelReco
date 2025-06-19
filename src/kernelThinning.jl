using LinearAlgebra
"""
`kernelThinning` Function

#Description:
The `kernelThinning` function performs the kernel-based reconstruction in an iterative fashion with greedy algorithms.

#Arguments:
- `A` Gram Matrix A_{K, Λ}
- `u` Vector of Measurements
- `maxIter` max. number of Iteerations
- `tol` tolerance for numerical dependence
- `X` X subset mathbb{R}^d evaluation set
"""
function kernelThinning(maxIter, tol, problemData::ProblemData{K}) where {K<:AbstractKernel}
    weightNormalization!(problemData)
    kernelMatrix = computeKernelMatrix(problemData.inherentGrid, problemData.inherentGrid, problemData.epsilon, typeof(problemData.reconstructionKernel))

    A = problemData.voxelVolume .^ 2 .* transpose(problemData.systemMatrix) * kernelMatrix * problemData.systemMatrix
    A = (transpose(A) + A) ./ 2 # Gram Matrix
    A = A + 1e-9 * I(size(A,1))
    m = size(A, 1) # Number of Operators
    # Init Indices
    ind = [1]

    # Init Newton basis
    V = zeros(m, maxIter) # Vandermonde Matrix
    V[:, 1] = (1 / sqrt(A[1, 1])) * A[:, 1]
    N = zeros(size(problemData.evaluationGrid)[1], maxIter) # Evaluation of Newton basis in evaluation set X
    rieszRepresenterEvaluations = computeRieszRepresenterEvaluations(problemData.evaluationGrid, problemData)
    N[:, 1] = (1 / sqrt(A[1, 1])) * rieszRepresenterEvaluations[:, 1]


    # Init Distance / suared power function
    dist = diag(A) - V[:, 1] .^ 2

    # Init residual and coefficients
    c = zeros(maxIter)
    c[1] = problemData.u[1] / sqrt(A[1, 1])
    res = problemData.u - c[1] * V[:, 1]

    # Set start/stop for greedy selection loop
    start = 1
    stop = maxIter - 1

    # Init non-selected indices
    notIn = [i for i in 1:m if i ∉ ind]
    residualNorms = []

    # c = hcat(c, zeros(maxIter))

    for k ∈ start:stop
        # Greedy selction (provisorisch)
        i = argmax(dist[notIn])

        index = notIn[i]

        # Determine power function value at chosen functional
        p = sqrt(dist[index])

        if p < tol
            println("Power function too small!")
            c = c[:, 1:k]
            V = V[:, 1:k]
            break
        end

        # Update Newton basis evaluation at Γ
        V[notIn, k+1] = (1 / p) * (A[notIn, index] - V[notIn, 1:k] * V[index, 1:k])

        # Update Newton basis evaluation at X (grid)
        N[:, k+1] = (1 / p) * (rieszRepresenterEvaluations[:, index] - N[:, 1:k] * V[index, 1:k])

        # Update power / distance function
        dist[notIn] = dist[notIn] - V[notIn, k+1].^2

        # Determin coefficient of interpolant and update residual
        c[k+1] = res[index] / p
        res[notIn] = res[notIn] - c[k+1] * V[notIn, k + 1]

        push!(ind, index)
        filter!(e -> e ≠ index, notIn)
        push!(residualNorms, norm(res))
    end # loop

    reconstructionResult = ReconstructionResult(N*c, N, c, A, res, -1.0)
    # return c, V, N, dist, res, ind
    return reconstructionResult, residualNorms, ind
end
