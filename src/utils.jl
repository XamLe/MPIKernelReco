function scaleLinearSystemByRow(A::Matrix{Float64}, b::Vector{Float64})
	# Ensure dimensions are compatible
    m, n = size(A)
    @assert length(b) == m "The length of b must match the number of rows in A"

    # Initialize scaled matrix and vector
    A_scaled = copy(A)
    b_scaled = copy(b)#

    # Scale each row of A and adjust b
    for i in 1:m
        # Compute the norm of the ith row of A
        row_norm = norm(A[i, :], Inf)

        # Scale the row if its norm is non-zero
        if row_norm > 0
            A_scaled[i, :] /= row_norm
            b_scaled[i] /= row_norm
        end
    end

    return A_scaled, b_scaled
end

function scaleLinearSystemByMaximum(A::Matrix{Float64}, b::Vector{Float64})
	# Ensure dimensions are compatible
    m, n = size(A)
    @assert length(b) == m "The length of b must match the number of rows in A"

    # Initialize scaled matrix and vector
    A_scaled = copy(A)
    b_scaled = copy(b)#

    maximum_norm = norm(A, Inf)

    A_scaled /= maximum_norm
    b_scaled /= maximum_norm

    return A_scaled, b_scaled
end

"""
    findLinearlyIndependent(SM::Matrix, u::Vector, tol::Float64=1e-5) -> (Matrix, Vector)

Identifies and extracts the linearly independent columns from the given matrix `SM` and adjusts the corresponding elements in the vector `u`.

# Arguments
- `SM::Matrix`: A matrix where each column represents a vector that might be linearly dependent on the others. The function aims to keep only those columns that are linearly independent.
- `u::Vector`: A vector whose elements correspond to the columns of `SM`. The function adjusts this vector by only keeping elements corresponding to the linearly independent columns of `SM`.
- `tol::Float64=1e-5`: A tolerance level for determining linear dependence. Columns of `SM` with corresponding values in the R factor of the QR decomposition below this tolerance will be considered linearly dependent. The default value is `1e-5`.

# Returns
- A tuple `(Matrix, Vector)`:
  - `Matrix`: A new matrix consisting of the linearly independent columns from the original matrix `SM`.
  - `Vector`: A new vector consisting of the elements from the original vector `u` that correspond to the linearly independent columns of `SM`.

# Method
1. Perform QR decomposition on the matrix `SM`, obtaining matrices `Q` and `R`.
2. Identify linearly dependent columns by checking the absolute values of the diagonal elements of `R`. Columns with diagonal values below the specified tolerance `tol` are considered dependent.
3. Generate a list of indices for linearly independent columns.
4. Extract the linearly independent columns from `SM` and the corresponding elements from `u`.
5. If linearly dependent columns were removed, recursively call the function to ensure complete reduction.
6. Return the reduced matrix and vector when no linearly dependent columns are present.
"""
function findLinearlyIndependent(SM::Matrix, u::Vector, tol::Float64=1e-5)
    Q, R = qr(SM)
    m = size(SM, 2)
    linearlyDependentIndices = findall(x -> x < tol, abs.(diag(R)))
    remainingIndices = [i for i in 1:m if !(i in linearlyDependentIndices)]
    SM = SM[:, remainingIndices]
    u = u[remainingIndices]
    if (linearlyDependentIndices == Int64[])
        return SM, u
    else
        findLinearlyIndependent(SM, u, tol)
    end
end


function solveLinearSystem(A::Matrix, u::Vector, solver::String; lambda::Float64=0.0)
    if solver == "CG"
        lambda = 0
        sol, log = cg(A, u; log=true)
        println(log)
    elseif solver == "regularizedCG"
        regularizedA = A + lambda .* I(size(A, 1))
        sol, log = cg(regularizedA, u; log=true)
        println(log)
    elseif solver == "CGNormal"
        lambda = 0
        sol, log = cg(transpose(A) * A, transpose(A) * u; log=true)
        println(log)
    elseif solver == "regularizedKaczmarz"
        # lambda = calculateTraceOfNormalMatrix(A) * 0.000000000001 / size(A, 1)
        println("Lambda for kaczmarz algorithm: $lambda")
        sol = real.(kaczmarzReg(A, u, 100, lambda, true, true, false))
    elseif solver == "backslash"
        lambda = 0
        sol = A \ u
    elseif solver == "regularizedBackslash"
        regularizedA = A + lambda .* I(size(A, 1))
        sol = regularizedA \ u
    elseif solver == "regularizedCholesky"
        regularizedA = A + lambda .* I(size(A, 1))
        C = cholesky(regularizedA)
        sol = C \ u
    else
        println("Solver unknown: ", solver)
        throw(error("Unknown Solver"))
    end

    return sol
end
