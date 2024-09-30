# These methods were copied from the MPIsimTools package, for reference see the readme.org file.
using Random

"""
  calculateTraceOfNormalMatrix(...)

COPIED from IBI
"""
function calculateTraceOfNormalMatrix(S)
  energy = zeros(Float64, size(S, 2))
  for m = 1:size(S, 2)
    energy[m] = norm(S[:, m])
  end
  return norm(energy)^2
end


"""
  dot_with_matrix_row(...)

COPIED from IBI
"""
function dot_with_matrix_row(A, x, k)
  tmp = 0.0
  for n = 1:size(A, 1)
    tmp += A[n, k] * x[n]
  end
  tmp
end



"""
COPIED from IBI.
The regularized kaczmarz algorithm solves the Thikonov regularized least squares Problem
argminₓ(‖Ax-b‖² + λ‖b‖²).
# Arguments
* `A::Array{Complex{Float32},2}`: System matrix A
* `b::Vector{Complex{Float64}}`: Measurement vector b
* `iterations::Int`: Number of iterations of the iterative solver
* `lambd::Float64`: The regularization parameter, relative to the matrix trace
* `shuff::Bool`: Enables random shuffeling of rows during iterations in the kaczmarz algorithm
* `enforceReal::Bool`: Enable projection of solution on real plane during iteration
* `enforcePositive::Bool`: Enable projection of solution onto positive halfplane during iteration
"""
function kaczmarzReg(A, b, iterations, lambd, shuff, enforceReal, enforcePositive)
  M = size(A, 2)
  N = size(A, 1)

  x = zeros(Complex{Float64}, N)
  residual = zeros(Complex{Float64}, M)

  energy = zeros(Float64, M)
  for m = 1:M
    energy[m] = norm(A[:, m])
  end

  rowIndexCycle = collect(1:M)

  if shuff
    shuffle(rowIndexCycle)
  end

  lambdIter = lambd

  for l = 1:iterations
    for m = 1:M
      k = rowIndexCycle[m]
      if energy[k] > 0
        tmp = dot_with_matrix_row(A, x, k)

        beta = (b[k] - tmp - sqrt(lambdIter) * residual[k]) / (energy[k]^2 + lambd)

        for n = 1:size(A, 1)
          x[n] += beta * conj(A[n, k])
        end

        residual[k] = residual[k] + beta * sqrt(lambdIter)
      end
    end

    if enforceReal && eltype(x) <: Complex
      x = complex.(real.(x), 0)
    end
    if enforcePositive
      x[real(x).<0] .= 0
    end
  end

  return x
end

"""
  reconstruction(...)

COPIED from IBI.
"""
function reconstruction(SM, u;
  iterations=1, lambda=0.1, SNRThresh=1.8,
  minFreq=30e3, maxFreq=1.25e6, recChannels=1:3,
  frames=1,#1:acqNumFGFrames(MPIFile(filenameMeas)),
  periods=1)#1:acqNumPeriodsPerFrame(MPIFile(filenameMeas)),
  #bgCorrection=true)
  S = SM
  lambda_ = calculateTraceOfNormalMatrix(S) * lambda / size(S, 1)
  # reconstruct using kaczmarz algorithm
  c = kaczmarzReg(S, u, iterations, lambda_, false, true, true)
  return c #reshape(c, calibSize(fCalib)..., :)
end
