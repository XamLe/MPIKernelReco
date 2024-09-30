using MPIsimTools
using OpenMPIData
using IterativeSolvers
using Plots
using Random

"""
    mainInterpolatedKernelReco(patch=10, numPatches=19, filenameSM=joinpath(OpenMPIData.basedir(), "data", "calibrations", "2.mdf"),
                               filenameMeas=joinpath(OpenMPIData.basedir(), "data", "measurements", "shapePhantom", "2.mdf"),
                               ::Type{InterpolationKernel}=GaussianKernel, epsilon_interpol=60.0,
                               ::Type{ReconstructionKernel}=GaussianKernel, epsilon_reco=60.0, lambda::Float64=0.0,
                               solver::String="CGNormal") where {ReconstructionKernel<:AbstractKernel,
                               InterpolationKernel<:AbstractKernel}

Reads the system matrix and measurement data, processes them, and performs a kernel-based interpolation reconstruction. The function uses predefined filenames for the system matrix and measurement data, which are read, reshaped, and then passed to the `mainInterpolatedKernelReco` function for processing and reconstruction.

# Arguments
- `patch::Int=10`: The specific patch index of the measurement to reconstruct.
- `numPatches::Int=19`: The number of patches in the measurement data to process.
- `filenameSM::String`: The path to the system matrix file (default points to calibration data).
- `filenameMeas::String`: The path to the measurement data file (default points to shape phantom data).
- `::Type{InterpolationKernel}`: The type of kernel used for interpolation (default is `GaussianKernel`).
- `epsilon_interpol::Float64`: The shape parameter for the interpolation kernel (default is `60.0`).
- `::Type{ReconstructionKernel}`: The type of kernel used for reconstruction (default is `GaussianKernel`).
- `epsilon_reco::Float64`: The shape parameter for the reconstruction kernel (default is `60.0`).
- `lambda::Float64=0.0`: Regularization parameter for solving the reconstruction problem (default is `0.0`).
- `solver::String="CGNormal"`: Solver type for the linear system (default is `"CGNormal"`).

# Returns
- The output of the `mainInterpolatedKernelReco` function, which includes:
    - `c::Vector`: The reconstructed values (concentration map).
    - `interpolation_coefficients::Matrix`: The coefficients from kernel interpolation.
    - `A::Matrix`: The assembled reconstruction matrix.
    - `SMInterpol::Matrix`: The interpolated system matrix.
    - `system_grid::Vector`: The grid used for system matrix evaluation.

# Notes
1. **Data Loading**: The function reads the system matrix and measurement data from `.mdf` files. The paths to these files can be customized.
2. **Data Processing**: The function processes the system matrix and measurements using `MPIsimTools`, reshapes them, and extracts the relevant frequency components.
3. **Reconstruction**: Once the data is reshaped and preprocessed, the function calls `mainInterpolatedKernelReco` to perform kernel interpolation and reconstruction.
4. **Customizable Kernels**: You can specify different types of kernels for interpolation and reconstruction via the `InterpolationKernel` and `ReconstructionKernel` arguments, as long as they are subtypes of `AbstractKernel`.
"""
function mainInterpolatedKernelReco(patch=10,
    numPatches=19,
    filenameSM=joinpath(OpenMPIData.basedir(), "data", "calibrations", "2.mdf"),
    filenameMeas=joinpath(OpenMPIData.basedir(), "data", "measurements", "shapePhantom", "2.mdf"),
    ::Type{InterpolationKernel}=GaussianKernel,
    epsilon_interpol=100000.0,
    ::Type{ReconstructionKernel} = GaussianKernel,
    epsilon_reco=20000.0,
                                    lambda::Float64=0.0,
                                    solver::String="CGNormal") where {ReconstructionKernel<:AbstractKernel,
    InterpolationKernel<:AbstractKernel}

    """
    1. Read system matrix and measurements
    """
    #load the data and parameters
    u, measP, mfP, tPaC, tP = readParams(filenameMeas, "Measurement", numPatches=numPatches)
    measuredSM, smP = readParams(filenameSM, "SM", numPatches=numPatches)

    #process the data for reconstruction
    u, goodFreqIndex = MPIsimTools.processMeasurement(u, measP, tP, mfP)
    measuredSM = MPIsimTools.processSM(measuredSM, smP, tP, goodFreqIndex)

    u = reshape(u[:, 1:3, patch, :], length(goodFreqIndex) * 3)
    SM = convert(Array{Complex{Float32},2}, reshape(measuredSM[:, :, 1:3, :],
        smP.numVoxels[1] * smP.numVoxels[2] * smP.numVoxels[3], length(goodFreqIndex) * 3))

    return mainInterpolatedKernelReco(SM, u, smP, InterpolationKernel, epsilon_interpol, ReconstructionKernel, epsilon_reco, lambda, solver, patch, false, "./tmp/", nothing)
end

"""
    mainInterpolatedKernelReco(SM::Matrix, u::Vector, smP, ::Type{InterpolationKernel}, epsilon_interpol::Float64,
                               ::Type{ReconstructionKernel}, epsilon_reco::Float64, lambda::Float64;
                               solver::String="CGNormal", patch::Int=10, saveFigure=false,
                               saveDataPath="./tmp/interpolatedKernelReco/", interpolation_coefficients=nothing)

Performs an interpolated kernel reconstruction based on the input system matrix `SM`, observed data `u`, and the specified interpolation and reconstruction kernels. The process involves:
1. Defining variables and reshaping the system grid.
2. Interpolating the system functions using the specified `InterpolationKernel`.
3. Assembling the reconstruction matrix using `ReconstructionKernel`.
4. Solving the reconstruction problem using the specified solver.
5. Evaluating the reconstructed data.
6. Visualizing the reconstruction results.

# Arguments
- `SM::Matrix`: The system matrix representing the measurement process.
- `u::Vector`: The vector of observed data corresponding to the system matrix.
- `smP`: A structure containing information about the system grid and voxel sizes (e.g., `smP.rX`, `smP.rY`, `smP.rZ` for grid coordinates).
- `::Type{InterpolationKernel}`: The type of kernel used for interpolation, which must be a subtype of `AbstractKernel`.
- `epsilon_interpol::Float64`: Shape parameter for the interpolation kernel.
- `::Type{ReconstructionKernel}`: The type of kernel used for reconstruction, which must be a subtype of `AbstractKernel`.
- `epsilon_reco::Float64`: Shape parameter for the reconstruction kernel.
- `lambda::Float64`: Regularization parameter for solving the reconstruction problem.
- `solver::String="CGNormal"`: The solver to use for the linear system (default is `"CGNormal"`). Can be changed depending on the solver used in `solveLinearSystem`.
- `patch::Int=10`: Patch size for visualization (default is `10`).
- `saveFigure::Bool=false`: Whether to save the reconstruction figures (default is `false`).
- `saveDataPath::String="./tmp/interpolatedKernelReco/"`: Directory where figures should be saved (default is `"./tmp/interpolatedKernelReco/"`).
- `interpolation_coefficients=nothing`: Precomputed interpolation coefficients. If provided, interpolation is skipped.

# Returns
- `c::Vector`: The reconstructed values (concentration map) based on the computed coefficients.
- `interpolation_coefficients::Matrix`: The coefficients from kernel interpolation, which can be reused in future interpolations.
- `A::Matrix`: The assembled reconstruction matrix.
- `SMInterpol::Matrix`: The interpolated system matrix.
- `system_grid::Vector`: The grid used for the system matrix evaluation.

# Notes
1. **Interpolation Phase**: If `interpolation_coefficients` is provided, the interpolation step is skipped, and the precomputed coefficients are used. Otherwise, interpolation is performed by computing the interpolation matrix and solving for coefficients.
2. **Reconstruction Matrix Assembly**: The reconstruction matrix `A` is constructed using the reconstruction kernel on an internal grid, and the system matrix is adjusted to ensure linear independence.
3. **Solver**: The default solver is `"CGNormal"`, but this can be adjusted as needed. The regularization parameter `lambda` is applied during the solving process.
4. **Visualization**: After reconstruction, results can be visualized and optionally saved as PNG images.
"""
function mainInterpolatedKernelReco(SM::Matrix,
    u::Vector,
    smP,
    ::Type{InterpolationKernel},
    epsilon_interpol::Float64,
    ::Type{ReconstructionKernel},
    epsilon_reco::Float64,
    lambda::Float64,
    solver::String="CGNormal",
    patch::Int=10,
    saveFigure=false,
    saveDataPath="./tmp/interpolatedKernelReco/",
    interpolation_coefficients=nothing) where {InterpolationKernel<:AbstractKernel,
    ReconstructionKernel<:AbstractKernel}

    """
    1. Define Variables
    """
    println("1. Define Variables")

    voxelVolume = prod(smP.voxelSize) # * 100 * 100 * 100
    SM = hcat(real.(SM), imag.(SM))
    SM /= voxelVolume
    u = vcat(real.(u), imag.(u))


    # The centers of the systemMatrix evaluation points are saved in the arrays smP.rX, smP.rY, smP.rZ
    x1_system = smP.rX # * 100
    x2_system = smP.rY # * 100
    x3_system = smP.rZ # * 100
    system_grid = [[i, j, k] for i in x1_system, j in x2_system, k in x3_system]
    system_grid = reshape(system_grid, length(x1_system) * length(x2_system) * length(x3_system))

    random = true
    if(random)
        n = length(system_grid)
        half_n = div(n,2)
        random_indices = randperm(n)
        selected_indices = random_indices[1:half_n]
        system_grid = system_grid[selected_indices]
        SM = SM[selected_indices, :]
    end

    """
    2. Interpolate system functions
    """
    println("2. Interpolate the system functions (this may take a while)")
    if (isnothing(interpolation_coefficients))
        interpolation_matrix = choleskyDecomposeInterpolationMatrix(system_grid, epsilon_interpol, InterpolationKernel)

        number_of_interpolation_operators = size(SM)[2]
        interpolation_coefficients = zeros(size(SM))
        for interpolation_operator in 1:number_of_interpolation_operators # for 2 * frequency components (real and imaginary part separately)
            SM_values = SM[:, interpolation_operator]
            # interpolation_coefficients[:,interpolation_operator] = getKernelInterpolationCoefficients(kernel_interpolation_matrix, SM_values, system_grid, epsilon_interpol, GaussianKernel)
            interpolation_coefficients[:, interpolation_operator] = getKernelInterpolationCoefficients(SM_values, interpolation_matrix)

            if floor.(interpolation_operator % (number_of_interpolation_operators / 10)) == 0
                println(interpolation_operator / number_of_interpolation_operators * 100, "% of frequency components processed.")
            end
        end
    else
        println("System functions do not need to be interpolated as coefficients were already provided.")
    end

    # interpolation_coefficients, u = findLinearlyIndependent(interpolation_coefficients, u, 1e-5)

    """
    3. Assemble reconstruction matrix
    """
    println("3. Assemble the reconstruction matrix")
    length_internal = [25, 25, 25]
    x1_internal= collect(range(smP.FOV[1, 1], smP.FOV[1, 2], length=length_internal[1])) # * 100
    x2_internal= collect(range(smP.FOV[2, 1], smP.FOV[2, 2], length=length_internal[2])) # * 100
    x3_internal= collect(range(smP.FOV[3, 1], smP.FOV[3, 2], length=length_internal[3])) # * 100
    # x3_internal = smP.rZ
    internal_grid = [[i, j, k] for i in x1_internal, j in x2_internal, k in x3_internal]
    internal_grid = reshape(internal_grid, prod(length_internal))
    voxelVolumeInternal = prod((smP.FOV[:, 2] - smP.FOV[:,1]) ./ length_internal) # * 100 * 100 * 100

    KInterpol = computeKernelMatrix(internal_grid, system_grid, epsilon_interpol, InterpolationKernel)
    KReco = computeKernelMatrix(internal_grid, internal_grid, epsilon_reco, ReconstructionKernel)
    SMInterpol = KInterpol * interpolation_coefficients
    SMInterpol, u = findLinearlyIndependent(SMInterpol, u, 1e-5)
    # KConv = voxelVolume * KInterpol * KReco
    # KConvConv = voxelVolume * transpose(KInterpol) * transpose(KConv)
    # A = interpolation_coefficients' * KConvConv * interpolation_coefficients
    A = voxelVolumeInternal^2 * transpose(SMInterpol) * KReco * SMInterpol
    A = (transpose(A) + A)/2

    """
    4. Perform the reconstruction
    """
    println("4. Perform the reconstruction")

    beta = solveLinearSystem(A, u, solver; lambda=lambda)

    """
    5. Evaluate the reconstruction
    """
    println("5. Evaluate the reconstruction")
    length_eval = [19, 19, 19]
    x1_eval = collect(range(smP.FOV[1, 1], smP.FOV[1, 2], length=length_eval[1])) # * 100
    x2_eval = collect(range(smP.FOV[2, 1], smP.FOV[2, 2], length=length_eval[2])) # * 100
    x3_eval = collect(range(smP.FOV[3, 1], smP.FOV[3, 2], length=length_eval[3])) # * 100
    # x3_eval = smP.rZ
    eval_grid = [[i, j, k] for i in x1_eval, j in x2_eval, k in x3_eval]
    eval_grid = reshape(eval_grid, prod(length_eval))
    # KInterpolEval = computeKernelMatrix(internal_grid, system_grid, epsilon_interpol, InterpolationKernel)
    KRecoEval = computeKernelMatrix(internal_grid, eval_grid, epsilon_reco, ReconstructionKernel)
    # KConvEval = voxelVolumeInternal * transpose(KRecoEval) * KInterpolEval
    c = transpose(voxelVolumeInternal * transpose(SMInterpol) * KRecoEval) * beta
    # c = transpose(transpose(SMInterpol) * KRecoEval) * beta

    """
    6. Visualization
    """
    println("6. Visualization")
    patch = 10
    p, plotCoefficients, plotConcentration, plotConcentrationPositive = plotCoefficientsAndConcentration(beta, c, length(x1_eval), length(x2_eval), length(x3_eval), patch)
    if (saveFigure)
        println("Speichere die Plots")
        savefig(plotCoefficients, saveDataPath * "coefficients_recoKernel=$(ReconstructionKernel)_epsilon=$(epsilon_reco)_solver=$(solver)_lambda=$(lambda).png")
        savefig(plotConcentration, saveDataPath * "concentration_recoKernel=$(ReconstructionKernel)_epsilon=$(epsilon_reco)_solver=$(solver)_lambda=$(lambda).png")
        savefig(plotConcentrationPositive, saveDataPath * "concentrationPositive_recoKernel=$(ReconstructionKernel)_epsilon=$(epsilon_reco)_solver=$(solver)_lambda=$(lambda).png")
    end
    # c *= 100
    return c, interpolation_coefficients, A, SMInterpol, system_grid
end
