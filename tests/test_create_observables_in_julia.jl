using LinearAlgebra
# Import package to time matrix multiplication
using BenchmarkTools

# Define an random "flux_matrix" that has shape (5885, 59)
flux_matrix = rand(Float64, 5885, 59)

# define an empty "detector_matrix" that has shape (601, 5885)
detector_matrix = rand(Float64, 601, 5885)

# define an empty flux matrix that has shape (120, 59)
flux = rand(Float64, 59, 120)

# define efficiency matrix with shape (601, 120)
efficiency_matrix = rand(Float64, 601, 120)

# do the matrix multiplication eight times and time the whole thing
function create_observables(flux)
    for i in 1:6
        result = detector_matrix * (flux_matrix * flux)
        result = result .* efficiency_matrix
    end
end

@btime create_observables(flux)


