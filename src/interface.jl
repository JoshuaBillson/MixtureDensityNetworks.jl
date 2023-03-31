"Struct for storing the hyperparameters of the MDN model."
mutable struct MDN
    mixtures::Int
    layers::Vector{Int}
    η::Float64
    epochs::Int
    batchsize::Int
    fitresult
end

"""
    MDN(; mixtures=5, layers=[128], η=1e-3, epochs=10, batchsize=16)

Defines an MDN model with the given hyperparameters.
"""
function MDN(; mixtures=5, layers=[128], η=1e-3, epochs=10, batchsize=16)
    return MDN(mixtures, layers, η, epochs, batchsize, nothing)
end

function fit!(model::MDN, X::Matrix{Float32}, Y::Matrix{Float32})
	# Create Model
	m = isnothing(model.fitresult) ? MixtureModel(size(X, 1), model.layers, model.mixtures) : model.fitresult

    # Define Optimizer
    opt = Flux.setup(Flux.Adam(model.η), m)

    # Prepare Training Data
    data = Flux.DataLoader((X, Y); batchsize=model.batchsize, shuffle=true)

	# Fit Model
    for epoch in 1:model.epochs
        Flux.train!(m, data, opt) do m, x, y
            likelihood_loss(m(x)..., y)
        end
    end

    # Save Fitted Model
    model.fitresult = m
end

"""
    fit!(model, X, Y)

Fit the model to the data given by X and Y.

X is expected to be a dxn matrix where d is the dimension of the data and n is the number of samples.

Y is expected to be a 1xn matrix where n is the number of samples.
"""
function fit!(model::MDN, X::Matrix{<:Number}, Y::Matrix{<:Number})
    fit!(model, Float32.(X), Float32.(Y))
end

function predict(model::MDN, X::Matrix{Float32})
    @assert !isnothing(model.fitresult) "Error: Must call fit!(model, X, Y) before predicting!"
    μ, σ, pi = model.fitresult(X)
	dists = Distributions.MixtureModel[]
	for i in eachindex(μ[1,:])
		d = Distributions.MixtureModel(Distributions.Normal.(μ[:,i], σ[:,i]), pi[:,i])
		push!(dists, d)
	end
	return dists
end

"""
    predict(model, X)

Make a prediction with the fitted model given the features specified by X.

X is expected to be a dxn matrix where d is the dimension of the data and n is the number of samples.

Returns a vector of Distributions.MixtureModel.
"""
function predict(model::MDN, X::Matrix{<:Number})
    predict(model, Float32.(X))
end