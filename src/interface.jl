"""
$(TYPEDEF)

The hyperparameters defining the classical MDN model.

# Parameters
$(TYPEDFIELDS)
"""
mutable struct MDN
    mixtures::Int
    layers::Vector{Int}
    η::Float64
    epochs::Int
    batchsize::Int
    fitresult
end

"""
    MDN(; mixtures=5, layers=[128], η=1e-3, epochs=1, batchsize=32)

Defines an MDN model with the given hyperparameters.

# Parameters
- `mixtures`: The number of gaussian mixtures to use in estimating the conditional distribution (default=5).
- `layers`: A vector indicating the number of nodes in each of the hidden layers (default=[128,]).
- `η`: The learning rate to use when training the model (default=1e-3).
- `epochs`: The number of epochs to train the model (default=1).
- `batchsize`: The batchsize to use during training (default=32).
"""
function MDN(; mixtures=5, layers=[128], η=1e-3, epochs=1, batchsize=32)
    return MDN(mixtures, layers, η, epochs, batchsize, nothing)
end

"""
$(TYPEDSIGNATURES)

Fit the model to the data given by X and Y.

# Parameters
- `model`: The MDN to be trained.
- `X`: A dxn matrix where d is the number of features and n is the number of samples.
- `Y`: A 1xn matrix where n is the number of samples.
"""
function fit!(model::MDN, X::Matrix{<:Number}, Y::Matrix{<:Number})
    fit!(model, Float32.(X), Float32.(Y))
end

function fit!(model::MDN, X::Matrix{Float32}, Y::Matrix{Float32})
	# Create Model
	m = isnothing(model.fitresult) ? MixtureDensityNetwork(size(X, 1), model.layers, model.mixtures) : model.fitresult

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
$(TYPEDSIGNATURES)

Predict the full conditional distribution P(Y|X).

# Parameters
- `model`: The MDN with which we want to generate a prediction.
- `X`: A dxn matrix where d is the number of features and n is the number of samples.

# Returns
Returns a vector of Distributions.MixtureModel objects representing the conditional distribution for each sample.
"""
function predict(model::MDN, X::Matrix{<:Number})
    predict(model, Float32.(X))
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
$(TYPEDSIGNATURES)

Predict the mean of the conditional distribution P(Y|X). 

# Parameters
- `model`: The MDN with which we want to generate a prediction.
- `X`: A dxn matrix where d is the number of features and n is the number of samples.

# Returns
Returns a vector of real numbers representing the mean of the conditional distribution P(Y|X) for each sample.
"""
function predict_mean(model::MDN, X::Matrix{<:Number})
    predict_mean(model, Float32.(X))
end

function predict_mean(model::MDN, X::Matrix{Float32})
    @assert !isnothing(model.fitresult) "Error: Must call fit!(model, X, Y) before predicting!"
    return predict(model, X) .|> mean
end

"""
$(TYPEDSIGNATURES)

Predict the mean of the Gaussian with the largest prior in the conditional distribution P(Y|X). 

# Parameters
- `model`: The MDN with which we want to generate a prediction.
- `X`: A dxn matrix where d is the number of features and n is the number of samples.

# Returns
Returns a vector of real numbers representing the mean of the gaussian with the largest prior for each sample.
"""
function predict_mode(model::MDN, X::Matrix{<:Number})
    predict_mode(model, Float32.(X))
end

function predict_mode(model::MDN, X::Matrix{Float32})
    @assert !isnothing(model.fitresult) "Error: Must call fit!(model, X, Y) before predicting!"
    
    # Run Forward Pass
    μ, σ, π = model.fitresult(X)

    # Find The Maximum Priors For Each Observation
    max_priors = mapslices(argmax, π, dims=1)[1,:]

    # Extract The Mode Of The Distribution Matching The Max Prior
    return [μ[max_prior,i] for (i, max_prior) in enumerate(max_priors)]
end