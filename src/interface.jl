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
end

"""
$(TYPEDEF)

A struct for associating a model's hyperparameters with its training state.

# Parameters
$(TYPEDFIELDS)
"""
mutable struct Machine
    hypers::MDN
    fitresult::Union{MixtureDensityNetwork,Nothing}
    report::NamedTuple{(:learning_curve, :best_epoch, :best_loss), Tuple{Vector{Float64}, Int, Float64}}
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
    return MDN(mixtures, layers, η, epochs, batchsize)
end

"""
    Machine(hypers::MDN)

Binds a collection of hyperparameters with a training state for fitting, evaluating, and predicting.

# Parameters
- `hypers`: The hyperparameters we want our model to conform to.
"""
function Machine(hypers::MDN)
    return Machine(hypers, nothing, (learning_curve=Float64[], best_epoch=0, best_loss=Inf))
end

"""
$(TYPEDSIGNATURES)

Fit the model to the data given by X and Y.

# Parameters
- `machine`: The machine to be trained.
- `X`: A dxn matrix where d is the number of features and n is the number of samples.
- `Y`: A 1xn matrix where n is the number of samples.
"""
function fit!(machine::Machine, X::Matrix{<:Real}, Y::Matrix{<:Real})
    fit!(machine, Float64.(X), Float64.(Y))
end

function fit!(machine::Machine, X::Matrix{Float64}, Y::Matrix{Float64})
    fitresult, report = _fit(machine.hypers, machine.fitresult, X, Y)
    machine.fitresult = fitresult
    machine.report = (
        learning_curve=vcat(machine.report.learning_curve, report.learning_curve), 
        best_epoch=report.best_epoch, 
        best_loss=report.best_loss )
end

"""
$(TYPEDSIGNATURES)

Predict the full conditional distribution P(Y|X).

# Parameters
- `machine`: The machine with which we want to generate a prediction.
- `X`: A dxn matrix where d is the number of features and n is the number of samples.

# Returns
Returns a vector of Distributions.MixtureModel objects representing the conditional distribution for each sample.
"""
function predict(machine::Machine, X::Matrix{<:Real})
    return predict(machine, Float64.(X))
end

function predict(machine::Machine, X::Matrix{Float64})
    @assert !isnothing(machine.fitresult) "Error: Must call fit!(machine, X, Y) before predicting!"
    return _predict(machine.fitresult, X)
end

"""
$(TYPEDSIGNATURES)

Predict the mean of the conditional distribution P(Y|X). 

# Parameters
- `machine`: The machine with which we want to generate a prediction.
- `X`: A dxn matrix where d is the number of features and n is the number of samples.

# Returns
Returns a vector of real numbers representing the mean of the conditional distribution P(Y|X) for each sample.
"""
function predict_mean(machine::Machine, X::Matrix{<:Real})
    return predict_mean(machine, Float64.(X))
end

function predict_mean(machine::Machine, X::Matrix{Float64})
    @assert !isnothing(machine.fitresult) "Error: Must call fit!(machine, X, Y) before predicting!"
    return _predict(machine.fitresult, X) .|> mean
end

"""
$(TYPEDSIGNATURES)

Predict the mean of the Gaussian with the largest prior in the conditional distribution P(Y|X). 

# Parameters
- `machine`: The machine with which we want to generate a prediction.
- `X`: A dxn matrix where d is the number of features and n is the number of samples.

# Returns
Returns a vector of real numbers representing the mean of the gaussian with the largest prior for each sample.
"""
function predict_mode(machine::Machine, X::Matrix{<:Real})
    return predict_mode(machine, Float64.(X))
end

function predict_mode(machine::Machine, X::Matrix{Float64})
    @assert !isnothing(machine.fitresult) "Error: Must call fit!(machine, X, Y) before predicting!"
    return _predict_mode(machine.fitresult, X)
end


#########################
#### PRIVATE METHODS ####
#########################


function _fit(model::MDN, fitresult::Union{Nothing, MixtureDensityNetwork}, X::Matrix{Float64}, Y::Matrix{Float64})
	# Create Model
    m = isnothing(fitresult) ? MixtureDensityNetwork(size(X, 1), model.layers, model.mixtures) : fitresult

    # Define Optimizer
    opt = Flux.Adam(model.η)

    # Get Parameters
    params = Flux.params(m)

    # Prepare Training Data
    data = Flux.DataLoader((X, Y); batchsize=model.batchsize, shuffle=true)

	# Iterate Over Epochs
    best_model = deepcopy(m)
    learning_curve = Float64[]
    @progress for epoch in 1:model.epochs

        # Iterate Over Data
        losses = Float64[]
        for (x, y) in data

            # Compute Loss and Gradient
            l, grad = Flux.withgradient(params) do 
                likelihood_loss(m(x)..., y)
            end

            # Update Parameters
            Flux.update!(opt, params, grad)

            # Save Loss
            push!(losses, l)
        end

        # Add Average Loss To Learning Curve
        push!(learning_curve, mean(losses))

        # Save Best Performing Model
        if length(learning_curve) == 1 || learning_curve[end] < minimum(learning_curve[1:end-1])
            best_model = deepcopy(m)
        end

    end

    # Return Results
    report = (learning_curve=learning_curve, best_epoch=argmin(learning_curve), best_loss=minimum(learning_curve))
    return best_model, report
end

function _predict(fitresult::MixtureDensityNetwork, X::Matrix{Float64})
    μ, σ, pi = fitresult(X)
	dists = Distributions.MixtureModel[]
	for i in eachindex(μ[1,:])
		d = Distributions.MixtureModel(Distributions.Normal.(μ[:,i], σ[:,i]), pi[:,i])
		push!(dists, d)
	end
	return dists
end

function _predict_mean(fitresult::MixtureDensityNetwork, X::Matrix{Float64})
    return _predict(fitresult, X) .|> mean
end

function _predict_mode(fitresult::MixtureDensityNetwork, X::Matrix{Float64})
    μ, σ, w = fitresult(X)
    max_priors = mapslices(argmax, w, dims=1)[1,:]
    return [μ[max_prior,i] for (i, max_prior) in enumerate(max_priors)]
end

function _predict_median(fitresult::MixtureDensityNetwork, X::Matrix{Float64})
    return _predict(fitresult, X) .|> median
end