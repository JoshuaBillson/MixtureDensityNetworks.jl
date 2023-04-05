function MLJModelInterface.reformat(::MDN, X, y) 
    return (matrix(X, transpose=true), reshape(y, (1,:)))
end

function MLJModelInterface.reformat(::MDN, X) 
    return (matrix(X, transpose=true),)
end

function MLJModelInterface.selectrows(::MDN, I, Xmatrix, y)
	return (Xmatrix[:,I], y[:,I])
end

function MLJModelInterface.selectrows(::MDN, I, Xmatrix)
    return (Xmatrix[:,I],)
end

function MLJModelInterface.clean!(model::MDN)
	warning = ""
	if model.mixtures <= 0
		warning *= "Need mixtures > 0. Resetting mixtures=5. "
		model.mixtures = 5
	end
	if isempty(model.layers)
		warning *= "Need at least one hidden layer. Resetting layers=[128,]. "
		model.layers = [128,]
	end
	if model.η <= 0
		warning *= "Need η > 0. Resetting η=1e-3. "
		model.η = 1e-3 
	end
	if model.epochs <= 0
		warning *= "Need epochs > 0. Resetting epochs=1. "
		model.epochs = 1
	end
	if model.batchsize <= 0
		warning *= "Need batchsize > 0. Resetting batchsize=32. "
		model.batchsize = 32
	end
	return warning
end

function MLJModelInterface.fit(model::MDN, verbosity, X, y)
    fitresult, report = _fit(model, nothing, X, y)
    return fitresult, nothing, report
end

function MLJModelInterface.update(model::MDN, verbosity, old_fitresult, old_cache, X, y)
    fitresult, report = _fit(model, old_fitresult, X, y)
    return fitresult, old_cache, report
end

function MLJModelInterface.predict(model::MDN, fitresult, Xnew)
    return _predict(fitresult, Xnew)
end

function MLJModelInterface.predict_mode(model::MDN, fitresult, Xnew)
    return _predict_mode(fitresult, Xnew)
end

function MLJModelInterface.supports_training_losses(::Type{MDN})
    return true
end

function MLJModelInterface.iteration_parameter(::Type{MDN})
    return :epochs
end

function MLJModelInterface.training_losses(::MDN, report)
    return report.learning_curve
end

MLJModelInterface.metadata_pkg(
  MDN,
  package_name="MixtureDensityNetworks",
  package_uuid="521d8788-cab4-41cb-a05a-da376f16ad79",
  package_url="https://github.com/JoshuaBillson/MixtureDensityNetworks.jl",
  is_pure_julia=true, 
  package_license="MIT", 
)

MLJModelInterface.metadata_model(
  MDN,
  input_scitype=MMI.Table(MMI.Continuous),
  target_scitype=AbstractVector{<:MMI.Continuous},
  load_path="MixtureDensityNetworks.MDN", 
  human_name="MDN", 
)

"""
$(MLJModelInterface.doc_header(MDN))

A neural network which parameterizes a Gaussian Mixture Model (GMM) 
distributed over the target varible `y` conditioned on the features `X`.

# Training Data
In MLJ or MLJBase, bind an `MDN` instance `model` to data with
    mach = machine(model, X, y)
where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  belong to the `Continuous` scitypes`.
- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`.

# Hyperparameters
- `mixtures=5`: number of Gaussian mixtures in the predicted distribution
- `layers=[128,]`: hidden layer topology, starting from the first hidden layer
- `η=1e-3`: learning rate used for the optimizer
- `epochs=1`: number of epochs to train the model
- `batchsize=32`: batch size used during training

# Operations
- `predict(mach, Xnew)`: return the distributions over the target conditioned on the new
  features `Xnew` having the same scitype as `X` above.
- `predict_mode(mach, Xnew)`: return the largest modes of the distributions over targets 
   conditioned on the new features `Xnew` having the same scitype as `X` above.
- `predict_mean(mach, Xnew)`: return the means of the distributions over targets 
   conditioned on the new features `Xnew` having the same scitype as `X` above.
- `predict_median(mach, Xnew)`: return the medians of the distributions over targets 
   conditioned on the new features `Xnew` having the same scitype as `X` above.

# Fitted Parameters
The fields of `fitted_params(mach)` are:
- `fitresult`: the trained mixture density model, compatible with the Flux ecosystem.

# Report
- `learning_curve`: the average training loss for each epoch.
- `best_epoch`: the epoch (starting from 1) with the lowest training loss.
- `best_loss`: the best (lowest) loss encountered durind training. Corresponds
   to the average loss of the best epoch.

# Accessor Functions
- `training_losses(mach)` returns the learning curve as a vector of average
   training losses for each epoch.

# Examples
```
using MLJ
MDN = @load MDN pkg=MixtureDensityNetworks
mdn = MDN(mixtures=12, epochs=100, layers=[512, 256, 128])
X, y = make_regression(100, 1) # synthetic data
mach = machine(mdn, X, y) |> fit!
Xnew, _ = make_regression(3, 1)
ŷ = predict(mach, Xnew) # new predictions
report(mach).best_epoch # best epoch encountered during training 
report(mach).best_loss # best loss encountered during training 
training_losses(mach) # learning curve
```
"""
MDN