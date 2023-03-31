var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = MixtureDensityNetworks","category":"page"},{"location":"#MixtureDensityNetworks","page":"Home","title":"MixtureDensityNetworks","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for MixtureDensityNetworks.","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#API","page":"Home","title":"API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [MixtureDensityNetworks]\nPrivate = false","category":"page"},{"location":"#MixtureDensityNetworks.MDN","page":"Home","title":"MixtureDensityNetworks.MDN","text":"mutable struct MDN\n\nThe hyperparameters defining the classical MDN model.\n\nParameters\n\nmixtures::Int64\nlayers::Vector{Int64}\nη::Float64\nepochs::Int64\nbatchsize::Int64\nfitresult::Any\n\n\n\n\n\n","category":"type"},{"location":"#MixtureDensityNetworks.MDN-Tuple{}","page":"Home","title":"MixtureDensityNetworks.MDN","text":"MDN(; mixtures=5, layers=[128], η=1e-3, epochs=1, batchsize=32)\n\nDefines an MDN model with the given hyperparameters.\n\nParameters\n\nmixtures: The number of gaussian mixtures to use in estimating the conditional distribution (default=5).\nlayers: A vector indicating the number of nodes in each of the hidden layers (default=[128,]).\nη: The learning rate to use when training the model (default=1e-3).\nepochs: The number of epochs to train the model (default=1).\nbatchsize: The batchsize to use during training (default=32).\n\n\n\n\n\n","category":"method"},{"location":"#MixtureDensityNetworks.fit!-Tuple{MDN, Matrix{<:Number}, Matrix{<:Number}}","page":"Home","title":"MixtureDensityNetworks.fit!","text":"fit!(\n    model::MDN,\n    X::Matrix{<:Number},\n    Y::Matrix{<:Number}\n) -> Any\n\n\nFit the model to the data given by X and Y.\n\nParameters\n\nmodel: The MDN to be trained.\nX: A dxn matrix where d is the number of features and n is the number of samples.\nY: A 1xn matrix where n is the number of samples.\n\n\n\n\n\n","category":"method"},{"location":"#MixtureDensityNetworks.likelihood_loss-Tuple{Matrix{<:Real}, Matrix{<:Real}, Matrix{<:Real}, Matrix{<:Real}}","page":"Home","title":"MixtureDensityNetworks.likelihood_loss","text":"likelihood_loss(\n    μ::Matrix{<:Real},\n    σ::Matrix{<:Real},\n    pi::Matrix{<:Real},\n    y::Matrix{<:Real}\n) -> Float64\n\n\nConpute the negative log-likelihood loss for a set of labels y under a Gaussian Mixture Model defined by the parameters μ, σ, and pi.\n\nParameters\n\nμ: A mxn matrix of means where m is the number of Gaussian mixtures and n is the number of samples.\nσ: A mxn matrix of standard deviations where m is the number of Gaussian mixtures and n is the number of samples.\npi: A mxn matrix of priors where m is the number of Gaussian mixtures and n is the number of samples.\ny: A 1xn matrix of labels where n is the number of samples.\n\n\n\n\n\n","category":"method"},{"location":"#MixtureDensityNetworks.predict-Tuple{MDN, Matrix{<:Number}}","page":"Home","title":"MixtureDensityNetworks.predict","text":"predict(\n    model::MDN,\n    X::Matrix{<:Number}\n) -> Vector{Distributions.MixtureModel}\n\n\nPredict the full conditional distribution P(Y|X).\n\nParameters\n\nmodel: The MDN with which we want to generate a prediction.\nX: A dxn matrix where d is the number of features and n is the number of samples.\n\nReturns\n\nReturns a vector of Distributions.MixtureModel objects representing the conditional distribution for each sample.\n\n\n\n\n\n","category":"method"},{"location":"#MixtureDensityNetworks.predict_mean-Tuple{MDN, Matrix{<:Number}}","page":"Home","title":"MixtureDensityNetworks.predict_mean","text":"predict_mean(\n    model::MDN,\n    X::Matrix{<:Number}\n) -> AbstractVector\n\n\nPredict the mean of the conditional distribution P(Y|X). \n\nParameters\n\nmodel: The MDN with which we want to generate a prediction.\nX: A dxn matrix where d is the number of features and n is the number of samples.\n\nReturns\n\nReturns a vector of real numbers representing the mean of the conditional distribution P(Y|X) for each sample.\n\n\n\n\n\n","category":"method"},{"location":"#MixtureDensityNetworks.predict_mode-Tuple{MDN, Matrix{<:Number}}","page":"Home","title":"MixtureDensityNetworks.predict_mode","text":"predict_mode(model::MDN, X::Matrix{<:Number}) -> Any\n\n\nPredict the mean of the Gaussian with the largest prior in the conditional distribution P(Y|X). \n\nParameters\n\nmodel: The MDN with which we want to generate a prediction.\nX: A dxn matrix where d is the number of features and n is the number of samples.\n\nReturns\n\nReturns a vector of real numbers representing the mean of the gaussian with the largest prior for each sample.\n\n\n\n\n\n","category":"method"},{"location":"#Example","page":"Home","title":"Example","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using Distributions, CairoMakie # hide\nn_samples = 1000\nY = rand(Uniform(-10.5, 10.5), 1, n_samples)\nμ = 7sin.(0.75 .* Y) + 0.5 .* Y\nX = rand.(Normal.(μ, 1.0))\nnothing # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"fig, ax, plt = scatter(X[1,:], Y[1,:], markersize=5)\nsave(\"data.png\", fig)\nnothing # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"}]
}
