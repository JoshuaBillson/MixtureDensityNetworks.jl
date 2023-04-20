var documenterSearchIndex = {"docs":
[{"location":"mlj/","page":"MLJ Compatibility","title":"MLJ Compatibility","text":"CurrentModule = MixtureDensityNetworks","category":"page"},{"location":"mlj/#MLJ-Compatibility","page":"MLJ Compatibility","title":"MLJ Compatibility","text":"","category":"section"},{"location":"mlj/","page":"MLJ Compatibility","title":"MLJ Compatibility","text":"This package implements the interface specified by MLJModelInterface and is thus fully compatible with the MLJ ecosystem. Below is an example demonstrating the use of this package in conjunction with MLJ. ","category":"page"},{"location":"mlj/","page":"MLJ Compatibility","title":"MLJ Compatibility","text":"using MixtureDensityNetworks, Distributions, Logging, TerminalLoggers, CairoMakie, MLJ, Random\n\nRandom.seed!(123)\n\nconst n_samples = 1000\nconst epochs = 500\nconst batchsize = 128\nconst mixtures = 8\nconst layers = [128, 128]\n\nfunction main()\n    # Generate Data\n    X, Y = generate_data(n_samples)\n\n    # Create Model\n    mach = MLJ.machine(MDN(epochs=epochs, mixtures=mixtures, layers=layers, batchsize=batchsize), MLJ.table(X'), Y[1,:])\n\n    # Fit Model on Training Data, Then Evaluate on Test\n    with_logger(TerminalLogger()) do \n        @info \"Evaluating...\"\n        evaluation = MLJ.evaluate!(\n            mach, \n            resampling=Holdout(shuffle=true), \n            measure=[rsq, rmse, mae, mape], \n            operation=MLJ.predict_mean\n        )\n        names = [\"R²\", \"RMSE\", \"MAE\", \"MAPE\"]\n        metrics = round.(evaluation.measurement, digits=3)\n        @info \"Metrics: \" * join([\"$name: $metric\" for (name, metric) in zip(names, metrics)], \", \")\n    end\n\n    # Fit Model on Entire Dataset\n    with_logger(TerminalLogger()) do \n        @info \"Training...\"\n        MLJ.fit!(mach)\n    end\n\n    # Plot Learning Curve\n    fig, _, _ = lines(1:epochs, MLJ.training_losses(mach), axis=(;xlabel=\"Epochs\", ylabel=\"Loss\"))\n    save(\"LearningCurve.png\", fig)\n\n    # Plot Learned Distribution\n    Ŷ = MLJ.predict(mach) .|> rand\n    fig, ax, plt = scatter(X[1,:], Ŷ, markersize=4, label=\"Predicted Distribution\")\n    scatter!(ax, X[1,:], Y[1,:], markersize=3, label=\"True Distribution\")\n    axislegend(ax, position=:lt)\n    save(\"PredictedDistribution.png\", fig)\n\n    # Plot Conditional Distribution\n    cond = MLJ.predict(mach, MLJ.table(reshape([-2.1], (1,1))))[1]\n    fig = Figure(resolution=(1000, 500))\n    density(fig[1,1], rand(cond, 10000), npoints=10000)\n    save(\"ConditionalDistribution.png\", fig)\nend\n\nmain()","category":"page"},{"location":"reference/","page":"API (Reference Manual)","title":"API (Reference Manual)","text":"CurrentModule = MixtureDensityNetworks","category":"page"},{"location":"reference/#Index","page":"API (Reference Manual)","title":"Index","text":"","category":"section"},{"location":"reference/","page":"API (Reference Manual)","title":"API (Reference Manual)","text":"","category":"page"},{"location":"reference/#API","page":"API (Reference Manual)","title":"API","text":"","category":"section"},{"location":"reference/","page":"API (Reference Manual)","title":"API (Reference Manual)","text":"Modules = [MixtureDensityNetworks]\nPrivate = false","category":"page"},{"location":"reference/#MixtureDensityNetworks.MDN","page":"API (Reference Manual)","title":"MixtureDensityNetworks.MDN","text":"MDN\n\nA model type for constructing a MDN, based on MixtureDensityNetworks.jl, and implementing the MLJ model interface.\n\nFrom MLJ, the type can be imported using\n\nMDN = @load MDN pkg=MixtureDensityNetworks\n\nDo model = MDN() to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in MDN(mixtures=...).\n\nA neural network which parameterizes a Gaussian Mixture Model (GMM)  distributed over the target varible y conditioned on the features X.\n\nTraining Data\n\nIn MLJ or MLJBase, bind an MDN instance model to data with     mach = machine(model, X, y) where\n\nX: any table of input features (eg, a DataFrame) whose columns belong to the Continuous scitypes`.\ny: the target, which can be any AbstractVector whose element scitype is Continuous.\n\nHyperparameters\n\nmixtures=5: number of Gaussian mixtures in the predicted distribution\nlayers=[128,]: hidden layer topology, starting from the first hidden layer\nη=1e-3: learning rate used for the optimizer\nepochs=1: number of epochs to train the model\nbatchsize=32: batch size used during training\n\nOperations\n\npredict(mach, Xnew): return the distributions over the target conditioned on the new features Xnew having the same scitype as X above.\npredict_mode(mach, Xnew): return the largest modes of the distributions over targets   conditioned on the new features Xnew having the same scitype as X above.\npredict_mean(mach, Xnew): return the means of the distributions over targets   conditioned on the new features Xnew having the same scitype as X above.\npredict_median(mach, Xnew): return the medians of the distributions over targets   conditioned on the new features Xnew having the same scitype as X above.\n\nFitted Parameters\n\nThe fields of fitted_params(mach) are:\n\nfitresult: the trained mixture density model, compatible with the Flux ecosystem.\n\nReport\n\nlearning_curve: the average training loss for each epoch.\nbest_epoch: the epoch (starting from 1) with the lowest training loss.\nbest_loss: the best (lowest) loss encountered durind training. Corresponds  to the average loss of the best epoch.\n\nAccessor Functions\n\ntraining_losses(mach) returns the learning curve as a vector of average  training losses for each epoch.\n\nExamples\n\nusing MLJ\nMDN = @load MDN pkg=MixtureDensityNetworks\nmdn = MDN(mixtures=12, epochs=100, layers=[512, 256, 128])\nX, y = make_regression(100, 1) # synthetic data\nmach = machine(mdn, X, y) |> fit!\nXnew, _ = make_regression(3, 1)\nŷ = predict(mach, Xnew) # new predictions\nreport(mach).best_epoch # best epoch encountered during training \nreport(mach).best_loss # best loss encountered during training \ntraining_losses(mach) # learning curve\n\n\n\n\n\n","category":"type"},{"location":"reference/#MixtureDensityNetworks.MDN-Tuple{}","page":"API (Reference Manual)","title":"MixtureDensityNetworks.MDN","text":"MDN(; mixtures=5, layers=[128], η=1e-3, epochs=1, batchsize=32)\n\nDefines an MDN model with the given hyperparameters.\n\nParameters\n\nmixtures: The number of gaussian mixtures to use in estimating the conditional distribution (default=5).\nlayers: A vector indicating the number of nodes in each of the hidden layers (default=[128,]).\nη: The learning rate to use when training the model (default=1e-3).\nepochs: The number of epochs to train the model (default=1).\nbatchsize: The batchsize to use during training (default=32).\n\n\n\n\n\n","category":"method"},{"location":"reference/#MixtureDensityNetworks.MixtureDensityNetwork","page":"API (Reference Manual)","title":"MixtureDensityNetworks.MixtureDensityNetwork","text":"struct MixtureDensityNetwork{T}\n\nA custom Flux model whose predictions paramaterize a Gaussian Mixture Model.\n\nParameters\n\nhidden::Flux.Chain\noutput::Any\n\n\n\n\n\n","category":"type"},{"location":"reference/#MixtureDensityNetworks.MixtureDensityNetwork-Tuple{Int64, Int64, Vector{Int64}, Int64}","page":"API (Reference Manual)","title":"MixtureDensityNetworks.MixtureDensityNetwork","text":"MixtureDensityNetwork(\n    input::Int64,\n    output::Int64,\n    layers::Vector{Int64},\n    mixtures::Int64\n) -> Union{MixtureDensityNetwork{MultivariateGMM}, MixtureDensityNetwork{UnivariateGMM}}\n\n\nConstruct a standard Mixture Density Network.\n\nParameters\n\ninput: The length of the input feature vectors.\noutput: The length of the output feature vectors.\nlayers: The topolgy of the hidden layers, starting from the first layer.\nmixtures: The number of Gaussian mixtures to use in the predicted distribution.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MixtureDensityNetworks.MultivariateGMM","page":"API (Reference Manual)","title":"MixtureDensityNetworks.MultivariateGMM","text":"struct MultivariateGMM\n\nA layer which produces a multivariate Gaussian Mixture Model as its output.\n\nParameters\n\noutputs::Int64\nmixtures::Int64\nμ::Flux.Dense\nΣ::Flux.Dense\nw::Flux.Chain\n\n\n\n\n\n","category":"type"},{"location":"reference/#MixtureDensityNetworks.MultivariateGMM-Tuple{Int64, Int64, Int64}","page":"API (Reference Manual)","title":"MixtureDensityNetworks.MultivariateGMM","text":"MultivariateGMM(\n    input::Int64,\n    output::Int64,\n    mixtures::Int64\n) -> MultivariateGMM\n\n\nConstruct a layer which returns a multivariate Gaussian Mixture Model as its output.\n\nParameters\n\ninput: Specifies the length of the feature vectors. The layer expects a matrix with the dimensions input x N as input.\noutput: Specifies the length of the label vectors. The layer returns a matrix with dimensions output x N as output.\nmixtures: The number of mixtures to use in the GMM.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MixtureDensityNetworks.UnivariateGMM","page":"API (Reference Manual)","title":"MixtureDensityNetworks.UnivariateGMM","text":"struct UnivariateGMM\n\nA layer which produces a univariate Gaussian Mixture Model as its output.\n\nParameters\n\nμ::Flux.Dense\nΣ::Flux.Dense\nw::Flux.Chain\n\n\n\n\n\n","category":"type"},{"location":"reference/#MixtureDensityNetworks.UnivariateGMM-Tuple{Int64, Int64}","page":"API (Reference Manual)","title":"MixtureDensityNetworks.UnivariateGMM","text":"UnivariateGMM(\n    input::Int64,\n    mixtures::Int64\n) -> UnivariateGMM\n\n\nConstruct a layer which returns a univariate Gaussian Mixture Model as its output.\n\nParameters\n\ninput: Specifies the length of the feature vectors. The layer expects a matrix with the dimensions input x N as input.\nmixtures: The number of mixtures to use in the GMM.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MixtureDensityNetworks.fit!-Tuple{Any, Matrix{<:Real}, Matrix{<:Real}}","page":"API (Reference Manual)","title":"MixtureDensityNetworks.fit!","text":"fit!(\n    m,\n    X::Matrix{<:Real},\n    Y::Matrix{<:Real};\n    opt,\n    batchsize,\n    epochs\n) -> Tuple{Any, NamedTuple{(:learning_curve, :best_epoch, :best_loss), Tuple{Vector{Float64}, Int64, Float64}}}\n\n\nFit the model to the data given by X and Y.\n\nParameters\n\nm: The model to be trained.\nX: A dxn matrix where d is the number of features and n is the number of samples.\nY: A 1xn matrix where n is the number of samples.\nopt: The optimization algorithm to use during training (default = Adam(1e-3)).\nbatchsize: The batch size dor each iteration of gradient descent (default = 32).\nepochs: The number of epochs to train for (default = 100).\n\n\n\n\n\n","category":"method"},{"location":"reference/#MixtureDensityNetworks.generate_data-Tuple{Int64}","page":"API (Reference Manual)","title":"MixtureDensityNetworks.generate_data","text":"generate_data(\n    n_samples::Int64\n) -> Tuple{Matrix{Float64}, Matrix{Float64}}\n\n\nGenerate some synthetic data for testing purposes. \n\nParameters\n\nn_samples: The number of samples we want to generate.\n\nReturns\n\nThe sythetic features X and labels Y as a tuple (X, Y).\n\n\n\n\n\n","category":"method"},{"location":"reference/#MixtureDensityNetworks.likelihood_loss-Tuple{Vector{<:Distributions.MixtureModel{Distributions.Multivariate}}, Matrix{<:Real}}","page":"API (Reference Manual)","title":"MixtureDensityNetworks.likelihood_loss","text":"likelihood_loss(\n    distributions::Vector{<:Distributions.MixtureModel{Distributions.Multivariate}},\n    y::Matrix{<:Real}\n)\n\n\nConpute the negative log-likelihood loss for a set of labels y under a set of multivariate Gaussian Mixture Models.\n\nParameters\n\ndistributions: A vector of multivariate Gaussian Mixture Model distributions.\ny: A kxn matrix of labels where k is the dimension of each label and n is the number of samples.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MixtureDensityNetworks.likelihood_loss-Tuple{Vector{<:Distributions.MixtureModel{Distributions.Univariate}}, Matrix{<:Real}}","page":"API (Reference Manual)","title":"MixtureDensityNetworks.likelihood_loss","text":"likelihood_loss(\n    distributions::Vector{<:Distributions.MixtureModel{Distributions.Univariate}},\n    y::Matrix{<:Real}\n) -> Any\n\n\nConpute the negative log-likelihood loss for a set of labels y under a set of univariate Gaussian Mixture Models.\n\nParameters\n\ndistributions: A vector of univariate Gaussian Mixture Model distributions.\ny: A 1xn matrix of labels where n is the number of samples.\n\n\n\n\n\n","category":"method"},{"location":"reference/#MixtureDensityNetworks.predict_mode-Tuple{MixtureDensityNetwork, Matrix{<:Real}}","page":"API (Reference Manual)","title":"MixtureDensityNetworks.predict_mode","text":"predict_mode(\n    m::MixtureDensityNetwork,\n    X::Matrix{<:Real}\n) -> Any\n\n\nPredict the point associated with the highest probability in the conditional distribution P(Y|X).\n\nParameters\n\nm: The model from which we want to generate a prediction.\nX: The input features to be passed to m. Expected to be a matrix with dimensions d x n where d is the length of each feature vector.\n\nReturns\n\nThe mode of each distribution returned by m(X).\n\n\n\n\n\n","category":"method"},{"location":"","page":"Introduction","title":"Introduction","text":"CurrentModule = MixtureDensityNetworks","category":"page"},{"location":"#MixtureDensityNetworks","page":"Introduction","title":"MixtureDensityNetworks","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"This package provides a simple interface for defining, training, and deploying Mixture Density Networks (MDNs). MDNs were first proposed by Bishop (1994). We can think of an MDN as a specialized type of Artificial Neural Network (ANN), which takes some features X and returns a distribution over the labels Y under a Gaussian Mixture Model (GMM). Unlike an ANN, MDNs maintain the full conditional distribution P(Y|X). This makes them particularly well-suited for situations where we want to maintain some measure of the uncertainty in our predictions. Moreover, because GMMs can represent multimodal distributions, MDNs are capable of modelling one-to-many relationships, which occurs when each input X can be associated with more than one output Y. ","category":"page"},{"location":"#Example","page":"Introduction","title":"Example","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"First, let's create our dataset. To properly demonstrate the power of MDNs, we'll generate a many-to-one dataset where each x-value can map to more than one y-value.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using Flux, Distributions, CairoMakie, MixtureDensityNetworks\n\nconst n_samples = 1000\n\nX, Y = generate_data(n_samples)\n\nfig, ax, plt = scatter(X[1,:], Y[1,:], markersize=5)","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"(Image: )","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Now we'll define a standard univariate MDN. For this example, we construct a network with 2 hidden layers of size 128, which outputs a distribution with 5 Gaussian mixtures.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"model = MixtureDensityNetwork(1, 1, [128, 128], 5)","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"We can fit our model to our data by calling fit!(m, X, Y; opt=Flux.Adam(), batchsize=32, epochs=100). We specify that we want to train our model for 500 epochs with the Adam optimiser and a batch size of 128. This method returns the model with the lowest loss as its first value and a named tuple  containing the learning curve, best epoch, and lowest loss observed during training as its second value. We can use Makie's lines method to visualize the learning curve.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"model, report = MixtureDensityNetworks.fit!(model, X, Y; epochs=500, opt=Flux.Adam(1e-3), batchsize=128)\nfig, _, _ = lines(1:500, lc, axis=(;xlabel=\"Epochs\", ylabel=\"Loss\"))","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"(Image: )","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Let's evaluate how well our model learned to replicate our data by plotting both the learned and true distributions. We observe that our model has indeed learned to replicate the true distribution.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Ŷ = model(X)\nfig, ax, plt = scatter(X[1,:], rand.(Ŷ), markersize=4, label=\"Predicted Distribution\")\nscatter!(ax, X[1,:], Y[1,:], markersize=3, label=\"True Distribution\")\naxislegend(ax, position=:lt)","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"(Image: )","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"We can also visualize the conditional distribution predicted by our model at x = -2.1.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"cond = model(reshape([-2.1], (1,1)))[1]\nfig = Figure(resolution=(1000, 500))\ndensity(fig[1,1], rand(cond, 10000), npoints=10000)","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"(Image: )","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Below is a script for running the complete example.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using MixtureDensityNetworks, Distributions, CairoMakie, Logging, TerminalLoggers\n\nconst n_samples = 1000\nconst epochs = 1000\nconst batchsize = 128\nconst mixtures = 8\nconst layers = [128, 128]\n\nfunction main()\n    # Generate Data\n    X, Y = generate_data(n_samples)\n\n    # Create Model\n    model = MixtureDensityNetwork(1, 1, layers, mixtures)\n\n    # Fit Model\n    model, report = with_logger(TerminalLogger()) do \n        MixtureDensityNetworks.fit!(model, X, Y; epochs=epochs, opt=Flux.Adam(1e-3), batchsize=batchsize)\n    end\n\n    # Plot Learning Curve\n    fig, _, _ = lines(1:epochs, report.learning_curve, axis=(;xlabel=\"Epochs\", ylabel=\"Loss\"))\n    save(\"LearningCurve.png\", fig)\n\n    # Plot Learned Distribution\n    Ŷ = model(X)\n    fig, ax, plt = scatter(X[1,:], rand.(Ŷ), markersize=4, label=\"Predicted Distribution\")\n    scatter!(ax, X[1,:], Y[1,:], markersize=3, label=\"True Distribution\")\n    axislegend(ax, position=:lt)\n    save(\"PredictedDistribution.png\", fig)\n\n    # Plot Conditional Distribution\n    cond = model(reshape([-2.1], (1,1)))[1]\n    fig = Figure(resolution=(1000, 500))\n    density(fig[1,1], rand(cond, 10000), npoints=10000)\n    save(\"ConditionalDistribution.png\", fig)\nend\n\nmain()","category":"page"}]
}
