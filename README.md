# MixtureDensityNetworks

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JoshuaBillson.github.io/MixtureDensityNetworks.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoshuaBillson.github.io/MixtureDensityNetworks.jl/dev/)
[![Build Status](https://github.com/JoshuaBillson/MixtureDensityNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoshuaBillson/MixtureDensityNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JoshuaBillson/MixtureDensityNetworks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JoshuaBillson/MixtureDensityNetworks.jl)

This package provides a simple interface for defining, training, and deploying Mixture Density Networks (MDNs). MDNs were first proposed by [Bishop (1994)](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf). We can think of an MDN as a specialized type of Artificial Neural Network (ANN), which takes some features `X` and returns a distribution over the labels `Y` under a Gaussian Mixture Model (GMM). Unlike an ANN, MDNs maintain the full conditional distribution P(Y|X). This makes them particularly well-suited for situations where we want to maintain some measure of the uncertainty in our predictions. Moreover, because GMMs can represent multimodal distributions, MDNs are capable of modelling one-to-many relationships, which occurs when each input `X` can be associated with more than one output `Y`. 

![](https://github.com/JoshuaBillson/MixtureDensityNetworks.jl/blob/main/docs/src/figures/PredictedDistribution.png?raw=true)

# MLJ Compatibility

This package implements the interface specified by [MLJModelInterface](https://github.com/JuliaAI/MLJModelInterface.jl) and is thus fully compatible
with the MLJ ecosystem. Below is an example demonstrating the use of this package in conjunction with MLJ. 

# Example (Native Interface)

```julia
using Flux, MixtureDensityNetworks, Distributions, CairoMakie, Logging, TerminalLoggers

const n_samples = 1000
const epochs = 1000
const batchsize = 128
const mixtures = 8
const layers = [128, 128]

function main()
    # Generate Data
    X, Y = generate_data(n_samples)

    # Create Model
    model = MixtureDensityNetwork(1, 1, layers, mixtures)

    # Fit Model
    model, report = with_logger(TerminalLogger()) do 
        MixtureDensityNetworks.fit!(model, X, Y; epochs=epochs, opt=Flux.Adam(1e-3), batchsize=batchsize)
    end

    # Plot Learning Curve
    fig, _, _ = lines(1:epochs, report.learning_curve, axis=(;xlabel="Epochs", ylabel="Loss"))
    save("LearningCurve.png", fig)

    # Plot Learned Distribution
    Ŷ = model(X)
    fig, ax, plt = scatter(X[1,:], rand.(Ŷ), markersize=4, label="Predicted Distribution")
    scatter!(ax, X[1,:], Y[1,:], markersize=3, label="True Distribution")
    axislegend(ax, position=:lt)
    save("PredictedDistribution.png", fig)

    # Plot Conditional Distribution
    cond = model(reshape([-2.1], (1,1)))[1]
    fig = Figure(resolution=(1000, 500))
    density(fig[1,1], rand(cond, 10000), npoints=10000)
    save("ConditionalDistribution.png", fig)
end

main()
```

# Example (MLJ Interface)

```julia
using MixtureDensityNetworks, Distributions, Logging, TerminalLoggers, CairoMakie, MLJ, Random

const n_samples = 1000
const epochs = 500
const batchsize = 128
const mixtures = 8
const layers = [128, 128]

function main()
    # Generate Data
    X, Y = generate_data(n_samples)

    # Create Model
    mach = MLJ.machine(MDN(epochs=epochs, mixtures=mixtures, layers=layers, batchsize=batchsize), MLJ.table(X'), Y[1,:])

    # Fit Model on Training Data, Then Evaluate on Test
    with_logger(TerminalLogger()) do 
        @info "Evaluating..."
        evaluation = MLJ.evaluate!(
            mach, 
            resampling=Holdout(shuffle=true), 
            measure=[rsq, rmse, mae, mape], 
            operation=MLJ.predict_mean
        )
        names = ["R²", "RMSE", "MAE", "MAPE"]
        metrics = round.(evaluation.measurement, digits=3)
        @info "Metrics: " * join(["$name: $metric" for (name, metric) in zip(names, metrics)], ", ")
    end

    # Fit Model on Entire Dataset
    with_logger(TerminalLogger()) do 
        @info "Training..."
        MLJ.fit!(mach)
    end

    # Plot Learning Curve
    fig, _, _ = lines(1:epochs, MLJ.training_losses(mach), axis=(;xlabel="Epochs", ylabel="Loss"))
    save("LearningCurve.png", fig)

    # Plot Learned Distribution
    Ŷ = MLJ.predict(mach) .|> rand
    fig, ax, plt = scatter(X[1,:], Ŷ, markersize=4, label="Predicted Distribution")
    scatter!(ax, X[1,:], Y[1,:], markersize=3, label="True Distribution")
    axislegend(ax, position=:lt)
    save("PredictedDistribution.png", fig)

    # Plot Conditional Distribution
    cond = MLJ.predict(mach, MLJ.table(reshape([-2.1], (1,1))))[1]
    fig = Figure(resolution=(1000, 500))
    density(fig[1,1], rand(cond, 10000), npoints=10000)
    save("ConditionalDistribution.png", fig)
end

main()
```