# MixtureDensityNetworks

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JoshuaBillson.github.io/MixtureDensityNetworks.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoshuaBillson.github.io/MixtureDensityNetworks.jl/dev/)
[![Build Status](https://github.com/JoshuaBillson/MixtureDensityNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoshuaBillson/MixtureDensityNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JoshuaBillson/MixtureDensityNetworks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JoshuaBillson/MixtureDensityNetworks.jl)

Mixture Density Networks (MDNs) were first proposed by [Bishop (1994)](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf). We can think of them as a specialized type of neural network, which are typically employed when our data has a lot of uncertainty or when the relationship between features and labels is one-to-many. Unlike a traditional neural network, which predicts a point-estimate equal to the mode of the learned conditional distribution P(Y|X), an MDN maintains the full condtional distribution by predicting the parameters of a Gaussian Mixture Model (GMM). The multi-modal nature of GMMs are precisely what makes MDNs so well-suited to modeling one-to-many relationships. This package aims to provide a simple interface for defining, training, and deploying MDNs.

# Example

Below is an example of fitting an MDN to the visualized one-to-many distribution.

![](https://github.com/JoshuaBillson/MixtureDensityNetworks.jl/blob/main/docs/src/figures/PredictedDistribution.png?raw=true)

```julia
using MixtureDensityNetworks, Distributions, CairoMakie, Logging, TerminalLoggers

const n_samples = 1000
const epochs = 1000
const mixtures = 6
const layers = [128, 128]


function main()
    # Generate Data
    X, Y = generate_data(n_samples)

    # Create Model
    machine = MDN(epochs=epochs, mixtures=mixtures, layers=layers) |> Machine

    # Fit Model
    report = with_logger(ConsoleLogger()) do 
        fit!(machine, X, Y)
    end

    # Plot Learning Curve
    fig, _, _ = lines(1:epochs, report.learning_curve, axis=(;xlabel="Epochs", ylabel="Loss"))
    save("LearningCurve.png", fig)

    # Plot Learned Distribution
    Ŷ = predict(machine, X)
    fig, ax, plt = scatter(X[1,:], rand.(Ŷ), markersize=4, label="Predicted Distribution")
    scatter!(ax, X[1,:], Y[1,:], markersize=3, label="True Distribution")
    axislegend(ax, position=:lt)
    save("PredictedDistribution.png", fig)

    # Plot Conditional Distribution
    cond = predict(machine, reshape([-2.0], (1,1)))[1]
    fig = Figure(resolution=(1000, 500))
    density(fig[1,1], rand(cond, 10000), npoints=10000)
    save("ConditionalDistribution.png", fig)

    return machine
end

main()
```
