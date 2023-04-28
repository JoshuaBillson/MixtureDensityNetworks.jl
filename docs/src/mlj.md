```@meta
CurrentModule = MixtureDensityNetworks
```

# MLJ Compatibility

This package implements the interface specified by [MLJModelInterface](https://github.com/JuliaAI/MLJModelInterface.jl) and is thus fully compatible
with the MLJ ecosystem. Below is an example demonstrating the use of this package in conjunction with MLJ. 

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