push!(LOAD_PATH, "../src")

using MixtureDensityNetworks, Distributions, CairoMakie

const n_samples = 1000
const epochs = 1000
const mixtures = 5
const layers = [128, 128]


function main()
    # Generate Data
    Y = rand(Uniform(-10.5, 10.5), 1, n_samples)
    μ = 7sin.(0.75 .* Y) + 0.5 .* Y
    X = rand.(Normal.(μ, 1.0))

    # Create Model
    model = MDN(epochs=epochs, mixtures=mixtures, layers=layers)

    # Fit Model
    lc = fit!(model, X, Y)

    # Plot Learning Curve
    fig, _, _ = lines(1:epochs, lc, axis=(;xlabel="Epochs", ylabel="Loss"))
    save("LearningCurve.png", fig)

    # Plot Learned Distribution
    Ŷ = predict(model, X)
    fig, ax, plt = scatter(X[1,:], rand.(Ŷ), markersize=4, label="Predicted Distribution")
    scatter!(ax, X[1,:], Y[1,:], markersize=3, label="True Distribution")
    axislegend(ax, position=:lt)
    save("PredictedDistribution.png", fig)

    # Plot Conditional Distribution
    cond = predict(model, reshape([-2.0], (1,1)))[1]
    fig = Figure(resolution=(1000, 500))
    density(fig[1,1], rand(cond, 10000), npoints=10000)
    save("ConditionalDistribution.png", fig)
end

main()