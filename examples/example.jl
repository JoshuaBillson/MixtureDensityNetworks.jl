push!(LOAD_PATH, "../src")

using MixtureDensityNetworks, Distributions, CairoMakie

const n_samples = 1000


function main()
    Y = rand(Uniform(-10.5, 10.5), 1, n_samples)
    μ = 7sin.(0.75 .* Y) + 0.5 .* Y
    X = rand.(Normal.(μ, 1.0))

    model = MDN(epochs=200, mixtures=10, layers=[128, 128, 128])

    # Fit Model
    lc = fit!(model, X, Y)

    # Plot Learning Curve
    fig, _, _ = lines(1:200, lc)
    save("LearningCurve.png", fig)

    Ŷ = predict(model, X)

    fig, ax, plt = scatter(X[1,:], rand.(Ŷ))

    save("predicted_distribution.png", fig)

end