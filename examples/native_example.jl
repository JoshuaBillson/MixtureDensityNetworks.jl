using MixtureDensityNetworks, Distributions, CairoMakie, Logging, TerminalLoggers

const n_samples = 1000
const epochs = 1000
const mixtures = 6
const layers = [128, 128]


function main()
    # Generate Data
    Y = rand(Uniform(-10.5, 10.5), 1, n_samples)
    μ = 7sin.(0.75 .* Y) + 0.5 .* Y
    X = rand.(Normal.(μ, 1.0))

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