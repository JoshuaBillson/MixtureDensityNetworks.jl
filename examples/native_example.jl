using MixtureDensityNetworks, Distributions, CairoMakie, Logging, TerminalLoggers

const n_samples = 1000
const epochs = 1000
const mixtures = 6
const layers = [128, 128]

function main()
    # Generate Data
    X, Y = generate_data(n_samples)

    # Create Model
    machine = MixtureDensityNetworks.Machine(MDN(epochs=epochs, mixtures=mixtures, layers=layers))

    # Fit Model
    report = with_logger(TerminalLogger()) do 
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
end

main()