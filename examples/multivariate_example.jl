using Flux, MixtureDensityNetworks, Distributions, CairoMakie, Logging, TerminalLoggers

const n_samples = 1000
const epochs = 250
const batchsize = 256
const mixtures = 12
const layers = [256, 512, 1024]

function main()
    # Generate Data
    Y = rand(Uniform(-10.5, 10.5), 1, n_samples)
    μ_x = (7sin.(0.75 .* Y) + 0.5 .* Y)
    X = rand.(Normal.(μ_x, 0.5))
    μ_z = (-0.5 .* X) .+ 2.0
    Z = rand.(Normal.(μ_z, 0.6))
    Y = cat(Y, Z, dims=1)

    # Normalize Features
    X̄ = (X .- mean(X, dims=2)) ./ std(X, dims=2)

    # Create Model
    model = MixtureDensityNetwork(1, 2, layers, mixtures)

    # Fit Model
    model, report = with_logger(TerminalLogger()) do 
        MixtureDensityNetworks.fit!(model, X̄, Y; batchsize=batchsize, epochs=epochs)
    end

    # Plot Learning Curve
    fig, _, _ = lines(1:epochs, report.learning_curve, axis=(;xlabel="Epochs", ylabel="Loss"))
    save("MultivariateLearningCurve.png", fig)

    # Plot Learned Distribution
    Ŷ = model(X̄) .|> rand
    fig = Figure(resolution=(2000,1000), figure_padding=100)
    ax1 = Axis3(fig[1,1], title="True Distribution", elevation=0.2π, azimuth=0.25π, titlesize=48, titlegap=50)
    ax2 = Axis3(fig[1,2], title="Predicted Distribution", elevation=0.2π, azimuth=0.25π, titlesize=48, titlegap=50)
    scatter!(ax1, X[1,:], Y[1,:], Y[2,:], markersize=3.0)
    scatter!(ax2, X[1,:], [x[1] for x in Ŷ], [x[2] for x in Ŷ], markersize=3.0)
    xlims!(ax1, -15, 15); zlims!(ax1, -7, 10); ylims!(ax1, -13, 13)
    xlims!(ax2, -15, 15); zlims!(ax2, -7, 10); ylims!(ax2, -13, 13)
    save("MultivariateDistributions.png", fig)
end

main()