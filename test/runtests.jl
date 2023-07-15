using MixtureDensityNetworks
using Test
using Random
using MLJ
using Flux

@testset "Native Interface" begin
    n_samples = 1000
    epochs = 500
    batchsize = 128
    mixtures = 8
    layers = [128, 128]

    Random.seed!(123)

    # Generate Data
    X, Y = generate_data(n_samples)

    # Create Model
    model = MixtureDensityNetwork(1, 1, layers, mixtures)

    # Fit Model
    fitresult, report = MixtureDensityNetworks.fit!(model, X, Y; epochs=epochs, opt=Flux.Adam(1e-3), batchsize=batchsize)

    @test report.best_loss < 1.60

    @test likelihood_loss(fitresult(X), Y) < 1.60
end

@testset "MLJ Interface" begin
    n_samples = 1000
    epochs = 500
    batchsize = 128
    mixtures = 8
    layers = [128, 128]

    Random.seed!(123)

    # Generate Data
    X, Y = generate_data(n_samples)

    # Create Model
    mach = MLJ.machine(MDN(epochs=epochs, mixtures=mixtures, layers=layers, batchsize=batchsize), MLJ.table(X'), Y[1,:])

    # Evaluate Model
    results = evaluation = MLJ.evaluate!(
        mach, 
        resampling=Holdout(shuffle=true), 
        measure=[rsq, rmse, mae, mape], 
        operation=MLJ.predict_mean
    )
    @test results.measurement[1] > 0.3
    @test results.measurement[2] < 5.0
    @test results.measurement[3] < 4.5
    @test results.measurement[4] < 1.5

    # Fit Model
    MLJ.fit!(mach)

    @test MLJ.report(mach).best_loss < 1.60

    @test likelihood_loss(MLJ.predict(mach), Y) < 1.60
end

