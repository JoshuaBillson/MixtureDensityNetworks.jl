using MixtureDensityNetworks
using Test
using Random
using MLJ

@testset "Native Interface" begin
    n_samples = 1000
    epochs = 1000
    mixtures = 6
    layers = [128, 128]

    Random.seed!(123)

    # Generate Data
    X, Y = generate_data(n_samples)

    # Create Model
    machine = MixtureDensityNetworks.Machine(MDN(epochs=epochs, mixtures=mixtures, layers=layers))

    # Fit Model
    MixtureDensityNetworks.fit!(machine, X, Y)

    @test machine.report.best_loss < 1.70

    @test likelihood_loss(machine.fitresult(X)..., Y) ≈ 1.4020062446704458 atol=0.1
end

@testset "MLJ Interface" begin
    n_samples = 1000
    epochs = 1000
    mixtures = 6
    layers = [128, 128]

    Random.seed!(123)

    # Generate Data
    X, Y = generate_data(n_samples)

    # Create Model
    mach = MLJ.machine(MDN(epochs=epochs, mixtures=mixtures, layers=layers), MLJ.table(X'), Y[1,:])

    # Evaluate Model
    results = evaluation = MLJ.evaluate!(
        mach, 
        resampling=Holdout(shuffle=true), 
        measure=[rsq, rmse, mae, mape], 
        operation=MLJ.predict_mean
    )

    @test all((≈).(results.measurement, [0.34, 4.90, 4.09, 1.10], atol=0.1))

    # Fit Model
    MLJ.fit!(mach)

    @test  MLJ.report(mach).best_loss < 1.70

    @test likelihood_loss(MLJ.fitted_params(mach).fitresult(X)..., Y) ≈ 1.4459579546517374 atol=0.1
end

