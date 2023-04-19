"""
$(TYPEDSIGNATURES)

Fit the model to the data given by X and Y.

# Parameters
- `m`: The model to be trained.
- `X`: A dxn matrix where d is the number of features and n is the number of samples.
- `Y`: A 1xn matrix where n is the number of samples.
- `opt`: The optimization algorithm to use during training (default = Adam(1e-3)).
- `batchsize`: The batch size dor each iteration of gradient descent (default = 32).
- `epochs`: The number of epochs to train for (default = 100).
"""
function fit!(m, X::Matrix{<:Real}, Y::Matrix{<:Real}; opt=Flux.Adam(), batchsize=32, epochs=100)
    fit!(m, Float64.(X), Float64.(Y); opt=opt, batchsize=batchsize, epochs=epochs)
end

function fit!(m, X::Matrix{Float64}, Y::Matrix{Float64}; opt=Flux.Adam(), batchsize=32, epochs=100)
    # Get Parameters
    params = Flux.params(m)

    # Prepare Training Data
    data = Flux.DataLoader((X, Y); batchsize=batchsize, shuffle=true)

    # Iterate Over Epochs
    best_model = deepcopy(m)
    learning_curve = Float64[]
    @progress for epoch in 1:epochs

        # Iterate Over Data
        losses = Float64[]
        for (x, y) in data

            # Compute Loss and Gradient
            l, grad = Flux.withgradient(() -> likelihood_loss(m(x), y), params)

            # Update Parameters
            Flux.update!(opt, params, grad)

            # Save Loss
            push!(losses, l)
        end

        # Add Average Loss To Learning Curve
        push!(learning_curve, mean(losses))

        # Save Best Performing Model
        if length(learning_curve) == 1 || learning_curve[end] < minimum(learning_curve[1:end-1])
            best_model = deepcopy(m)
        end
    end

    # Return Results
    report = (learning_curve=learning_curve, best_epoch=argmin(learning_curve), best_loss=minimum(learning_curve))
    return best_model, report
end

"""
$(TYPEDSIGNATURES)

Generate some synthetic data for testing purposes. 

# Parameters
- `n_samples`: The number of samples we want to generate.

# Returns
The sythetic features `X` and labels `Y` as a tuple `(X, Y)`.
"""
function generate_data(n_samples::Int)
    Y = rand(Uniform(-10.5, 10.5), 1, n_samples)
    μ = 7sin.(0.75 .* Y) + 0.5 .* Y
    X = rand.(Normal.(μ, 1.0))
    return X, Y
end

function predict_mode(m::MixtureDensityNetwork, X::Matrix{Float64})
    dists = m(X)
    max_μ = [map(x->pdf(dist, x), [component.μ for component in dist.components]) |> argmax for dist in dists]
    return [dist.components[μ].μ for (μ, dist) in zip(max_μ, dists)]
end