"""
$(TYPEDEF)

A Flux model for implementing a standard Mixture Density Network.

# Parameters
$(TYPEDFIELDS)
"""
struct MixtureDensityNetwork{T}
    hidden::Flux.Chain
    output::T
end

Flux.@functor MixtureDensityNetwork (hidden, output)

"""
$(TYPEDSIGNATURES)

Construct a standard Mixture Density Network.

# Parameters
- `input`: The dimension of the input features.
- `output`: The dimension of the output. Setting output = 1 indicates a univariate model, whereas output > 1 indicates a multivariate model.
- `layers`: The topolgy of the hidden layers, starting from the first layer.
- `mixtures`: The number of Gaussian mixtures to use in the predicted distribution.
"""
function MixtureDensityNetwork(input::Int, output::Int, layers::Vector{Int}, mixtures::Int)
    # Define Weight Initializer
    init(out, in) = Float64.(Flux.glorot_uniform(out, in))

    # Construct Hidden Layers
    hidden = []
    layers = vcat([input], layers)
    for (dim_in, dim_out) in zip(layers, layers[2:end])
        push!(hidden, Flux.Dense(dim_in=>dim_out, init=init))
        push!(hidden, Flux.BatchNorm(dim_out, Flux.relu, initβ=zeros, initγ=ones, eps=1e-5, momentum=0.1))
    end
    hidden_layer = Flux.Chain(hidden...)
    
    # Construct Output Layer
    output = output == 1 ? UnivariateGMM(layers[end], mixtures) : MultivariateGMM(layers[end], output, mixtures)

    return MixtureDensityNetwork(hidden_layer, output)
end

function (m::MixtureDensityNetwork)(X::AbstractMatrix{<:Real})
    Float64.(X) |> m
end

function (m::MixtureDensityNetwork)(X::AbstractMatrix{Float64})
    return @pipe m.hidden(X) |> m.output
end