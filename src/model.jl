"""
$(TYPEDEF)

A custom Flux model whose predictions paramaterize a Gaussian Mixture Model.

# Parameters
$(TYPEDFIELDS)
"""
struct MixtureDensityNetwork
    hidden::Flux.Chain
    μ::Flux.Dense
    Σ::Flux.Dense
    π::Flux.Chain
end

Flux.@functor MixtureDensityNetwork

"""
    MixtureDensityNetwork(input::Int, layers::Vector{Int}, mixtures::Int, dropout=0.0)

Construct a new MixtureDensityNetwork.

# Examples
```julia-repl
julia> MixtureDensityNetwork(5, [512, 256, 128, 64], 5)
```
"""
function MixtureDensityNetwork(input::Int, layers::Vector{Int}, mixtures::Int)
    # Define Weight Initializer
    init(out, in) = Float64.(Flux.glorot_uniform(out, in))

    # Construct Hidden Layers
    hidden = []
    layers = vcat([input], layers)
    for (dim_in, dim_out) in zip(layers, layers[2:end])
        push!(hidden, Flux.Dense(dim_in=>dim_out, init=init))
        push!(hidden, Flux.BatchNorm(dim_out, Flux.relu, initβ=zeros, initγ=ones, ϵ=1e-5, momentum=0.1))
    end
    hidden_layer = Flux.Chain(hidden...)
    
    # Construct Output Layer
    μ = Flux.Dense(layers[end]=>mixtures, init=init)
    Σ = Flux.Dense(layers[end]=>mixtures, exp, init=init)
    π = Flux.Chain(Flux.Dense(layers[end]=>mixtures, init=init), x -> Flux.softmax(x; dims=1))
    return MixtureDensityNetwork(hidden_layer, μ, Σ, π)
end

function (m::MixtureDensityNetwork)(X::AbstractMatrix{<:Real})
    Float64.(X) |> m
end

function (m::MixtureDensityNetwork)(X::AbstractMatrix{Float64})
    h = m.hidden(X)
    μ = m.μ(h)
    Σ = m.Σ(h)
    π = m.π(h)
    μ, Σ, π
end