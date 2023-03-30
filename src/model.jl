"Struct for storing the internal state of an MDN."
struct MixtureModel
    hidden::Flux.Chain
    μ::Flux.Dense
    Σ::Flux.Dense
    π::Flux.Chain
end

Flux.@functor MixtureModel

"""
    MixtureModel(input::Int, layers::Vector{Int}, mixtures::Int, dropout=0.0)

Construct a new MDN.

# Examples
```julia-repl
julia> MixtureModel(5, [512, 256, 128, 64], 5)
```
"""
function MixtureModel(input::Int, layers::Vector{Int}, mixtures::Int)
    # Construct Hidden Layers
    hidden = []
    layers = vcat([input], layers)
    for (dim_in, dim_out) in zip(layers, layers[2:end])
        push!(hidden, Flux.Dense(dim_in=>dim_out))
        push!(hidden, Flux.BatchNorm(dim_out, Flux.relu))
    end
    hidden_layer = Flux.Chain(hidden...)
    
    # Construct Output Layer
    μ = Flux.Dense(layers[end]=>mixtures)
    Σ = Flux.Dense(layers[end]=>mixtures, exp)
    π = Flux.Chain(Flux.Dense(layers[end]=>mixtures), x -> Flux.softmax(x; dims=1))
    return MixtureModel(hidden_layer, μ, Σ, π)
end

"MixtureModel forward pass."
function (m::MixtureModel)(X::AbstractMatrix{Float32})
    # Forward Pass
    h = m.hidden(X)
    μ = m.μ(h)
    Σ = m.Σ(h)
    π = m.π(h)
    μ, Σ, π
end

"MixtureModel forward pass."
function (m::MixtureModel)(X::AbstractMatrix{<:Number})
    Float32.(X) |> m
end