"""
$(TYPEDEF)

A layer which produces a univariate Gaussian Mixture Model as its output.

# Parameters
$(TYPEDFIELDS)
"""
struct UnivariateGMM
    μ::Flux.Dense
    Σ::Flux.Dense
    w::Flux.Chain
end

Flux.@functor UnivariateGMM

"""
$(TYPEDSIGNATURES)

Construct a layer which returns a univariate Gaussian Mixture Model as its output.

# Parameters
- `input`: Specifies the length of the feature vectors. The layer expects a matrix with the dimensions `input x N` as input.
- `mixtures`: The number of mixtures to use in the GMM.
"""
function UnivariateGMM(input::Int, mixtures::Int)
    # Define Weight Initializer
    init(out, in) = Float64.(Flux.glorot_uniform(out, in))

    # Construct Output Layer
    μ = Flux.Dense(input=>mixtures, init=init)
    Σ = Flux.Dense(input=>mixtures, exp, init=init)
    w = Flux.Chain(Flux.Dense(input=>mixtures, init=init), x -> Flux.softmax(x; dims=1))
    return UnivariateGMM(μ, Σ, w)
end

function (m::UnivariateGMM)(X::AbstractMatrix{<:Real})
    Float64.(X) |> m
end

function (m::UnivariateGMM)(X::AbstractMatrix{Float64})
    # Forward Pass
    μ = m.μ(X)
    σ = m.Σ(X)
    w = m.w(X)
 
    # Return Distributions
    return map(eachindex(μ[1,:])) do i
        Distributions.MixtureModel(Distributions.Normal.(μ[:,i], σ[:,i]), w[:,i])
    end
end

"""
$(TYPEDEF)

A layer which produces a multivariate Gaussian Mixture Model as its output.

# Parameters
$(TYPEDFIELDS)
"""
struct MultivariateGMM
    outputs::Int
    mixtures::Int
    μ::Flux.Dense
    Σ::Flux.Dense
    w::Flux.Chain
end

Flux.@functor MultivariateGMM (μ, Σ, w)

"""
$(TYPEDSIGNATURES)

Construct a layer which returns a multivariate Gaussian Mixture Model as its output.

# Parameters
- `input`: Specifies the length of the feature vectors. The layer expects a matrix with the dimensions `input x N` as input.
- `output`: Specifies the length of the label vectors. The layer returns a matrix with dimensions `output x N` as output.
- `mixtures`: The number of mixtures to use in the GMM.
"""
function MultivariateGMM(input::Int, output::Int, mixtures::Int)
    # Define Weight Initializer
    init(out, in) = Float64.(Flux.glorot_uniform(out, in))

    # Construct Output Layer
    μ = Flux.Dense(input=>(output * mixtures), init=init)
    Σ = Flux.Dense(input=>(output * output * mixtures), init=init)
    w = Flux.Chain(Flux.Dense(input=>mixtures, init=init), x -> Flux.softmax(x; dims=1))
    
    # Return Layer
    return MultivariateGMM(output, mixtures, μ, Σ, w)
end

function (m::MultivariateGMM)(X::AbstractMatrix{<:Real})
    Float64.(X) |> m
end

function (m::MultivariateGMM)(X::AbstractMatrix{Float64})
    # Forward Pass
    μ = reshape(m.μ(X), (m.mixtures, m.outputs, :))
    Σ = reshape(m.Σ(X), (m.mixtures, m.outputs, m.outputs, :))
    w = reshape(m.w(X), (m.mixtures, :))

    # Get Cholesky Decomposition Of Σ
    d_mask = [b == c ? 1.0 : 0.0 for a in 1:1, b in 1:m.outputs, c in 1:m.outputs, d in 1:1]
    u_mask = [b < c ? 1.0 : 0.0 for a in 1:1, b in 1:m.outputs, c in 1:m.outputs, d in 1:1]
    U = exp.(Σ .* d_mask) .+ (Σ .* u_mask)

    # Return Distributions
    return map(eachindex(w[1,:])) do obs
        MixtureModel([MultivariateNormal(μ[mixture,:,obs], U[mixture,:,:,obs]' * U[mixture,:,:,obs] + 1e-9I) for mixture in eachindex(μ[:,1,1])], w[:,obs])
    end
end