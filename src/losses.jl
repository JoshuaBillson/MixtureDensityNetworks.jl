"""
$(TYPEDSIGNATURES)

Conpute the negative log-likelihood loss for a set of labels `y` under a set of univariate Gaussian Mixture Models.

# Parameters
- `distributions`: A vector of univariate Gaussian Mixture Model distributions.
- `y`: A 1xn matrix of labels where n is the number of samples.
"""
function likelihood_loss(distributions::Vector{<:MixtureModel{Univariate}}, y::Matrix{<:Real})
    return likelihood_loss(distributions, Float64.(y))
end

function likelihood_loss(distributions::Vector{<:MixtureModel{Univariate}}, y::Matrix{Float64})
    return -logpdf.(distributions, y[1,:]) |> mean
end

"""
$(TYPEDSIGNATURES)

Conpute the negative log-likelihood loss for a set of labels `y` under a set of multivariate Gaussian Mixture Models.

# Parameters
- `distributions`: A vector of multivariate Gaussian Mixture Model distributions.
- `y`: A kxn matrix of labels where k is the dimension of each label and n is the number of samples.
"""
function likelihood_loss(distributions::Vector{<:MixtureModel{Multivariate}}, y::Matrix{<:Real})
    return likelihood_loss(distributions, Float64.(y))
end

function likelihood_loss(distributions::Vector{<:MixtureModel{Multivariate}}, y::Matrix{Float64})
    return -logpdf.(distributions, eachcol(y)) |> mean
end