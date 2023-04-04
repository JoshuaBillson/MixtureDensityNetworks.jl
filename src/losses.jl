"""
$(TYPEDSIGNATURES)

Conpute the negative log-likelihood loss for a set of labels `y` under a Gaussian Mixture Model defined by the parameters `μ`, `σ`, and `pi`.

# Parameters
- `μ`: A mxn matrix of means where m is the number of Gaussian mixtures and n is the number of samples.
- `σ`: A mxn matrix of standard deviations where m is the number of Gaussian mixtures and n is the number of samples.
- `pi`: A mxn matrix of priors where m is the number of Gaussian mixtures and n is the number of samples.
- `y`: A 1xn matrix of labels where n is the number of samples.
"""
function likelihood_loss(μ::Matrix{<:Real}, σ::Matrix{<:Real}, pi::Matrix{<:Real}, y::Matrix{<:Real})
    return likelihood_loss(Float64.(μ), Float64.(σ), Float64.(pi), Float64.(y))
end

function likelihood_loss(μ::Matrix{Float64}, σ::Matrix{Float64}, pi::Matrix{Float64}, y::Matrix{Float64})
	@pipe pi .* (1.0 ./ ((sqrt(2π) .* σ)).*exp.(-0.5((y .- μ) ./ σ).^2.0)) |>
	sum(_, dims=1) |>
	clamp.(_, eps(Float64), 10000.0) |>
	-log.(_) |>
	mean(_)
end