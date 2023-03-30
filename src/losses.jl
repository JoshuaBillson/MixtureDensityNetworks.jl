"""
    likelihood_loss(μ, σ, pi, y)

Conpute the negative log-likelihood loss for a set of labels `y` under a Gaussian Mixture Model given by the parameters `μ`, `σ`, and `pi`.
"""
function likelihood_loss(μ::AbstractMatrix{Float32}, σ::AbstractMatrix{Float32}, pi::AbstractMatrix{Float32}, y::AbstractMatrix{Float32})
	@pipe pi .* (1.0 ./ ((sqrt(2π) .* σ)).*exp.(-0.5((y .- μ) ./ σ).^2.0)) |>
	sum(_, dims=1) |>
	clamp.(_, eps(Float64), 10000.0) |>
	-log.(_) |>
	mean(_)
end