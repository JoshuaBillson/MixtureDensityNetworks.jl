```@meta
CurrentModule = MixtureDensityNetworks
```

# MixtureDensityNetworks

Documentation for [MixtureDensityNetworks](https://github.com/JoshuaBillson/MixtureDensityNetworks.jl).

## Index

```@index
```

## API

```@autodocs
Modules = [MixtureDensityNetworks]
Private = false
```

## Example

```@example 1
using Distributions, CairoMakie # hide
n_samples = 1000
Y = rand(Uniform(-10.5, 10.5), 1, n_samples)
μ = 7sin.(0.75 .* Y) + 0.5 .* Y
X = rand.(Normal.(μ, 1.0))
nothing # hide
```

```@example 1
fig, ax, plt = scatter(X[1,:], Y[1,:], markersize=5)
save("data.png", fig)
nothing # hide
```

![](data.png)
