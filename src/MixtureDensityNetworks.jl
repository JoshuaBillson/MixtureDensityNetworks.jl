module MixtureDensityNetworks

import Flux
using Distributions
using Statistics
using ProgressLogging
using MLJModelInterface
using DocStringExtensions
using Pipe: @pipe

const MMI = MLJModelInterface

include("model.jl")
include("losses.jl")
include("native_interface.jl")
include("mlj_interface.jl")

export MixtureDensityNetwork, Machine, likelihood_loss, MDN, fit!, predict, predict_mean, predict_mode, generate_data

end