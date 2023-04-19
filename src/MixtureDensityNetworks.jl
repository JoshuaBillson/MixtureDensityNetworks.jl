module MixtureDensityNetworks

import Flux
import Zygote
using Distributions
using Statistics
using ProgressLogging
using MLJModelInterface
using DocStringExtensions
using Pipe: @pipe

const MMI = MLJModelInterface

include("layers.jl")
include("model.jl")
include("losses.jl")
include("native_interface.jl")
include("mlj_interface.jl")

# Export Types
export MDN

# Export Layers
export UnivariateGMM, MultivariateGMM

# Export Models
export MixtureDensityNetwork

# Export Functions
export likelihood_loss, fit!, predict_mode, generate_data

end