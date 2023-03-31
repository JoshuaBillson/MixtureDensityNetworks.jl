module MixtureDensityNetworks

import Flux

using Distributions

using Pipe: @pipe

include("model.jl")
include("losses.jl")
include("interface.jl")

export likelihood_loss, MDN, fit!, predict


end
