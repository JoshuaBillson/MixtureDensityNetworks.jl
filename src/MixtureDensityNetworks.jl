module MixtureDensityNetworks

import Flux

using Distributions

using Pipe: @pipe

include("model.jl")
include("interface.jl")
include("losses.jl")

export likelihood_loss, MDN, fit!, predict


end
