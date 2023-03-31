push!(LOAD_PATH,"../src/")

using MixtureDensityNetworks
using Documenter

DocMeta.setdocmeta!(MixtureDensityNetworks, :DocTestSetup, :(using MixtureDensityNetworks); recursive=true)

makedocs(;
    modules=[MixtureDensityNetworks],
    authors="Joshua Billson",
    repo="https://github.com/JoshuaBillson/MixtureDensityNetworks.jl/blob/{commit}{path}#{line}",
    sitename="MixtureDensityNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JoshuaBillson.github.io/MixtureDensityNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JoshuaBillson/MixtureDensityNetworks.jl",
    devbranch="main",
)
