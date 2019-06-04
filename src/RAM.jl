module RAM

const DIR = @__DIR__
const SAVEDIR = abspath(joinpath(DIR, "..", "checkpoints"))

include("model.jl")
include("train.jl")
include("visualization.jl")

end # module
