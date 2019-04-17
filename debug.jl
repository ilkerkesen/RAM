include("model.jl")
include(Knet.dir("data", "mnist.jl"))
using Debugger

σ = 1.
num_glimpses = 8
patch_size = 8
num_patches = 3
glimpse_scale = 1.
num_channels = 1
loc_hidden = 64
glimpse_hidden = 64
hidden_size = 128
num_classes = 10

ram = RAM(
    patch_size,
    num_patches,
    glimpse_scale,
    num_channels,
    loc_hidden,
    glimpse_hidden,
    σ,
    hidden_size,
    num_classes,
    num_glimpses,
)

dtrn, dtst = mnistdata()
x, y = first(dtrn)
B = length(y)
lt, ht = initstates(ram, B)
ram.controller.h = ht
