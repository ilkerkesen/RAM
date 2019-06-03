include("model.jl")
include(Knet.dir("data", "mnist.jl"))
using Debugger
using NPZ
using LinearAlgebra
Knet.seed!(1)

σ = .17
num_glimpses = 2
patch_size = 8
num_patches = 1
glimpse_scale = 2.
num_channels = 1
loc_hidden = 128
glimpse_hidden = 128
hidden_size = 256
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

dtrn, dtst = mnistdata(;batchsize=32)
x, y = first(dtrn)
B = length(y)
lt, ht = initstates(ram, B)
# ram.controller.h = ht
atype = artype(ram)
etype = eltype(ram)

fc_w = npzread("weights/baseliner.fc.weight.npy") .+ 0
fc_b = npzread("weights/baseliner.fc.bias.npy") .+ 0
ram.baseline_net.w = Param(atype{etype}(fc_w))
ram.baseline_net.b = Param(atype{etype}(fc_b))

fc_w = npzread("weights/classifier.fc.weight.npy") .+ 0
fc_b = npzread("weights/classifier.fc.bias.npy") .+ 0
ram.softmax_layer.w = Param(atype{etype}(fc_w))
ram.softmax_layer.b = Param(atype{etype}(fc_b))

fc_w = npzread("weights/locator.fc.weight.npy") .+ 0
fc_b = npzread("weights/locator.fc.bias.npy") .+ 0
ram.location_net.layer.w = Param(atype{etype}(fc_w))
ram.location_net.layer.b = Param(atype{etype}(fc_b))

fc_w = npzread("weights/sensor.fc1.weight.npy") .+ 0
fc_b = npzread("weights/sensor.fc1.bias.npy") .+ 0
ram.glimpse_net.fc1.w = Param(atype{etype}(fc_w))
ram.glimpse_net.fc1.b = Param(atype{etype}(fc_b))

fc_w = npzread("weights/sensor.fc2.weight.npy") .+ 0
fc_b = npzread("weights/sensor.fc2.bias.npy") .+ 0
ram.glimpse_net.fc2.w = Param(atype{etype}(fc_w))
ram.glimpse_net.fc2.b = Param(atype{etype}(fc_b))

fc_w = npzread("weights/sensor.fc3.weight.npy") .+ 0
fc_b = npzread("weights/sensor.fc3.bias.npy") .+ 0
ram.glimpse_net.fc3.w = Param(atype{etype}(fc_w))
ram.glimpse_net.fc3.b = Param(atype{etype}(fc_b))

fc_w = npzread("weights/sensor.fc4.weight.npy") .+ 0
fc_b = npzread("weights/sensor.fc4.bias.npy") .+ 0
ram.glimpse_net.fc4.w = Param(atype{etype}(fc_w))
ram.glimpse_net.fc4.b = Param(atype{etype}(fc_b))


fc_w = npzread("weights/rnn.h2h.weight.npy") .+ 0
fc_b = npzread("weights/rnn.h2h.bias.npy") .+ 0
ram.controller.h2h.w = Param(atype{etype}(fc_w))
ram.controller.h2h.b = Param(atype{etype}(fc_b))

fc_w = npzread("weights/rnn.i2h.weight.npy") .+ 0
fc_b = npzread("weights/rnn.i2h.bias.npy") .+ 0
ram.controller.i2h.w = Param(atype{etype}(fc_w))
ram.controller.i2h.b = Param(atype{etype}(fc_b))

x = permutedims(npzread("weights/x.npy"), (4,3,2,1))
y = npzread("weights/y.npy") .+1
g_noise = atype{etype}(npzread("weights/noise.npy")' .+ 0.0)
lt, ht = initstates(ram, 32)

loss(x, ygold) = sum(ram(x,ygold)[1:3])
