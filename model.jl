using Knet
using Sloth
import Base: eltype
using Random
using Statistics


struct Chain; layers; end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)


mutable struct VanillaRNN
    i2h
    h2h
end


function VanillaRNN(xsize::Int, hsize::Int)
    return VanillaRNN(Linear(xsize, hsize), Linear(hsize, hsize))
end


(m::VanillaRNN)(x, ht) = relu.(m.i2h(x) + m.h2h(ht))


mutable struct RAM
    σ
    num_glimpses
    glimpse_net
    controller
    location_net
    softmax_layer
    baseline_net
    locations
    history
end


function RAM(patch_size, num_patches, glimpse_scale, num_channels, loc_hidden,
             glimpse_hidden, σ, hidden_size, num_classes, num_glimpses)
    return RAM(
        σ,
        num_glimpses,
        GlimpseNet(loc_hidden, glimpse_hidden, patch_size,
                   num_patches, glimpse_scale, num_channels),
        # RNN(hidden_size, hidden_size; rnnType=:relu),
        VanillaRNN(hidden_size, hidden_size),
        LocationNet(hidden_size, 2, σ),
        Linear(hidden_size, num_classes),
        BaselineNet(hidden_size, 1),
        [],
        zeros(6)) # 1-3=>loss,4=>iter,5=>correct,6=># of samples
end


# just one step
function (ram::RAM)(x, l_prev, h_prev)
    gt = ram.glimpse_net(x, l_prev)
    ht = ram.controller(gt, h_prev)
    H, B = size(ht); ht = reshape(ht, H, B)
    μ, lt = ram.location_net(ht)
    bt = ram.baseline_net(ht)
    σ = ram.σ
    σ² = σ .^ 2
    logπ = -(abs.(lt - μ) .^ 2) / 2σ² .- log(σ) .- log(√(2π))
    logπ = sum(logπ, dims=1)
    return ht, lt, bt, logπ
end


# all timesteps
function (ram::RAM)(x)
    B = size(x)[end]
    lt, ht = initstates(ram, B)
    # ram.controller.h = ht
    xs, logπs, baselines, ram.locations = [], [], [], []
    for t = 1:ram.num_glimpses
        ht, lt, bt, logπ = ram(x, lt, ht)
        push!(ram.locations, lt)
        push!(baselines, bt)
        push!(logπs, logπ)
    end
    baseline, logπ = vcat(baselines...), vcat(logπs...)
    scores = ram.softmax_layer(mat(ht))
    return scores, baseline, logπ, ht
end


# loss
function (ram::RAM)(x, y::Union{Array{Int64}, Array{UInt8}})
    atype = artype(ram)
    etype = eltype(ram)
    scores, baseline, logπ, ht = ram(x)
    ŷ = vec(map(i->i[1], argmax(Array(value(scores)), dims=1)))
    r = ŷ .== y; r = reshape(r, 1, :)
    R = zeros(Float32, size(baseline)...); R[end,:] = r
    R = convert(atype{etype}, R)
    R̂ = R .- value(baseline)
    loss_action = nll(scores, y)
    loss_baseline = sum(abs2, baseline .- R) / length(baseline)
    loss_reinforce = mean(sum(-logπ .* R̂, dims=1))
    return loss_action, loss_baseline, loss_reinforce, sum(R), length(R)
end
loss(x, ygold) = sum(ram(x,ygold)[1:3])

(ram::RAM)(d::Knet.Data) = mean(sum(ram(x,y)[1:3]) for (x,y) in d)


function validate(ram::RAM, data)
    losses = zeros(3)
    ncorrect = ninstances = 0
    for (x,y) in data
        ret = ram(x,y)
        for i = 1:3; losses[i] += ret[i]; end
        ncorrect += ret[4]
        ninstances += ret[5]
    end
    losses = losses / length(data)
    losses = [sum(losses), losses...]
    return losses, ncorrect / ninstances
end


function initstates(ram::RAM, B)
    etype, atype = eltype(ram), artype(ram)
    hsize = get_hsize(ram)
    lsize = get_lsize(ram)
    # l₀ = atype(2rand(etype, lsize, B) .- 1)
    l₀ = atype(zeros(etype, lsize, B))
    h₀ = atype(zeros(etype, hsize, B))
    return l₀, h₀
end


eltype(ram::RAM) = eltype(ram.softmax_layer.w.value)
artype(ram::RAM) = ifelse(
    typeof(ram.softmax_layer.w.value) <: KnetArray,
    KnetArray,
    Array)
# get_hsize(ram::RAM) = ram.controller.hiddenSize
get_hsize(ram::RAM) = size(ram.controller.h2h.w)[1]
get_lsize(ram::RAM) = size(ram.location_net.layer.w)[1]


mutable struct GlimpseNet
    fc1
    fc2
    fc3
    fc4
    retina
end


function GlimpseNet(
    loc_hidden, glimpse_hidden, patch_size, num_patches, scale, num_channels)
    retina = Retina(patch_size, num_patches, scale)
    fc1 = FullyConnected(
        patch_size*patch_size*num_patches*num_channels, glimpse_hidden)
    fc2 = FullyConnected(2, loc_hidden)
    fc3 = Linear(glimpse_hidden, glimpse_hidden+loc_hidden)
    fc4 = Linear(loc_hidden, glimpse_hidden+loc_hidden)
    return GlimpseNet(fc1, fc2, fc3, fc4, retina)
end


function (m::GlimpseNet)(x, loc_t_prev)
    phi = m.retina(x, loc_t_prev)
    loc_t_prev = mat(loc_t_prev)
    atype = typeof(m.fc1.w.value) <: KnetArray ? KnetArray : Array
    etype = eltype(m.fc1.w.value)
    phi = atype{etype}(phi)
    yphi = m.fc1(phi)
    yloc = m.fc2(loc_t_prev)
    y = m.fc3(yphi)
    l = m.fc4(yloc)
    return relu.(y+l)
end


mutable struct Retina
    patch_size
    num_patches
    glimpse_scale
end


# no need for Retina constructor!


# foveate
function (r::Retina)(x, l)
    phi = []
    sz = r.patch_size
    atype = typeof(l) <: KnetArray ? KnetArray : Array
    etype = eltype(l)

    for k = 1:r.num_patches
        push!(phi, r(x, l, sz))
        sz = Int(r.glimpse_scale * sz)
    end

    for i = 1:length(phi)
        k = div(size(phi[i])[1], r.patch_size)
        phi[i] = pool(phi[i]; window=k, mode=1)
    end

    phi2d = map(mat, phi)
    return atype{etype}(mat(vcat(phi2d...)))
end


# extract patch
# size(x) = H,W,C,B
# size(l) = 2,B
function (r::Retina)(x, l, k)
    H, W, C, B = size(x)
    coords = denormalize(H, l)

    patch_x = coords[1,:] .- div(k,2) .+ 1.
    patch_y = coords[2,:] .- div(k,2) .+ 1.

    patches = []
    atype = typeof(x) <: KnetArray ? KnetArray : Array
    etype = eltype(x)
    x2d = mat(x)
    for i = 1:B
        img = reshape(x2d[:, i:i], H, W, C, 1)
        img = Array(img)
        T = size(img)[1]

        sz = r.patch_size
        from_x, from_y = Int(round(patch_x[i])), Int(round(patch_y[i]))
        to_x, to_y = from_x+sz-1, from_y+sz-1

        if exceeds(from_x, to_x, from_y, to_y, T)
            pd = div(sz,2)+1
            padded = zeros(2pd+W, 2pd+H, C, 1)
            from_x += pd+1; to_x += pd+1; from_y += pd+1; to_y += pd+1
            padded[pd+1:pd+W, pd+1:pd+H, :, :] = img
            img = padded
        end
        push!(patches, img[from_x:to_x,from_y:to_y,:,:])
    end

    return atype{etype}(cat(patches...; dims=4))
end


function denormalize(T, coords)
    return convert(Array{Int}, floor.(0.5T * (coords .+ 1.0)))
end


function exceeds(from_x, to_x, from_y, to_y, T)
    return from_x < 1 || from_y < 1 || to_x > T || to_y > T
end


mutable struct LocationNet
    σ
    layer
end


function LocationNet(input_size::Int, output_size::Int, σ)
    layer = FullyConnected(input_size, output_size, tanh)
    return LocationNet(σ, layer)
end


function (m::LocationNet)(ht)
    μ = m.layer(value(ht))
    noise = randn!(similar(μ))
    lt = μ .+ noise .* m.σ
    return μ, tanh.(lt)
end


BaselineNet = FullyConnected
