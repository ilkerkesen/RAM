using Knet
using Sloth
import Base: eltype
using Random
using Statistics


function randinit(d...; scale=0.1)
    A = rand(d...)
    A = 2A .* scale .- scale
end


function clip(x)
    x = max.(0, x)
    x = min.(1, x)
end


struct Chain; layers; end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)


mutable struct VanillaRNN
    i2h
    h2h
end


function VanillaRNN(xsize::Int, hsize::Int; atype=Sloth._atype, init=xavier)
    return VanillaRNN(Linear(input=xsize, output=hsize, atype=atype),
                      Linear(input=hsize, output=hsize, atype=atype))
end


(m::VanillaRNN)(x, ht) = relu.(m.i2h(x) + m.h2h(ht))


mutable struct Net
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


function Net(patch_size, num_patches, glimpse_scale, num_channels, loc_hidden,
             glimpse_hidden, σ, hidden_size, num_classes, num_glimpses;
             rnnType=:relu, atype=Sloth._atype, init=randinit)
    usegpu = atype <: KnetArray
    etype = eltype(atype)
    return Net(
        σ,
        num_glimpses,
        GlimpseNet(loc_hidden, glimpse_hidden, patch_size,
                   num_patches, glimpse_scale, num_channels;
                   atype=atype, init=init),
        RNN(hidden_size, hidden_size;
            rnnType=rnnType, usegpu=usegpu, dataType=etype, winit=init),
        # VanillaRNN(hidden_size, hidden_size; atype=atype, init=init),
        LocationNet(hidden_size, 2, σ; atype=atype, init=init),
        Linear(input=hidden_size, output=num_classes, atype=atype, init=init),
        BaselineNet(input=hidden_size, output=1, atype=atype, init=init),
        [],
        zeros(6)) # 1-3=>loss,4=>iter,5=>correct,6=># of samples
end


# one step
function (ram::Net)(x, ltprev; deterministic=false)
    gt = ram.glimpse_net(x, ltprev)
    ram.controller(gt)
    ht = ram.controller.h
    H, B = size(ht); ht = reshape(ht, H, B)
    μ, lt = ram.location_net(ht)
    bt = clip(ram.baseline_net(value(ht)))
    F = eltype(gt)
    σ = F(ram.σ)
    σ² = σ .^ 2
    logπ = -(abs.(lt - μ) .^ 2) / 2σ² .- log(σ) .- log(√(F(2)π))
    logπ = sum(logπ, dims=1)

    if deterministic
        lt = μ
    end
    return lt, bt, logπ
end


# all timesteps
function (ram::Net)(x; deterministic=false, centered=false)
    B = size(x)[end]
    lt, ht = initstates(ram, B; centered=centered)
    ram.controller.h = 0
    ram.controller.c = 0
    logπs, baselines, locations = [], [], Any[lt]
    for t = 1:ram.num_glimpses
        lt, bt, logπ = ram(x, lt; deterministic=deterministic)
        push!(locations, lt)
        push!(baselines, bt)
        push!(logπs, logπ)
    end
    baseline, logπ = vcat(baselines...), vcat(logπs...)
    ht = ram.controller.h
    H, B = size(ht); ht = reshape(ht, H, B)
    scores = ram.softmax_layer(mat(ht))
    return scores, baseline, logπ, locations
end


# loss
function (ram::Net)(x, y::Union{Array{Int64}, Array{UInt8}};
                    deterministic=false, centered=false)
    atype = artype(ram)
    etype = eltype(ram)
    scores, baseline, logπ, locations = ram(
        x; deterministic=deterministic, centered=centered)
    grid = exploration(ram, locations)
    rate = reshape(mean(grid .> 0, dims=(1,2)), 1, :);
    ŷ = vec(map(i->i[1], argmax(Array(value(scores)), dims=1)))
    r = ŷ .== y; r = reshape(r, 1, :)
    # r = r .+ rate;
    R = convert(atype{etype}, r)
    # R = zeros(Float32, size(baseline)...); R[end,:] = r
    # R = convert(atype{etype}, R)
    R̂ = R .- value(baseline)
    loss_action = nll(scores, y)
    loss_baseline = sum(abs2, baseline .- R) / length(baseline)
    loss_reinforce = mean(sum(-logπ .* R̂, dims=1))
    return loss_action, loss_baseline, loss_reinforce, sum(r), length(r)
end
loss(ram::Net, x, ygold) = sum(ram(x,ygold)[1:3])
loss(ram::Net, d::Knet.Data) = mean(sum(ram(x,y)[1:3]) for (x,y) in d)


function validate(ram::Net, data; deterministic=false, centered=false)
    losses = zeros(3)
    ncorrect = ninstances = 0
    for (x,y) in data
        ret = ram(x,y; deterministic=deterministic, centered=centered)
        for i = 1:3; losses[i] += ret[i]; end
        ncorrect += ret[4]
        ninstances += ret[5]
    end
    losses = losses / length(data)
    losses = [sum(losses), losses...]
    return losses, ncorrect / ninstances
end


function initstates(ram::Net, B; centered=false)
    etype, atype = eltype(ram), artype(ram)
    hsize = get_hsize(ram)
    lsize = get_lsize(ram)
    l₀ = atype{etype}(2rand(lsize, B) .- 1)
    if centered
        l₀ = atype(zeros(etype, lsize, B))
    end
    h₀ = atype{etype}(zeros(hsize, B))
    return l₀, h₀
end


eltype(ram::Net) = eltype(ram.softmax_layer.w.value)
artype(ram::Net) = ifelse(
    typeof(ram.softmax_layer.w.value) <: KnetArray,
    KnetArray,
    Array)
get_hsize(ram::Net) = get_hsize(ram.controller)
get_hsize(m::RNN) = m.hiddenSize
get_hsize(m::VanillaRNN) = size(m.h2h.w)[1]
get_lsize(ram::Net) = size(ram.location_net.layer.w)[1]


mutable struct GlimpseNet
    fc1
    fc2
    fc3
    fc4
    retina
end


function GlimpseNet(
    loc_hidden, glimpse_hidden, patch_size, num_patches, scale, num_channels;
    atype=Sloth._atype, init=randinit)
    retina = Retina(patch_size, num_patches, scale)
    fc1 = Dense(
        input=patch_size*patch_size*num_patches*num_channels,
        output=glimpse_hidden, atype=atype, init=init)
    fc2 = Dense(input=2, output=loc_hidden, atype=atype, init=init)
    fc3 = Linear(input=glimpse_hidden, output=glimpse_hidden+loc_hidden,
                 atype=atype, init=init)
    fc4 = Linear(input=loc_hidden, output=glimpse_hidden+loc_hidden,
                 atype=atype, init=init)
    return GlimpseNet(fc1, fc2, fc3, fc4, retina)
end


function (m::GlimpseNet)(x, lt)
    phi = m.retina(x, lt)
    lt = mat(lt)
    atype = typeof(m.fc1.w.value) <: KnetArray ? KnetArray : Array
    etype = eltype(m.fc1.w.value)
    phi = atype{etype}(phi)
    yphi = m.fc1(phi)
    yloc = m.fc2(lt)
    y = m.fc3(yphi)
    l = m.fc4(yloc)
    return relu.(y .* l)
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
# size(x) = W,H,C,B
# size(l) = 2,B
function (r::Retina)(x, l, k)
    W, H, C, B = size(x)
    coords = denormalize(H, l)

    patch_x = coords[1,:] .- div(k,2) .+ 1.
    patch_y = coords[2,:] .- div(k,2) .+ 1.

    patches = []
    atype = typeof(x) <: KnetArray ? KnetArray : Array
    etype = eltype(x)
    x2d = mat(x)
    for i = 1:B
        img = reshape(x2d[:, i:i], W, H, C, 1)
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

    glimpses = atype{etype}(cat(patches...; dims=4))
    return glimpses
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


function LocationNet(input_size::Int, output_size::Int, σ;
                     atype=Sloth._atype, init=randinit)
    layer = Dense(input=input_size, output=output_size, f=clip,
                  atype=atype, init=init)
    return LocationNet(σ, layer)
end


function (m::LocationNet)(ht)
    μ = clip(m.layer(value(ht)))
    F = eltype(μ)
    noise = F(m.σ) .* randn!(similar(μ))
    lt = μ + noise
    return μ, clip(lt)
end


BaselineNet = Linear
