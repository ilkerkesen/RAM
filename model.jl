using Knet
using Sloth


struct Chain; layers; end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)


mutable struct RAM
    σ
    glimpse_net
    controller
    location_net
    softmax_layer
    baseline_net
end


function RAM(patch_size, num_patches, glimpse_scale, num_channels, loc_hidden,
             glimpse_hidden, σ, hidden_size, num_classes)
    return RAM(
        σ,
        GlimpseNet(loc_hidden, glimpse_hidden, patch_size,
                   num_patches, glimpse_scale, num_channels),
        RNN(hidden_size, hidden_size),
        LocationNet(hidden_size, 2, σ),
        Linear(hidden_size, num_classes),
        BaselineNet(hidden_size, 1))
end


# just one step
function (ram::RAM)(x, l_prev, h_prev, last=false)
    gt = ram.glimpse_net(l_prev)
    ht = ram.controller(gt, h_prev)
    μ, lt = ram.location_net(ht)
    bt = baseline_net(ht)
    σ = ram.σ
    σ² = σ .^ 2
    logπ = -((lt - μ) ^ 2) / 2σ² - log(σ) - log(√2π)
    logp̂ = ifelse(!last, nothing, logp(ram.linear(mat(ht))))
    return ht, lt, bt, logπ, logp̂
end


# all timesteps
function (ram::RAM)(x)

end


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
        patch_size*num_patches*num_patches*num_channels, glimpse_hidden)
    fc2 = FullyConnected(2, loc_hidden)
    fc3 = Linear(glimpse_hidden, glimpse_hidden+loc_hidden)
    fc4 = Linear(loc_hidden, glimpse_hidden+loc_hidden)
    return GlimpseNet(fc1, fc2, fc3, fc4, retina)
end


function (m::GlimpseNet)(x, loc_t_prev)
    phi = m.retina(x, loc_t_prev)
    loc_t_prev = mat(loc_t_prev)
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

    for k = 1:r.num_patches
        push!(phi, r(x, l, sz))
        sz = Int(r.glimpse_scale * sz)
    end

    for i = 1:length(phi)
        k = div(size(phi[i])[1], r.patch_size)
        phi[i] = pool(phi[i]; window=k, mode=1)
    end

    return cat(4, phi...)
end


# extract patch
# size(x) = H,W,C,B
# size(l) = 2,B
function (r::Retina)(x, l, k)
    H, W, C, B = size(x)
    coords = denormalize(H, l)

    patch_x = coords[1,:] - div(k,2)
    patch_y = coords[2,:] - div(k,2)

    patches = []
    atype = Array{Float64}
    for i = 1:B
        img = x[:,:,:,i:i]
        T = size(img)[1]

        from_x, to_x = patch_x[i], patch_x[i] + size
        from_y, to_y = patch_y[i], patch_y[i] + size

        from_x, to_x = Int(round(from_x)), Int(round(to_x))
        from_y, to_y = Int(round(from_y)), Int(round(to_y))

        if exceeds(from_x, to_x, from_y, to_y, T)
            sz = size(img)[1]
            pd = div(sz,2)+1
            padded = atype(zeros(2pd+sz, 2pd+sz, C, 1))
            padded[pd+1:pd+sz, pd+1:pd+sz, :, :] = img
            img = padded
            from_x += pd; to_x += pd; from_y += pd; to_y = pd
        end

        # TODO: check to see whether x any are mixed
        push!(patches, img[from_x:to_x,from_y:to_y,:,:])
    end

    return cat(4, patches...)
end


function denormalize(T, coords)
    return 0.5 * T * (coords + 1.0)
end


function exceeds(from_x, to_x, from_y, to_y, T)
    return from_x < 0 || from_y < 0 || to_x > T || to_y > T
end


mutable struct LocationNet
    σ
    layer
end


function LocationNet(input_size::Int, output_size::Int, σ)
    layer = Linear(input_size, output_size)
    return LocationNet(σ, layer)
end


function (m::LocationNet)(ht)
    μ = m.layer(ht)
    noise = randn!(similar(μ))
    lt = μ .+ noise .* m.σ
    return μ, tanh.(lt)
end


BaselineNet = FullyConnected
