using Knet
using Sloth
import Base: eltype
using Random


struct Chain; layers; end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)


mutable struct RAM
    σ
    num_glimpses
    glimpse_net
    controller
    location_net
    softmax_layer
    baseline_net
end


function RAM(patch_size, num_patches, glimpse_scale, num_channels, loc_hidden,
             glimpse_hidden, σ, hidden_size, num_classes, num_glimpses)
    return RAM(
        σ,
        num_glimpses,
        GlimpseNet(loc_hidden, glimpse_hidden, patch_size,
                   num_patches, glimpse_scale, num_channels),
        RNN(hidden_size, hidden_size; rnnType=:relu),
        LocationNet(hidden_size, 2, σ),
        Linear(hidden_size, num_classes),
        BaselineNet(hidden_size, 1))
end


# just one step
function (ram::RAM)(x, l_prev, last=false)
    gt = ram.glimpse_net(x, l_prev)
    gt = reshape(gt, size(gt)..., :)
    ht = ram.controller(gt)
    H, B = size(ht); ht = reshape(ht, H, B)
    μ, lt = ram.location_net(ht)
    bt = ram.baseline_net(ht)
    σ = ram.σ
    σ² = σ .^ 2
    logπ = -(abs.(lt - μ) .^ 2) / 2σ² .- log(σ) .- log(√2π)
    logπ = sum(logπ, dims=2)
    # logp̂ = ifelse(!last, nothing, logp(ram.linear(mat(ht))))
    return ht, lt, bt, logπ
end


# # all timesteps
# function (ram::RAM)(x)
#     B = size(x)[end]
#     lt, ht = initstates(ram, B)

#     xs = []
#     locations = []
#     logπs = []
#     baselines = []
# end


function initstates(ram::RAM, B)
    etype, atype = eltype(ram), artype(ram)
    hsize = get_hsize(ram)
    lsize = get_lsize(ram)
    l₀ = atype(2rand(etype, lsize, B) .- 1)
    h₀ = atype(zeros(etype, hsize, B))
    return l₀, h₀
end


eltype(ram::RAM) = eltype(ram.controller.w.value)
artype(ram::RAM) = ifelse(
    typeof(ram.controller.w.value) <: KnetArray,
    KnetArray,
    Array)
get_hsize(ram::RAM) = ram.controller.hiddenSize
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

    phi2d = map(mat, phi)
    return mat(vcat(phi2d...))
end


# extract patch
# size(x) = H,W,C,B
# size(l) = 2,B
function (r::Retina)(x, l, k)
    H, W, C, B = size(x)
    coords = denormalize(H, l)

    patch_x = coords[1,:] .- div(k,2)
    patch_y = coords[2,:] .- div(k,2)

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
            # padded[from_x:to_x, from_y:to_y, :, :] = Array(img)
            padded[pd+1:pd+W, pd+1:pd+H, :, :] = img
            img = padded
        end

        # TODO: check to see whether x any are mixed
        push!(patches, img[from_x:to_x,from_y:to_y,:,:])
    end

    return atype{etype}(cat(patches...; dims=4))
end


function denormalize(T, coords)
    return convert(Array{Int}, round.(0.5T * (coords .+ 1.0)))
end


function exceeds(from_x, to_x, from_y, to_y, T)
    return from_x < 1 || from_y < 1 || to_x > T || to_y > T
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
