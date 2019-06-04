using Images
using Plots

function get_patch_locations(unnormalized_locations; W=28, H=28, patchsize=8)
    locations = denormalize(H, unnormalized_locations)
    px = locations[1,:] .- div(patchsize,2) .+ 1
    py = locations[2,:] .- div(patchsize,2) .+ 1
    from_x, from_y = Int.(round.(px)), Int.(round.(py))
    to_x, to_y = from_x .+ patchsize, from_y .+ patchsize
    from_x, from_y = max.(1, from_x), max.(1, from_y)
    to_x, to_y = min.(W, to_x), min.(H, to_y)
    return from_x[1], from_y[1], to_x[1], to_y[1]
end


function unrolled_view(images...)
    full = cat(images..., dims=1)
    colorview(RGB, permutedims(full, (3,2,1)))
end


function draw_glimpse(img, location, patchsize=8)
    W, H, C = size(img)[1:3]
    x = reshape(img, W, H, C)
    if C == 1
        x = repeat(x, 1, 1, 3)
    end
    x0, y0, x1, y1 = get_patch_locations(location; W=W, H=H,
                                         patchsize=patchsize)
    x[x0:x1,[y0,y1],1] .= 1.0;
    x[x0:x1,[y0,y1],2:3] .= 0.0;
    x[[x0,x1],y0:y1,1] .= 1.0;
    x[[x0,x1],y0:y1,2:3] .= 0.0;
    return x
end


function draw_glimpses(img, locations, patchsize=8)
    map(li->draw_glimpse(img,li,patchsize), locations)
end
