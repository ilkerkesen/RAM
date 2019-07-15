function exploration(ram::Net, raw_locations)
    sz = k = ram.glimpse_net.retina.patch_size
    locations = map(l->denormalize(28,l), raw_locations)
    locations = map(l->reshape(l, 2, :, 1), locations)
    locations = cat(locations..., dims=3)
    from_x = Int.(round.(locations[1,:,:] .- div(k,2) .+ 1))
    from_y = Int.(round.(locations[2,:,:] .- div(k,2) .+ 1))
    to_x, to_y = from_x .+ sz .- 1, from_y .+ sz .- 1
    B, T = size(from_x)
    from_x = reshape(from_x, B, T, 1)
    to_x = reshape(to_x, B, T, 1)
    from_y = reshape(from_y, B, T, 1)
    to_y = reshape(to_y, B, T, 1)
    points = cat(from_x, to_x, from_y, to_y, dims=3);
    grid = zeros(Int,28,28,B);
    points = min.(28, points);
    points = max.(1, points);
    for i = 1:B, t = 1:T
        fx,tx,fy,ty = points[i,t,:]
        grid[fy:ty,fx:tx,i] .+= 1
    end
    return grid
end
