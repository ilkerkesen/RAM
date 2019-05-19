# includet("model.jl")
# include("model.jl")
include(Knet.dir("data", "mnist.jl"))
using Plots
using ArgParse
using Dates

atype = gpu() >= 0 ? KnetArray{Float64} : Array{Float64}

function main(args="")
    s = ArgParseSettings()
    s.autofix_names = true
    s.description = "RAM implementation in Knet."

    @add_arg_table s begin
        ("--epochs"; help="# of epochs"; arg_type=Int; default=2000)
        ("--batchsize"; help="batch size"; arg_type=Int; default=256)
        ("--seed"; help="random seed"; arg_type=Int; default=-1)
        ("--atype"; help="array type"; default=string(atype))
        ("--lr"; help="learning rate"; arg_type=Float64; default=1e-3)
        ("--gclip"; help="gradient clip"; arg_type=Float64; default=5.)
        ("--optim"; help="optimizer"; default="rmsprop")
        ("--savefile"; help="save file"; default=nothing)
        ("--loadfile"; help="load file"; default=nothing)
        ("--checkpoint"; help="checkpoint file"; default=nothing)
        ("--sigma"; help="std dev"; arg_type=Float64; default=.17)
        ("--patchsize"; help="patch size"; arg_type=Int; default=8)
        ("--num-patches"; help="# of patches"; arg_type=Int; default=1)
        ("--glimpse-scale"; help="glimpse scale"; arg_type=Float64; default=2.)
        ("--num-channels"; help="# of channels"; arg_type=Int; default=1)
        ("--num-glimpses"; help="# of glimpses"; arg_type=Int; default=6)
        ("--loc-hidden"; help="# of units in location net"; arg_type=Int;
         default=128)
        ("--glimpse-hidden"; help="# of units in glimpse net"; arg_type=Int;
         default=128)
        ("--hiddensize"; help="# of units in contoller rnn"; arg_type=Int;
         default=256)
        ("--num-classes"; help="# of classes"; arg_type=Int; default=10)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    @info "Options parsed [$(now())]"
    println(o); flush(stdout)
    o[:atype] = eval(Meta.parse(o[:atype]))
    o[:seed] > 0 && Knet.seed!(o[:seed])
    optim = eval(Meta.parse(o[:optim]))

    xtrn, ytrn, xtst, ytst = mnist()
    μ, σ = mean(xtrn), std(xtrn)
    xtrn = (xtrn .- μ) ./ σ
    xtst = (xtst .- μ) ./ σ;

    ram = RAM(
        o[:patchsize],
        o[:num_patches],
        o[:glimpse_scale],
        o[:num_channels],
        o[:loc_hidden],
        o[:glimpse_hidden],
        o[:sigma],
        o[:hiddensize],
        o[:num_classes],
        o[:num_glimpses]; atype=o[:atype])

    dtrn = minibatch(xtrn, ytrn, o[:batchsize]; xtype=o[:atype])
    dtst = minibatch(xtst, ytst, o[:batchsize]; xtype=o[:atype])

    history = []
    bestacc = 0.0
    o[:loadfile] != nothing && Knet.@load o[:loadfile] ram history
    loss(x, ygold) = sum(ram(x, ygold)[1:3])
    for epoch = 1:o[:epochs]
        progress!(optim(loss, dtrn; lr=o[:lr], gclip=o[:gclip]))
        trn_losses, trn_acc = validate(ram, dtrn)
        tst_losses, tst_acc = validate(ram, dtst)
        println(
            "epoch=$(1+length(history)) ",
            "trnloss=$(trn_losses), trnacc=$trn_acc, ",
            "tstloss=$(tst_losses), tstacc=$tst_acc")
        push!(history, ([trn_losses..., trn_acc, tst_losses..., tst_acc]))
        o[:checkpoint] != nothing && Knet.@save o[:checkpoint] ram history

        if tst_acc > bestacc
            bestacc = tst_acc
            o[:savefile] != nothing && Knet.@save o[:savefile] ram history o
        end
    end
end


PROGRAM_FILE == "train.jl" && main(ARGS)
