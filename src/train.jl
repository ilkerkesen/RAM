include(Knet.dir("data", "mnist.jl"))
using ArgParse
using Dates

atype = gpu() >= 0 ? KnetArray{Float64} : Array{Float64}

function train(args="")
    s = ArgParseSettings()
    s.autofix_names = true
    s.description = "RAM implementation in Knet."

    @add_arg_table s begin
        ("--name"; help="experiment name"; default=nothing; required=true)
        ("--savedir"; help="checkpoint save directory"; default=SAVEDIR)
        ("--epochs"; help="# of epochs"; arg_type=Int; default=2000)
        ("--batchsize"; help="batch size"; arg_type=Int; default=256)
        ("--seed"; help="random seed"; arg_type=Int; default=-1)
        ("--atype"; help="array type"; default=string(atype))
        ("--lr"; help="learning rate"; arg_type=Float64; default=1e-4)
        ("--gclip"; help="gradient clip"; arg_type=Float64; default=0.1)
        ("--optim"; help="optimizer"; default="rmsprop")
        ("--loadfile"; help="load file"; default=nothing)
        ("--init"; help="init function"; default="randinit")
        ("--sigma"; help="std dev"; arg_type=Float64; default=.22)
        ("--patchsize"; help="patch size"; arg_type=Int; default=8)
        ("--num-patches"; help="# of patches"; arg_type=Int; default=1)
        ("--glimpse-scale"; help="glimpse scale"; arg_type=Float64; default=1.)
        ("--num-channels"; help="# of channels"; arg_type=Int; default=1)
        ("--num-glimpses"; help="# of glimpses"; arg_type=Int; default=6)
        ("--loc-hidden"; help="# of units in location net"; arg_type=Int;
         default=128)
        ("--glimpse-hidden"; help="# of units in glimpse net"; arg_type=Int;
         default=128)
        ("--hiddensize"; help="# of units in contoller rnn"; arg_type=Int;
         default=256)
        ("--rnntype"; help="type of controller (relu|lstm)"; default=":relu")
        ("--num-classes"; help="# of classes"; arg_type=Int; default=10)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    @info "Options parsed [$(now())]"
    println(o); flush(stdout)
    o[:atype] = eval(Meta.parse(o[:atype]))
    o[:seed] > 0 && Knet.seed!(o[:seed])
    optim = eval(Meta.parse(o[:optim]))
    o[:init] = eval(Meta.parse(o[:init]))
    o[:rnntype] = eval(Meta.parse(o[:rnntype]))
    o[:epochs] == -1 && return o

    xtrn, ytrn, xtst, ytst = mnist()

    ram = Net(
        o[:patchsize],
        o[:num_patches],
        o[:glimpse_scale],
        o[:num_channels],
        o[:loc_hidden],
        o[:glimpse_hidden],
        o[:sigma],
        o[:hiddensize],
        o[:num_classes],
        o[:num_glimpses]; rnnType=o[:rnntype], atype=o[:atype], init=o[:init])

    dtrn = minibatch(xtrn, ytrn, o[:batchsize]; xtype=o[:atype])
    dtst = minibatch(xtst, ytst, o[:batchsize]; xtype=o[:atype])

    bestmodel_path = abspath(joinpath(o[:savedir], o[:name]*"-best.jld2"))
    lastmodel_path = abspath(joinpath(o[:savedir], o[:name]*"-last.jld2"))
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
        Knet.@save lastmodel_path ram history o

        if tst_acc > bestacc
            bestacc = tst_acc
            Knet.@save bestmodel_path ram history o
        end
    end
end
