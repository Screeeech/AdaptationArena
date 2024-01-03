using Revise

include("SheepWolfGrass.jl")

using Agents, Random
using Distributions
using GLMakie
using Plots

gr()

function plot_population_timeseries(adf, mdf)
    figure = Figure(resolution = (600, 400))
    ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population")
    sheepl = lines!(ax, adf.step, adf.count_sheep, color = :cornsilk4)
    wolfl = lines!(ax, adf.step, adf.count_wolf, color = RGBAf(0.2, 0.2, 0.3))
    grassl = lines!(ax, mdf.step, mdf.count_grass, color = :green)
    figure[1, 2] = Legend(figure, [sheepl, wolfl, grassl], ["Sheep", "Wolves", "Grass"])
    figure
end

offset(a) = a isa swg.Sheep ? (-0.1, -0.1*rand()) : (+0.1, +0.1*rand())
ashape(a) = a isa swg.Sheep ? :circle : :utriangle
acolor(a) = a isa swg.Sheep ? RGBAf(1.0, 1.0, 1.0, 0.8) : RGBAf(0.2, 0.2, 0.3, 0.8)

grasscolor(model) = model.countdown ./ model.regrowth_time
heatkwargs = (colormap = [:brown, :green], colorrange = (0, 1))

plotkwargs = (;
    ac = acolor,
    as = 25,
    am = ashape,
    offset,
    scatterkwargs = (strokewidth = 1.0, strokecolor = :black),
    heatarray = grasscolor,
    heatkwargs = heatkwargs,
)

dims = (30, 30)
stable_params = (;
    n_sheep = 100,
    n_wolves = 20,
    dims = dims,
    regrowth_time = 30,
    Δenergy_sheep = 5,
    sheep_reproduce = 0.30,
    wolf_reproduce = 0.05,
    Δenergy_wolf = 30,
    sheep_gene_distribution = truncated(Normal(-.5, .3), -1, 1),
    wolf_gene_distribution = truncated(Normal(.5, .2), -1, 1),
    grass_gene_distribution = truncated(Normal(0, .3), -1, 1),
    wolf_gene_range = 0.12,
    sheep_mutation_rate = 0.1,
    wolf_mutation_rate = 0.03,
    grass_gene_range = 0.1,
    seed = 71758,
)
exp_params = (;
    n_sheep = 100,
    n_wolves = 20,
    dims = dims,
    regrowth_time = 30,
    Δenergy_sheep = 5,
    sheep_reproduce = 0.30,
    wolf_reproduce = 0.05,
    Δenergy_wolf = 30,
    sheep_gene_distribution = truncated(Normal(-.5, .3), -1, 1),
    wolf_gene_distribution = truncated(Normal(.5, .2), -1, 1),
    grass_gene_distribution = truncated(Normal(0, .3), -1, 1),
    wolf_gene_range = 0.12,
    sheep_mutation_rate = 0.1,
    wolf_mutation_rate = 0.03,
    grass_gene_range = 0.1,
    seed = 71758,
)


sheepwolfgrass = swg.initialize_model(;stable_params...)


sheep(a) = a isa swg.Sheep
wolf(a) = a isa swg.Wolf
count_grass(model) = count(model.fully_grown)

function gather_data(model, T)
    pop_data = zeros(T, 3)
    sheep_genes = []
    wolf_genes = []
    grass_genes = []

    for i in 1:T
        all_agents = collect(allagents(sheepwolfgrass))
        pop_data[i, :] = [count(sheep, all_agents), count(wolf, all_agents), count_grass(sheepwolfgrass)]
        push!(sheep_genes, [a.gene for a in filter(sheep, all_agents)])
        push!(wolf_genes, [a.gene_center for a in filter(wolf, all_agents)])
        push!(grass_genes, [sheepwolfgrass.gene_center[p...] for p in positions(sheepwolfgrass)])
        run!(sheepwolfgrass, swg.sheepwolf_step!, swg.grass_step!, 1)
    end

    return pop_data, sheep_genes, wolf_genes, grass_genes
end

function generate_histogram(time_step)
    histogram(sheep_genes[time_step], bins=-1:.2:1, xlims=(-1, 1), ylims=(0, 100), alpha=0.5, label="Sheep")
    histogram!(wolf_genes[time_step], bins=-1:.1:1, alpha=0.5, xlims=(-1, 1), ylims=(0, 100), label="Wolves")

    twinx()
    histogram!(grass_genes[time_step, :], bins=-1:.2:1,alpha=0.2, xlims=(-1, 1), ylims=(0,250), label="Grass", color=:green)
    Plots.yticks!(0:50:250)
    Plots.plot!(legend=:topright)
end

function gene_means(genes)
    means = zeros(size(genes[1])[1], length(genes))
    yerr = zeros(size(genes[1])[1], length(genes))
    for i in 1:size(means)[2]
        means[:, i] = mean.(genes[i])
        yerr[:, i] = std.(genes[i])
    end
    # Plot means with error shown as ribbons
    return means, yerr
end

#=
fig, ax, abmobs = abmplot(sheepwolfgrass;
    agent_step! = swg.sheepwolf_step!,
    model_step! = swg.grass_step!,
plotkwargs...)
fig
=#

#=
adata = [(sheep, count), (wolf, count)]
adf, mdf = run!(sheepwolfgrass, swg.sheepwolf_step!, swg.grass_step!, 2000; adata, mdata)
plot_population_timeseries(adf, mdf)
=#

#=
anim = @animate for i in 1:length(sheep_genes)
    generate_histogram(i)
end
gif(anim, "histogram_animation.gif", fps = 50)
=#

#=
pop_data, sheep_genes, wolf_genes, grass_genes = gather_data(sheepwolfgrass, 2000)
gene_mean, gene_err = gene_means((sheep_genes, wolf_genes, grass_genes))
Plots.plot(gene_mean[:, 3], ribbon=gene_err[:, 3], color=3, label="Grass", legend=:topright, xlabel="Time", ylabel="Gene Mean")
Plots.plot!(gene_mean[:, 1], ribbon=gene_err[:, 1], color=1, label="Sheep")
Plots.plot!(gene_mean[:, 2], ribbon=gene_err[:, 2], color=2, label="Wolves")
=#
