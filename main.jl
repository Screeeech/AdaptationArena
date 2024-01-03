using Revise

include("SheepWolfGrass.jl")

using Agents, Random
using Distributions
using GLMakie
using Plots
using ProgressBars

# gr()

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
        push!(grass_genes, [sheepwolfgrass.gene_center[p] for p in 1:dims[1]*dims[2]])
        run!(sheepwolfgrass, swg.sheepwolf_step!, swg.grass_step!, 1)
    end

    return pop_data, sheep_genes, wolf_genes, grass_genes
end

function generate_histogram(sheep_genes, wolf_genes, grass_genes, time_step)
    histogram(sheep_genes[time_step], bins=-1:.2:1, xlims=(-1, 1), ylims=(0, 100), alpha=0.5, label="Sheep")
    histogram!(wolf_genes[time_step], bins=-1:.1:1, alpha=0.8, xlims=(-1, 1), ylims=(0, 100), label="Wolves")
    Plots.plot!(legend=:topleft, xlabel="Gene Value", ylabel="Sheep/Wolf Count")

    histogram!(twinx(), grass_genes[time_step, :], bins=-1:.2:1,alpha=0.2, xlims=(-1, 1), ylims=(0,450), label="Grass", color=:green, 
                legend=:topright, ylabel="Grass Count")
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

function grid_search_pop(params, mutation_vals, reproduce_vals; n=10, T=2000)
    grid = zeros(length(mutation_vals), length(reproduce_vals))
    params_cp = Dict(pairs(params))
    for (i,vm) in ProgressBar(enumerate(mutation_vals)), (j,vr) in ProgressBar(enumerate(reproduce_vals))
        avg_extinction = 0
        for k in ProgressBar(1:n)
            params_cp[:wolf_mutation_rate] = vm
            params_cp[:wolf_reproduce] = vr
            params_cp[:seed] += k-1
            model = swg.initialize_model(;NamedTuple(params_cp)...)

            t = 1
            wolf_pop = zeros(T)
            wolf_pop[i] = -1
            while t <= T && wolf_pop[i] != 0
                all_agents = collect(allagents(model))
                wolf_pop[i] = count(wolf, all_agents)
                run!(model, swg.sheepwolf_step!, swg.grass_step!, 1)
                t += 1
            end
            avg_extinction += t/n
        end
        grid[i,j] = avg_extinction
    end
    return grid
end

# Here is all the actual interaction code
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
    n_sheep = 80,
    n_wolves = 30,
    dims = dims,
    Δenergy_sheep = 5,
    Δenergy_wolf = 30,
    sheep_gene_distribution = truncated(Normal(-1, .3), -1, 1),
    wolf_gene_distribution = truncated(Normal(0, .3), -1, 1),
    grass_gene_distribution = truncated(Normal(-1, .3), -1, 1),
    sheep_reproduce = 0.3,
    wolf_reproduce = 0.01,
    regrowth_time = 30,
    wolf_gene_range = 0.05,
    grass_gene_range = 0.1,
    sheep_mutation_rate = 0.3,
    wolf_mutation_rate = 0.01,
    grass_mutation_rate = 0.3,
    seed = 71759,
)
sheepwolfgrass = swg.initialize_model(;exp_params...)

#=
fig, ax, abmobs = abmplot(sheepwolfgrass;
    agent_step! = swg.sheepwolf_step!,
    model_step! = swg.grass_step!,
plotkwargs...)
fig
=#


#=
adata = [(sheep, count), (wolf, count)]
mdata = [count_grass]
adf, mdf = run!(sheepwolfgrass, swg.sheepwolf_step!, swg.grass_step!, 3000; adata, mdata)
plot_population_timeseries(adf, mdf)
=#

# pop_data, sheep_genes, wolf_genes, grass_genes = gather_data(sheepwolfgrass, 3000)

#=
anim = @animate for i in 1:length(sheep_genes)
    generate_histogram(sheep_genes, wolf_genes, grass_genes, i)
end
gif(anim, "histogram_animation.gif", fps = 50)
=#

#=
gene_mean, gene_err = gene_means((sheep_genes, wolf_genes, grass_genes))
Plots.plot(gene_mean[:, 3], ribbon=gene_err[:, 3], color=3, label="Grass", legend=:topright, xlabel="Time", ylabel="Gene Mean", lw=3)
Plots.plot!(gene_mean[:, 1], ribbon=gene_err[:, 1], color=1, label="Sheep", lw=3)
Plots.plot!(gene_mean[:, 2], ribbon=gene_err[:, 2], color=2, label="Wolves", lw=3)
=#

mutation_vals = 0:.05:.5
reproduce_vals = 0:.01:.3
grid = grid_search_pop(exp_params, mutation_vals, reproduce_vals; n=10, T=2000)
Plots.heatmap(reproduce_vals, mutation_vals, grid)