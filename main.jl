using Revise

include("SheepWolfGrass.jl")

using Agents, Random
using Distributions
using GLMakie

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

stable_params = (;
    n_sheep = 50,
    n_wolves = 18,
    dims = (30, 30),
    Δenergy_sheep = 5,
    sheep_reproduce = 0.31,
    wolf_reproduce = 0.06,
    Δenergy_wolf = 30,
    sheep_gene_distribution = truncated(Normal(-.5, .2), -1, 1),
    wolf_gene_distribution = truncated(Normal(.5, .3), -1, 1),
    grass_gene_distribution = truncated(Normal(0, .3), -1, 1),
    wolf_gene_range = 0.1,
    wolf_mutation_rate = 0.05,
    grass_gene_range = 0.1,
    seed = 71758,
)

sheepwolfgrass = swg.initialize_model(;stable_params...)

#=
fig, ax, abmobs = abmplot(sheepwolfgrass;
    agent_step! = swg.sheepwolf_step!,
    model_step! = swg.grass_step!,
plotkwargs...)
fig
=#
sheep(a) = a isa swg.Sheep
wolf(a) = a isa swg.Wolf
count_grass(model) = count(model.fully_grown)
adata = [(sheep, count), (wolf, count)]
mdata = [count_grass]
adf, mdf = run!(sheepwolfgrass, swg.sheepwolf_step!, swg.grass_step!, 2000; adata, mdata)
plot_population_timeseries(adf, mdf)
