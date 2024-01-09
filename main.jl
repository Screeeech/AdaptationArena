using Revise

include("SheepWolfGrass.jl")

using Agents, Random
using Distributions
using GLMakie
using Plots
using ProgressMeter
using DataFrames
using JLD2

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
        all_agents = collect(allagents(model))
        pop_data[i, :] = [count(sheep, all_agents), count(wolf, all_agents), count_grass(model)]
        push!(sheep_genes, [a.gene for a in filter(sheep, all_agents)])
        push!(wolf_genes, [a.gene_center for a in filter(wolf, all_agents)])
        push!(grass_genes, [model.gene_center[p] for p in 1:dims[1]*dims[2]])
        run!(model, swg.sheepwolf_step!, swg.grass_step!, 1)
    end

    return pop_data, sheep_genes, wolf_genes, grass_genes
end

function gather_wolf_data(model, T)
    pop_data = zeros(T)
    
    for i in 1:T
        all_agents = collect(allagents(model))
        pop_data[i] = count(wolf, all_agents)
        
        if pop_data[i] == 0
            return pop_data[1:i]
        else
            run!(model, swg.sheepwolf_step!, swg.grass_step!, 1)
        end
    end

    return pop_data
end

function generate_histogram(sheep_genes, wolf_genes, grass_genes, time_step)
    histogram(sheep_genes[time_step], bins=-1:.2:1, xlims=(-1, 1), ylims=(0, 100), alpha=0.5, label="Sheep")
    histogram!(wolf_genes[time_step], bins=-1:.1:1, alpha=0.8, xlims=(-1, 1), ylims=(0, 100), label="Wolves")
    Plots.plot!(legend=:topleft, xlabel="Gene Value", ylabel="Sheep/Wolf Count")

    # histogram!(twinx(), grass_genes[time_step, :], bins=-1:.2:1,alpha=0.2, xlims=(-1, 1), ylims=(0,450), label="Grass", color=:green, 
    #            legend=:topright, ylabel="Grass Count")
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
    original_seed = params_cp[:seed]    

    p = Progress(length(mutation_vals)*length(reproduce_vals), 1)
    Threads.@threads for i in eachindex(mutation_vals)
        Threads.@threads for j in eachindex(reproduce_vals)
            vm = mutation_vals[i]
            vr = reproduce_vals[j]
            avg_extinction = 0
            Threads.@threads for k in 1:n
                params_cp_cp = deepcopy(params_cp)
                params_cp_cp[:wolf_mutation_rate] = vm
                params_cp_cp[:wolf_reproduce] = vr
                params_cp_cp[:seed] = original_seed + k - 1
                model = swg.initialize_model(;NamedTuple(params_cp_cp)...)

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
            next!(p)
        end
    end
    return grid
end


function particle_swarm_optimization_params(optimize_params, params; n_particles=10, n_iterations=100, T=1000, found_stable=nothing)
    # Initialize particles
    n_params = length(optimize_params)

    particles = zeros(n_particles, n_params)
    if isnothing(found_stable)
        particles = rand(Uniform(0.2, 1), n_particles, n_params)
    else
        stable_optimize = [found_stable[p] for p in optimize_params]'
        particles[1:end-1, :] = rand(Uniform(0.2, 1), n_particles-1, n_params)
        println(stable_optimize)
        particles[end, :] = stable_optimize
    end
    velocities = zeros(n_particles, n_params)
    best_positions = copy(particles)
    best_values = zeros(n_particles)
    global_best_position = zeros(n_params)
    global_best_value = 0
    original_seed = params[:seed]

    # Full data
    data = zeros(n_particles*(n_iterations+1), n_params + 1)

    # Initialize model
    model = swg.initialize_model(;NamedTuple(params)...)

    # Initialize best values
    for i in 1:n_particles
        params_cp = Dict(pairs(params))
        for j in 1:n_params
            params_cp[optimize_params[j]] = particles[i, j]
        end
        model = swg.initialize_model(;NamedTuple(params_cp)...)

        extinction_time = 0
        for j in 1:10
            params_cp[:seed] = original_seed + j - 1
            model = swg.initialize_model(;NamedTuple(params_cp)...)
            pop_data = gather_wolf_data(model, T)
            extinction_time += (isnothing(findfirst(pop_data .== 0)) ? T : findfirst(pop_data .== 0))/10
        end

        data[i, 1:n_params] = particles[i, :]
        data[i, n_params + 1] = extinction_time
        # Best value is the point when the wolf population goes extinct
        best_values[i] = extinction_time
    end

    # Iterate
    @showprogress for i in 1:n_iterations
        # Didn't add multithreading for asynchronous updating
        for j in 1:n_particles
            # Update velocity
            velocities[j, :] = 0.5*velocities[j, :] + 2*rand(Uniform(0, 1), n_params).*(best_positions[j, :] - particles[j, :]) + 2*rand(Uniform(0, 1), n_params).*(global_best_position - particles[j, :])
            # Update position
            particles[j, :] += velocities[j, :]
            particles[j, :] = clamp.(particles[j, :], 0.2, 1)
            params_cp = Dict(pairs(params))

            for k in 1:n_params
                params_cp[optimize_params[k]] = particles[j, k]
            end

            extinction_time = 0
            Threads.@threads for k in 1:10
                params_cp[:seed] = original_seed + k - 1
                model = swg.initialize_model(;NamedTuple(params_cp)...)
                pop_data = gather_wolf_data(model, T)
                extinction_time += (isnothing(findfirst(pop_data .== 0)) ? T : findfirst(pop_data .== 0))/10
            end

            data[i*n_particles + j, 1:n_params] = particles[j, :]
            data[i*n_particles + j, n_params + 1] = extinction_time

            if extinction_time > best_values[j]
                best_values[j] = extinction_time
                best_positions[j, :] = particles[j, :]
            end
            # Update global best position
            if extinction_time > global_best_value
                global_best_value = extinction_time
                global_best_position = particles[j, :]
            end
        end
    end

    params_cp = Dict(pairs(params))
    for i in 1:n_params
        params_cp[optimize_params[i]] = global_best_position[i]
    end
    return params_cp, data
end


# Here is all the actual interaction code
stable_params = (;
    n_sheep = 80,
    n_wolves = 10,
    dims = dims,
    Δenergy_sheep = 5,
    Δenergy_wolf = 30,
    sheep_gene_distribution = truncated(Normal(-1, .3), -1, 1),
    wolf_gene_distribution = truncated(Normal(0, .3), -1, 1),
    grass_gene_distribution = truncated(Normal(0, .1), -1, 1),
    sheep_reproduce = 0.5,
    wolf_reproduce = 0.2,
    regrowth_time = 30,
    wolf_gene_range = 0.1,
    grass_gene_range = 1,
    sheep_mutation_rate = 0.2,
    wolf_mutation_rate = .01,
    grass_mutation_rate = 0.3,
    energy_transfer = 0.3,
    seed = 71759,
)
exp_params = (;
    n_sheep = 80,
    n_wolves = 10,
    dims = dims,
    Δenergy_sheep = 10,
    Δenergy_wolf = 15,
    sheep_gene_distribution = truncated(Normal(-1, .3), -1, 1),
    wolf_gene_distribution = truncated(Normal(-.5, .3), -1, 1),
    grass_gene_distribution = truncated(Uniform(0, .3), -1, 1),
    grass_gene_range = 0.2,
    regrowth_time = 30,
    sheep_reproduce = 0.5,
    wolf_reproduce = 0.3,
    wolf_gene_range = 0.05,
    sheep_mutation_rate = .3,
    wolf_mutation_rate = .1,
    seed = 71759,
)

#=
sheepwolfgrass = swg.initialize_model(;exp_params...)
fig, ax, abmobs = abmplot(sheepwolfgrass;
    agent_step! = swg.sheepwolf_step!,
    model_step! = swg.grass_step!,
plotkwargs...)
fig
=#

#=
# Symbols to optimize
optimize = [:sheep_reproduce, :wolf_reproduce, :sheep_mutation_rate, :wolf_mutation_rate, 
            :grass_mutation_rate, :grass_gene_range, :wolf_gene_range]
optimized_params, scatter_data = particle_swarm_optimization_params(optimize, exp_params; n_particles=20, n_iterations=100, T=2000, found_stable=stable_params)
scatter_data = DataFrame(scatter_data, [optimize..., :extinction_time])
# @save "constrained_pso.jld2" scatter_data
=#

sheepwolfgrass = swg.initialize_model(;exp_params...)
adata = [(sheep, count), (wolf, count)]
mdata = [count_grass]
adf, mdf = run!(sheepwolfgrass, swg.sheepwolf_step!, swg.grass_step!, 2000; adata, mdata)
display(plot_population_timeseries(adf, mdf))

# sheepwolfgrass = swg.initialize_model(;exp_params...)
# pop_data, sheep_genes, wolf_genes, grass_genes = gather_data(sheepwolfgrass, 2000)

#=
anim = @animate for i in 1:length(sheep_genes)
    generate_histogram(sheep_genes, wolf_genes, grass_genes, i)
end
gif(anim, "histogram_animation.gif", fps = 50)
=#

#=
gene_mean, gene_err = gene_means((sheep_genes, wolf_genes, grass_genes))
# Plots.plot(gene_mean[:, 3], ribbon=gene_err[:, 3], color=3, label="Grass", legend=:topright, xlabel="Time", ylabel="Gene Mean", lw=3)
p = Plots.plot(gene_mean[:, 1], ribbon=gene_err[:, 1], color=1, label="Sheep", lw=3)
Plots.plot!(gene_mean[:, 2], ribbon=gene_err[:, 2], color=2, label="Wolves", lw=3)
display(p)
=#

#=
mutation_vals = LinRange(0, 1, 50)
reproduce_vals = LinRange(0, 1, 50)
grid = grid_search_pop(exp_params, mutation_vals, reproduce_vals; n=10, T=2000)
Plots.heatmap(reproduce_vals, mutation_vals, grid, xlabel="Reproduction Rate", ylabel="Mutation Rate", title="Extinction Time")
=#
