module sw

export initialize_model, sheepwolf_step!, grass_step!, Sheep, Wolf

using Agents, Random
using Distributions
using GLMakie

@agent Sheep GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
    gene::Float64
    mutation_rate::Float64
    hunt_range::Int
end

@agent Wolf GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
    gene_center::Float64
    gene_range::Float64
    mutation_rate::Float64
    hunt_range::Int
end

function initialize_model(;
        n_sheep = 100,
        n_wolves = 50,
        dims = (20, 20),
        regrowth_time = 30,
        Δenergy_sheep = 4,
        Δenergy_wolf = 20,
        sheep_reproduce = 0.04,
        wolf_reproduce = 0.05,
        wolf_gene_range = 0.2,
        sheep_mutation_rate = 0.1,
        wolf_mutation_rate = 0.1,
        grass_gene_range = 0.2,
        grass_mutation_rate = 0.1,
        sheep_gene_distribution = truncated(Normal(0, 0.3), -1, 1),
        wolf_gene_distribution = truncated(Normal(0, 0.3), -1, 1),
        grass_gene_distribution = truncated(Normal(0, 0.3), -1, 1),
        wolf_hunt_range = 5,
        sheep_hunt_range = 5,
        energy_transfer = 0.25,
        seed = 23182,
    )

    rng = MersenneTwister(seed)
    space = GridSpace(dims, periodic = true)
    # Model properties contain the grass as two arrays: whether it is fully grown
    # and the time to regrow. Also have static parameter `regrowth_time`.
    # Notice how the properties are a `NamedTuple` to ensure type stability.
    properties = (
        fully_grown = falses(dims),
        countdown = zeros(Int, dims),
        regrowth_time = ones(Int32, dims) * regrowth_time,
        gene_center = zeros(Float64, dims),
        gene_range = ones(Float64, dims) * grass_gene_range,
        mutation_rate = ones(Float64, dims) * grass_mutation_rate,
        energy_transfer = energy_transfer,
    )
    model = ABM(Union{Sheep, Wolf}, space;
        properties, rng, scheduler = Schedulers.randomly, warn = false
    )

    # Add agents
    # Initial distribution for sheep
    for _ in 1:n_sheep
        energy = rand(model.rng, 1:(Δenergy_sheep*2)) - 1
        sheep_gene = rand(model.rng, sheep_gene_distribution)
        add_agent!(Sheep, model, energy, sheep_reproduce, Δenergy_sheep, sheep_gene, sheep_mutation_rate, sheep_hunt_range)
    end

    # Initial distribution for wolves
    for _ in 1:n_wolves
        energy = rand(model.rng, 1:(Δenergy_wolf*2)) - 1 
        wolf_gene_center = rand(model.rng, wolf_gene_distribution)
        add_agent!(Wolf, model, energy, wolf_reproduce, Δenergy_wolf, wolf_gene_center, wolf_gene_range, wolf_mutation_rate, wolf_hunt_range)
    end

    # Initial distribution for grass
    for p in positions(model)
        fully_grown = rand(model.rng, Bool)
        countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1
        model.countdown[p...] = countdown
        model.fully_grown[p...] = fully_grown
        model.gene_center[p...] = rand(model.rng, grass_gene_distribution)
    end
    return model
end

function sheepwolf_step!(sheep::Sheep, model)
    nearby_grass = nearby_edible_grass(sheep, model, sheep.hunt_range)
    if !isempty(nearby_grass)
        target = sign.(get_direction(sheep.pos, chebyshev_nearest_position(sheep, nearby_grass), model))
        walk!(sheep, target, model)
    else
        randomwalk!(sheep, model)
    end
    sheep.energy -= 1
    if sheep.energy < 0
        remove_agent!(sheep, model)
        return
    end
    eat!(sheep, model)
    if rand(model.rng) ≤ sheep.reproduction_prob
        reproduce!(sheep, model)
    end
end

function sheepwolf_step!(wolf::Wolf, model)
    for i in 1:1
        nearby_sheep = nearby_edible_sheep(wolf, model, wolf.hunt_range)
        if !isempty(nearby_sheep)
            target = sign.(get_direction(wolf.pos, chebyshev_nearest_position(wolf, nearby_sheep), model))
            walk!(wolf, target, model; ifempty = false)
        else
            randomwalk!(wolf, model; ifempty = false)
        end
        wolf.energy -= 1
        if wolf.energy < 0
            remove_agent!(wolf, model)
            return
        end
        # If there is any sheep on this grid cell, it's dinner time!
        dinner = first_edible_sheep(wolf, model)
        (!isnothing(dinner) && wolf.energy < wolf.Δenergy*2) && eat!(wolf, dinner, model)
        if rand(model.rng) ≤ wolf.reproduction_prob
            reproduce!(wolf, model)
        end
    end
end

function nearby_edible_sheep(wolf::Wolf, model, r=5)
    nearby = collect(nearby_agents(wolf, model, r))
    return map(x -> x.pos, filter(agent -> agent isa Sheep && abs(agent.gene - wolf.gene_center) < wolf.gene_range, nearby))
end

function nearby_edible_grass(sheep::Sheep, model, r=5)
    nearby = collect(nearby_positions(sheep, model, r))
    return filter(pos -> model.fully_grown[pos...] && abs(model.gene_center[pos...] - sheep.gene) < model.gene_range[pos...], nearby)
end

function chebyshev_nearest_position(agent::A, positions) where {A <: AbstractAgent}
    p = agent.pos
    return reduce((x, y) -> max(abs.(x .- p)...) < max(abs.(y .- p)...) ? x : y, positions)
end

function first_edible_sheep(wolf::Wolf, model)
    pos = wolf.pos
    ids = ids_in_position(pos, model)
    j = findfirst(id -> model[id] isa Sheep && abs(model[id].gene - wolf.gene_center) < wolf.gene_range, ids)
    isnothing(j) ? nothing : model[ids[j]]::Sheep
end

function eat!(sheep::Sheep, model)
    if model.fully_grown[sheep.pos...] && abs(model.gene_center[sheep.pos...] - sheep.gene) < model.gene_range[sheep.pos...]
        sheep.energy += sheep.Δenergy
        model.fully_grown[sheep.pos...] = false
    end
    return
end

function eat!(wolf::Wolf, sheep::Sheep, model)
    remove_agent!(sheep, model)
    # wolf.energy += sheep.energy * model.energy_transfer
    wolf.energy = min(wolf.Δenergy*3, wolf.energy + wolf.Δenergy)
    return
end

function reproduce!(agent::Sheep, model)
    agent.energy /= 2
    id = nextid(model)
    mutated_gene = clamp(agent.gene + rand(model.rng, Normal(0, agent.mutation_rate)), -1, 1)
    offspring = Sheep(id, agent.pos, agent.energy, agent.reproduction_prob, agent.Δenergy, 
                        mutated_gene, agent.mutation_rate, agent.hunt_range)
    add_agent_pos!(offspring, model)
    return
end

function reproduce!(agent::Wolf, model)
    agent.energy /= 2
    id = nextid(model)
    mutated_gene_center = clamp(agent.gene_center + rand(model.rng, Normal(0, agent.mutation_rate)), -1, 1)
    offspring = Wolf(id, agent.pos, agent.energy, agent.reproduction_prob, agent.Δenergy, 
                        mutated_gene_center, agent.gene_range, agent.mutation_rate, agent.hunt_range)
    add_agent_pos!(offspring, model)
    return
end

function grass_step!(model)
    @inbounds for p in positions(model)
        if !(model.fully_grown[p...])
            if model.countdown[p...] ≤ 0
                # Fully grown grass
                model.fully_grown[p...] = true
                model.countdown[p...] = model.regrowth_time[p...]
            elseif model.countdown[p...] == model.regrowth_time[p...]
                # Cloning grass if there is any adjacent fully grown squares
                clone_choice = random_nearby_position(p, model, r=1; filter=pos->model.fully_grown[pos...])
                if !isnothing(clone_choice)
                    model.gene_center[p...] = clamp(model.gene_center[clone_choice...] + rand(model.rng, Normal(0, model.mutation_rate[clone_choice...])), -1, 1)
                    model.countdown[p...] -= 1
                end
            else
                model.countdown[p...] -= 1
            end
        end
    end
end

function sheep_gene_values(sheep)
    return [i.gene for i in sheep]
end

function wolf_gene_values(wolf)
    return [i.gene_center for i in wolf]
end

end
