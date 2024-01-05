using DataFrames
using Plots
using JLD2

# Load the data
df = load("data.jld2", "scatter_data")

function pairwise_data(data)
    # Matrix of pairwise data
    return [DataFrame([data[:, i];;data[:, j]], :auto) for i in 1:size(data, 2)-1, j in 1:size(data, 2)-1]
end


function plot_pairwise(data)
    # Format the data for the scatter plots
    scores = data[:, end]
    data = pairwise_data(data)
    n = size(data, 1) # Dimension of the data

    # Create an nxn grid of scatter plots each using the data in data[i, j] and color the points according to the score
    plots = []
    for i in 1:n, j in 1:n
        if i == j
            push!(plots, Plots.plot(data[i, j][:, 1], data[i, j][:, 2], seriestype = :scatter, label = "", alpha=0.5, legend=false; 
                    zcolor=scores, color = :heat, clims=(0, 2000)))
        else
            push!(plots, Plots.plot!(data[i, j][:, 1], data[i, j][:, 2], seriestype = :scatter, label = "", alpha=0.5, legend=false; 
                    zcolor=scores, color = :heat, clims=(0, 2000)))
        end
    end
    p = Plots.plot(plots..., layout = (n, n), size = (1000, 1000))
end

plot_pairwise(df)
