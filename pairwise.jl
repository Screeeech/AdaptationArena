using DataFrames
using Plots
using JLD2

# Load the data
df = load_object("data.jld2")

function pairwise_data(data)
    # Matrix of pairwise data
    col_names = names(data)

    # As you go down the matrix, i increases and we must change the y axis
    # So i->y and j->x
    return [DataFrame([data[:, j];;data[:, i]], :auto) for i in 1:size(data, 2)-1, j in 1:size(data, 2)-1]
end


function plot_pairwise(data)
    # Format the data for the scatter plots
    scores = data[:, end]
    col_names = names(data)

    data = pairwise_data(data)
    n = size(data, 1) # Dimension of the data

    # Create an nxn grid of scatter plots each using the data in data[i, j] and color the points according to the score
    plots = []
    for i in 1:n, j in 1:n
        push!(plots, Plots.plot(data[i, j][:, 1], data[i, j][:, 2], seriestype = :scatter, label = "", alpha=1, legend=false,
                markersize=2, markerstrokewidth=.1, xlims=(0,1), ylims=(0,1); 
                zcolor=scores, color = :heat, clims=(0, 2000)))
        if i == 1
            Plots.title!(plots[end], col_names[j], titlefontsize=12)
        end
        if j == 1
            Plots.ylabel!(plots[end], col_names[i], leftmargin=8Plots.mm)
        end
    end

    # Plot the grid of scatter plots
    # set x and y ticks to be in steps of 0.2 from 0 to 1
    # Add some space to the left of the plot for the y labels
    p = Plots.plot(plots..., layout = (n, n), size = (1600, 1200))
    Plots.xticks!(p, 0.2:0.2:1)
end

plot_pairwise(df)
