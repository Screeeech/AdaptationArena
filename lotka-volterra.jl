using PlotlyJS
using Random
using Distributions

function modified_lotka_volterra_step(u, p, delta_t)
    x, y, z = u
    alpha, beta, gamma, delta, epsilon, zeta, eta, theta = p
    dx = delta_t * x * (alpha*(z-theta*x)-beta*y)
    dy = delta_t * y * (gamma*x-delta)
    dz = delta_t * z * (epsilon*(zeta-z)-eta*x)
    return [x+dx, y+dy, z+dz]
end

p = [1.5, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]
u_ensemble = [[1.0, 1.0, 1.0] + rand(Normal(0,.3),3) for i in 1:10]
# u_ensemble = vcat(u_ensemble, [[1.5, 3, 2] + rand(Normal(0,.3),3) for i in 1:10])
u_ensemble = vcat(u_ensemble, [[2, 1, 3] + rand(Normal(0,.3),3) for i in 1:10])

delta_t = 0.01
T = 200

plot3d()
trajectories = [zeros(3, Int(T/delta_t)) for i in 1:length(u_ensemble)]
for i in 1:length(u_ensemble)
    u0 = u_ensemble[i]
    for j in 1:size(trajectories[i],2)
        trajectories[i][:,j] = u0
        u0 = modified_lotka_volterra_step(u0, p, delta_t)
    end
    # plot each trajectory in a different color
    plot3d!(trajectories[i][1,:], trajectories[i][2,:], trajectories[i][3,:], color=i, label="")
end

xaxis!("sheep")
yaxis!("wolves")
zaxis!("grass")
