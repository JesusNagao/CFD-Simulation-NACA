module Solver

export solver

function solver(u::Matrix{Float64}, v::Matrix{Float64}, NX::Int, NY::Int, inds::Vector{CartesianIndex{2}}, dt::Float64, dx::Float64, dy::Float64, rho::Float64, p::Matrix{Float64}, nu::Float64, U_inlet::Float64=1.0)

    # Fractional step method for incompressible Navier-Stokes
    # Step 1: Predictor step - update velocity without pressure (forward Euler)
    u_pred = copy(u)
    v_pred = copy(v)

    for j in 2:NY, i in 2:NX
        if CartesianIndex(i, j) in inds
            u_pred[i, j] = 0.0
            v_pred[i, j] = 0.0
            continue
        end

        # Advection terms (central differencing)
        u_conv_x = u[i, j] * (u[i+1, j] - u[i-1, j]) / (2*dx)
        u_conv_y = v[i, j] * (u[i, j+1] - u[i, j-1]) / (2*dy)
        v_conv_x = u[i, j] * (v[i+1, j] - v[i-1, j]) / (2*dx)
        v_conv_y = v[i, j] * (v[i, j+1] - v[i, j-1]) / (2*dy)

        # Diffusion terms
        u_diff = nu * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx^2 +
                      (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy^2)
        v_diff = nu * ((v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dx^2 +
                      (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dy^2)

        # Predictor velocity (no pressure gradient)
        u_pred[i, j] = u[i, j] + dt * (-u_conv_x - u_conv_y + u_diff)
        v_pred[i, j] = v[i, j] + dt * (-v_conv_x - v_conv_y + v_diff)
    end

    # Step 2: Solve pressure Poisson equation ∇²p = ρ/Δt * ∇·u_pred
    p_new = copy(p)
    max_iter = 100  # Increased iterations
    tolerance = 1e-8  # Stricter tolerance

    for iter in 1:max_iter
        p_old_iter = copy(p_new)
        max_diff = 0.0

        for j in 2:NY-1, i in 2:NX-1
            if CartesianIndex(i, j) in inds
                continue
            end

            # Compute divergence of predicted velocity
            div_u = (u_pred[i+1, j] - u_pred[i-1, j]) / (2*dx) +
                   (v_pred[i, j+1] - v_pred[i, j-1]) / (2*dy)

            # Right-hand side: ρ/Δt * ∇·u_pred
            rhs = rho / dt * div_u

            # Jacobi update for Poisson equation ∇²p = rhs
            p_new[i, j] = ((p_old_iter[i+1, j] + p_old_iter[i-1, j]) * dy^2 +
                          (p_old_iter[i, j+1] + p_old_iter[i, j-1]) * dx^2 -
                          rhs * dx^2 * dy^2) / (2 * (dx^2 + dy^2))

            max_diff = max(max_diff, abs(p_new[i, j] - p_old_iter[i, j]))
        end

        if max_diff < tolerance
            break
        end
    end

    # Step 3: Corrector step - project velocity to be divergence-free
    for j in 2:NY, i in 2:NX
        if CartesianIndex(i, j) in inds
            u[i, j] = 0.0
            v[i, j] = 0.0
            continue
        end

        # Pressure gradient correction
        dp_dx = (p_new[i+1, j] - p_new[i-1, j]) / (2*dx)
        dp_dy = (p_new[i, j+1] - p_new[i, j-1]) / (2*dy)

        u[i, j] = u_pred[i, j] - dt / rho * dp_dx
        v[i, j] = v_pred[i, j] - dt / rho * dp_dy
    end

    # Update pressure field
    p .= p_new

    # Apply boundary conditions
    # Left inlet: fixed horizontal inflow U_inlet, no vertical inflow
    u[1, :] .= U_inlet
    v[1, :] .= 0.0

    # Top/bottom: free-slip walls (no penetration, free tangential flow)
    v[:, 1] .= 0.0  # No vertical flow through top
    v[:, end] .= 0.0  # No vertical flow through bottom
    u[:, 1] .= u[:, 2]  # Free tangential flow at top
    u[:, end] .= u[:, end-1]  # Free tangential flow at bottom

    # Right outlet: zero-gradient outflow
    u[end, :] .= u[end-1, :]
    v[end, :] .= v[end-1, :]
    p[end, :] .= p[end-1, :]

    return u, v, p
end

end #module