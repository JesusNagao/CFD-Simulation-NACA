module Solver

export solver

"""
Solver module for a simple 2D incompressible Navier-Stokes flow solver.
"""

"""
    solver(u, v, NX, NY, obstacle, dt, dx, dy, rho, p, nu, U_inlet=1.0)

Advance the velocity and pressure fields one timestep using a fractional-step projection.
All heavy arrays are `AbstractMatrix` so the same code runs on CPU arrays or GPU arrays
(CuArray, MtlArray, etc.) without modification.

Arguments:
- `u`, `v`: velocity components on a collocated grid, size `(NX+1, NY+1)`.
- `NX`, `NY`: number of interior grid points in x and y.
- `obstacle`: boolean mask of the same size as `u`; `true` marks solid cells.
- `dt`: timestep size.
- `dx`, `dy`: mesh spacing in x and y.
- `rho`: fluid density.
- `p`: pressure field.
- `nu`: kinematic viscosity.
- `U_inlet`: fixed horizontal velocity at the left boundary.

Returns updated `(u, v, p)`.
"""
function solver(u::AbstractMatrix, v::AbstractMatrix, NX::Int, NY::Int,
                obstacle::AbstractMatrix{Bool},
                dt::Real, dx::Real, dy::Real, rho::Real,
                p::AbstractMatrix, nu::Real, U_inlet::Real=1.0)

    T     = eltype(u)
    dt    = T(dt);   dx   = T(dx);   dy  = T(dy)
    nu    = T(nu);   rho  = T(rho);  U_inlet = T(U_inlet)
    dx2   = dx^2;    dy2  = dy^2
    denom = 2 * (dx2 + dy2)

    # Index ranges shared by predictor and corrector
    i1 = 2:NX;    i0 = 1:NX-1;  i2 = 3:NX+1
    j1 = 2:NY;    j0 = 1:NY-1;  j2 = 3:NY+1

    # Index ranges for the Poisson interior (one cell smaller on each side)
    ip = 2:NX-1;  ip0 = 1:NX-2;  ip2 = 3:NX
    jp = 2:NY-1;  jp0 = 1:NY-2;  jp2 = 3:NY

    # Float masks: 1.0 in fluid, 0.0 in obstacle (computed once per call)
    free   = T.(.!(@view obstacle[i1, j1]))
    free_p = T.(.!(@view obstacle[ip, jp]))

    # ── Step 1: Predictor (upwind advection + central diffusion) ─────────────
    u_c = @view u[i1, j1];   v_c = @view v[i1, j1]
    u_e = @view u[i2, j1];   u_w = @view u[i0, j1]
    u_n = @view u[i1, j2];   u_s = @view u[i1, j0]
    v_e = @view v[i2, j1];   v_w = @view v[i0, j1]
    v_n = @view v[i1, j2];   v_s = @view v[i1, j0]

    # Upwind: pick backward/forward stencil based on local flow direction.
    u_adv_x = (max.(u_c, T(0)) .* (u_c .- u_w) .+ min.(u_c, T(0)) .* (u_e .- u_c)) ./ dx
    u_adv_y = (max.(v_c, T(0)) .* (u_c .- u_s) .+ min.(v_c, T(0)) .* (u_n .- u_c)) ./ dy
    v_adv_x = (max.(u_c, T(0)) .* (v_c .- v_w) .+ min.(u_c, T(0)) .* (v_e .- v_c)) ./ dx
    v_adv_y = (max.(v_c, T(0)) .* (v_c .- v_s) .+ min.(v_c, T(0)) .* (v_n .- v_c)) ./ dy

    u_diff = nu .* ((u_e .- 2 .* u_c .+ u_w) ./ dx2 .+ (u_n .- 2 .* u_c .+ u_s) ./ dy2)
    v_diff = nu .* ((v_e .- 2 .* v_c .+ v_w) ./ dx2 .+ (v_n .- 2 .* v_c .+ v_s) ./ dy2)

    u_pred = copy(u)
    v_pred = copy(v)
    u_pred[i1, j1] .= free .* (u_c .+ dt .* (.-u_adv_x .- u_adv_y .+ u_diff))
    v_pred[i1, j1] .= free .* (v_c .+ dt .* (.-v_adv_x .- v_adv_y .+ v_diff))

    # BCs on predicted velocity (must be applied before the Poisson solve
    # so the RHS divergence is computed from a BC-consistent field).
    u_pred[1, :]   .= U_inlet;   v_pred[1, :]   .= 0
    v_pred[:, 1]   .= 0;         v_pred[:, end] .= 0
    u_pred[:, 1]   .= @view u_pred[:, 2]
    u_pred[:, end] .= @view u_pred[:, end-1]
    u_pred[end, :] .= @view u_pred[end-1, :]
    v_pred[end, :] .= @view v_pred[end-1, :]

    # ── Step 2: Pressure Poisson (Red-Black SOR) ─────────────────────────────
    # Forward-difference divergence is consistent with the backward-difference
    # pressure gradient applied in the correction step below.
    div_u = ((@view u_pred[ip2, jp]) .- (@view u_pred[ip,  jp])) ./ dx .+
            ((@view v_pred[ip,  jp2]) .- (@view v_pred[ip,  jp])) ./ dy
    rhs   = (rho / dt) .* div_u .* free_p   # zero inside obstacle

    # Checkerboard mask: true on "red" cells (i+j even), false on "black" cells.
    # Red cells only border black cells, so each colour can be updated in parallel
    # while still achieving Gauss-Seidel convergence rates (unlike plain Jacobi).
    rb = similar(p, Bool, length(ip), length(jp))
    rb .= Bool[((i + j) % 2 == 0) for i in ip, j in jp]
    ω  = T(1.7)
    α  = 1 - ω

    p_new = copy(p)
    p_new[end, :] .= 0   # Dirichlet: anchor pressure level at the outlet

    for iter in 1:500
        # Neumann BCs: zero normal gradient at inlet and walls; Dirichlet at outlet.
        p_new[1, :]   .= @view p_new[2, :]
        p_new[:, 1]   .= @view p_new[:, 2]
        p_new[:, end] .= @view p_new[:, end-1]
        p_new[end, :] .= 0

        # Red half-sweep: update all red cells simultaneously.
        p_c  = @view p_new[ip, jp]
        p_gs = (((@view p_new[ip2, jp]) .+ (@view p_new[ip0, jp])) .* dy2 .+
                ((@view p_new[ip, jp2]) .+ (@view p_new[ip, jp0])) .* dx2 .-
                rhs .* dx2 .* dy2) ./ denom
        p_new[ip, jp] .= ifelse.(rb, ω .* p_gs .+ α .* p_c, p_c)

        # Black half-sweep: update all black cells using the freshly updated red neighbours.
        p_c  = @view p_new[ip, jp]
        p_gs = (((@view p_new[ip2, jp]) .+ (@view p_new[ip0, jp])) .* dy2 .+
                ((@view p_new[ip, jp2]) .+ (@view p_new[ip, jp0])) .* dx2 .-
                rhs .* dx2 .* dy2) ./ denom
        p_new[ip, jp] .= ifelse.(rb, p_c, ω .* p_gs .+ α .* p_c)

        # Residual-based convergence check every 25 iterations.
        # Using the residual avoids saving a full copy of p_new for comparison.
        if iter % 25 == 0
            res = abs.(
                ((@view p_new[ip2, jp]) .+ (@view p_new[ip0, jp])) ./ dx2 .+
                ((@view p_new[ip, jp2]) .+ (@view p_new[ip, jp0])) ./ dy2 .-
                (2/dx2 + 2/dy2) .* (@view p_new[ip, jp]) .- rhs
            ) .* free_p
            maximum(res) < T(1e-5) && break
        end
    end

    # ── Step 3: Velocity correction ──────────────────────────────────────────
    # Backward-difference gradient pairs with the forward-difference divergence
    # used in the Poisson RHS, making the corrected field discretely divergence-free.
    dp_dx = ((@view p_new[i1, j1]) .- (@view p_new[i0, j1])) ./ dx
    dp_dy = ((@view p_new[i1, j1]) .- (@view p_new[i1, j0])) ./ dy

    u[i1, j1] .= free .* ((@view u_pred[i1, j1]) .- (dt / rho) .* dp_dx)
    v[i1, j1] .= free .* ((@view v_pred[i1, j1]) .- (dt / rho) .* dp_dy)

    copyto!(p, p_new)

    # BCs on corrected velocity
    u[1, :]   .= U_inlet;   v[1, :]   .= 0
    v[:, 1]   .= 0;         v[:, end] .= 0
    u[:, 1]   .= @view u[:, 2]
    u[:, end] .= @view u[:, end-1]
    u[end, :] .= @view u[end-1, :]
    v[end, :] .= @view v[end-1, :]
    p[end, :] .= 0

    return u, v, p
end

end #module
