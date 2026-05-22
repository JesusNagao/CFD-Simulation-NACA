module Forces

export compute_forces, compute_cl_cd

"""
    compute_forces(u, v, p, obstacle, dx, dy, rho, nu) → (Fx, Fy)

Integrate pressure and viscous stresses over every immersed-boundary face
to get the net aerodynamic force on the obstacle.

For each obstacle cell that shares a face with a fluid cell the traction
  t = σ · n_solid = (-p I + 2μ E) · n_solid
is summed, where n_solid is the outward unit normal from the obstacle into
the fluid.

With freestream in the +x direction:
  Fx > 0  →  drag  (parallel to freestream)
  Fy > 0  →  lift  (perpendicular, upward)
"""
function compute_forces(u::AbstractMatrix, v::AbstractMatrix, p::AbstractMatrix,
                        obstacle::AbstractMatrix{Bool},
                        dx::Real, dy::Real, rho::Real, nu::Real)
    NX1, NY1 = size(u)
    mu = rho * nu

    Fx = 0.0
    Fy = 0.0

    for i in 2:NX1-1, j in 2:NY1-1
        !obstacle[i, j] && continue

        # ── Right face: fluid at (i+1, j),  n_solid = +x̂ ──────────────────
        if !obstacle[i+1, j]
            # pressure: -p n_x → Fx -= p * dy
            Fx -= p[i+1, j] * dy
            # viscous normal: 2μ (∂u/∂x) n_x, one-sided ∂u/∂x = u_fluid / dx
            Fx += 2mu * u[i+1, j] / dx * dy
            # viscous shear: μ (∂u/∂y + ∂v/∂x) n_x, one-sided ∂v/∂x = v_fluid / dx
            du_dy = (u[i+1, j+1] - u[i+1, j-1]) / (2dy)
            Fy += mu * (du_dy + v[i+1, j] / dx) * dy
        end

        # ── Left face: fluid at (i-1, j),  n_solid = -x̂ ───────────────────
        if !obstacle[i-1, j]
            # pressure: -p n_x → Fx += p * dy  (n_x = -1)
            Fx += p[i-1, j] * dy
            # viscous normal: 2μ (∂u/∂x) n_x, one-sided ∂u/∂x = -u_fluid / dx
            Fx += 2mu * u[i-1, j] / dx * dy
            # viscous shear with n_x = -1
            du_dy = (u[i-1, j+1] - u[i-1, j-1]) / (2dy)
            Fy -= mu * (du_dy - v[i-1, j] / dx) * dy
        end

        # ── Top face: fluid at (i, j+1),  n_solid = +ŷ ─────────────────────
        if !obstacle[i, j+1]
            # pressure: -p n_y → Fy -= p * dx
            Fy -= p[i, j+1] * dx
            # viscous normal: 2μ (∂v/∂y) n_y, one-sided ∂v/∂y = v_fluid / dy
            Fy += 2mu * v[i, j+1] / dy * dx
            # viscous shear: μ (∂u/∂y + ∂v/∂x) n_y, one-sided ∂u/∂y = u_fluid / dy
            dv_dx = (v[i+1, j+1] - v[i-1, j+1]) / (2dx)
            Fx += mu * (u[i, j+1] / dy + dv_dx) * dx
        end

        # ── Bottom face: fluid at (i, j-1),  n_solid = -ŷ ──────────────────
        if !obstacle[i, j-1]
            # pressure: -p n_y → Fy += p * dx  (n_y = -1)
            Fy += p[i, j-1] * dx
            # viscous normal: one-sided ∂v/∂y = -v_fluid / dy, n_y = -1 → +2μ v/dy
            Fy += 2mu * v[i, j-1] / dy * dx
            # viscous shear with n_y = -1
            dv_dx = (v[i+1, j-1] - v[i-1, j-1]) / (2dx)
            Fx += mu * (u[i, j-1] / dy - dv_dx) * dx
        end
    end

    return Fx, Fy
end

"""
    compute_cl_cd(Fx, Fy, rho, U_inf, chord) → (CL, CD)

Non-dimensionalise force components into lift and drag coefficients.
Reference quantity: q_ref = ½ ρ U∞² · chord  (unit-span, 2D).
"""
function compute_cl_cd(Fx::Real, Fy::Real, rho::Real, U_inf::Real, chord::Real)
    q_ref = 0.5 * rho * U_inf^2 * chord
    return Fy / q_ref, Fx / q_ref   # CL, CD
end

end # module
