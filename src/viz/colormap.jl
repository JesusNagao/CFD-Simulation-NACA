# src/viz/colormap.jl
module Colormap

export compute_vorticity, compute_speed, compute_streamlines, compute_velocity_vectors, smooth_field, smooth_vorticity

"""
    compute_vorticity(u, v, dx, dy) → Matrix{Float32}

Calcula ω = ∂v/∂x − ∂u/∂y con diferencias centradas.
`u` es (nx+1)×ny, `v` es nx×(ny+1)  (staggered grid).
Devuelve ω en los centros de celda: nx×ny.
"""
function compute_vorticity(u::Matrix{Float64}, v::Matrix{Float64}, dx::Float64, dy::Float64)
    nx = size(u, 1) - 1
    ny = size(u, 2) - 1
    omega = Matrix{Float32}(undef, nx, ny)

    for j in 1:ny, i in 1:nx
        dvdx = (v[i+1, j] - v[i, j]) / dx      # ∂v/∂x centrado en celda (i,j)
        dudy = (u[i, j+1] - u[i, j]) / dy      # ∂u/∂y centrado en celda (i,j)
        omega[i, j] = Float32(dvdx - dudy)
    end

    return omega
end

"""
    compute_speed(u, v, dx, dy) → Matrix{Float32}

Calcula la magnitud de la velocidad |u| = sqrt(u^2 + v^2) en el centro de cada celda.
Devuelve una matriz nx×ny con valores de velocidad positiva.
"""
function compute_speed(u::Matrix{Float64}, v::Matrix{Float64}, dx::Float64, dy::Float64)
    nx = size(u, 1) - 1
    ny = size(u, 2) - 1
    speed = Matrix{Float32}(undef, nx, ny)

    for j in 1:ny, i in 1:nx
        u_c = 0.25 * (u[i, j] + u[i+1, j] + u[i, j+1] + u[i+1, j+1])
        v_c = 0.25 * (v[i, j] + v[i+1, j] + v[i, j+1] + v[i+1, j+1])
        speed[i, j] = Float32(sqrt(u_c^2 + v_c^2))
    end

    return speed
end

"""
    compute_streamlines(u, v, dx, dy, n_streamlines=20) → Vector{Vector{Tuple{Float64,Float64}}}

Compute streamlines by integrating velocity field from starting points.
Returns a vector of streamlines, each streamline is a vector of (x,y) points.
"""
function compute_streamlines(u::Matrix{Float64}, v::Matrix{Float64}, dx::Float64, dy::Float64, n_streamlines::Int=20)
    nx = size(u, 1) - 1
    ny = size(u, 2) - 1

    # Create grid coordinates
    x_coords = range(0, nx*dx, length=nx)
    y_coords = range(0, ny*dy, length=ny)

    streamlines = Vector{Vector{Tuple{Float64,Float64}}}()

    # Start streamlines from left inlet at different heights
    for i in 1:n_streamlines
        y_start = (i-1) * (ny*dy) / (n_streamlines-1)
        streamline = Vector{Tuple{Float64,Float64}}()

        x, y = 0.0, y_start
        push!(streamline, (x, y))

        # Integrate streamline using RK2
        max_steps = 1000
        ds = min(dx, dy) * 0.1  # Integration step size

        for step in 1:max_steps
            # Find grid cell containing current position
            i_cell = clamp(floor(Int, x / dx) + 1, 1, nx)
            j_cell = clamp(floor(Int, y / dy) + 1, 1, ny)

            # Interpolate velocity at current position
            if i_cell < nx && j_cell < ny
                # Bilinear interpolation
                fx = (x - (i_cell-1)*dx) / dx
                fy = (y - (j_cell-1)*dy) / dy

                u_interp = (1-fx)*(1-fy)*u[i_cell, j_cell] +
                          fx*(1-fy)*u[i_cell+1, j_cell] +
                          (1-fx)*fy*u[i_cell, j_cell+1] +
                          fx*fy*u[i_cell+1, j_cell+1]

                v_interp = (1-fx)*(1-fy)*v[i_cell, j_cell] +
                          fx*(1-fy)*v[i_cell+1, j_cell] +
                          (1-fx)*fy*v[i_cell, j_cell+1] +
                          fx*fy*v[i_cell+1, j_cell+1]

                # RK2 integration
                u1, v1 = u_interp, v_interp
                x_mid = x + 0.5 * ds * u1
                y_mid = y + 0.5 * ds * v1

                # Interpolate at midpoint
                i_mid = clamp(floor(Int, x_mid / dx) + 1, 1, nx)
                j_mid = clamp(floor(Int, y_mid / dy) + 1, 1, ny)

                if i_mid < nx && j_mid < ny
                    fx_mid = (x_mid - (i_mid-1)*dx) / dx
                    fy_mid = (y_mid - (j_mid-1)*dy) / dy

                    u2 = (1-fx_mid)*(1-fy_mid)*u[i_mid, j_mid] +
                         fx_mid*(1-fy_mid)*u[i_mid+1, j_mid] +
                         (1-fx_mid)*fy_mid*u[i_mid, j_mid+1] +
                         fx_mid*fy_mid*u[i_mid+1, j_mid+1]

                    v2 = (1-fx_mid)*(1-fy_mid)*v[i_mid, j_mid] +
                         fx_mid*(1-fy_mid)*v[i_mid+1, j_mid] +
                         (1-fx_mid)*fy_mid*v[i_mid, j_mid+1] +
                         fx_mid*fy_mid*v[i_mid+1, j_mid+1]

                    x += ds * u2
                    y += ds * v2

                    push!(streamline, (x, y))

                    # Stop if we exit the domain
                    if x > nx*dx || y < 0 || y > ny*dy
                        break
                    end
                else
                    break
                end
            else
                break
            end
        end

        push!(streamlines, streamline)
    end

    return streamlines
end

"""
    compute_velocity_vectors(u, v, dx, dy; spacing=16, scale=0.15) → Vector{Float32}

Create a set of small velocity vectors for visualization. Each item in the returned
vector contains two points (x0, y0) and (x1, y1) that form one arrow segment.
"""
function compute_velocity_vectors(u::Matrix{Float64}, v::Matrix{Float64}, dx::Float64, dy::Float64; spacing::Int=16, scale::Float64=0.15)
    nx = size(u, 1) - 1
    ny = size(u, 2) - 1
    vectors = Float32[]

    for j in 1:spacing:ny
        for i in 1:spacing:nx
            x = (i - 0.5) * dx
            y = (j - 0.5) * dy

            u_loc = 0.25 * (u[i, j] + u[i+1, j] + u[i, j+1] + u[i+1, j+1])
            v_loc = 0.25 * (v[i, j] + v[i+1, j] + v[i, j+1] + v[i+1, j+1])

            speed = sqrt(u_loc^2 + v_loc^2)
            if speed < 1e-8
                continue
            end

            length = max(0.1, scale * speed)
            dirx = u_loc / speed
            diry = v_loc / speed

            x_end = x + dirx * length
            y_end = y + diry * length

            # Arrow shaft
            push!(vectors, Float32(x), Float32(y), Float32(x_end), Float32(y_end))

            # Add arrowhead lines
            head_len = length * 0.35
            perp_x = -diry
            perp_y = dirx

            hx1 = x_end - dirx * head_len + perp_x * head_len * 0.5
            hy1 = y_end - diry * head_len + perp_y * head_len * 0.5
            hx2 = x_end - dirx * head_len - perp_x * head_len * 0.5
            hy2 = y_end - diry * head_len - perp_y * head_len * 0.5

            push!(vectors, Float32(x_end), Float32(y_end), Float32(hx1), Float32(hy1))
            push!(vectors, Float32(x_end), Float32(y_end), Float32(hx2), Float32(hy2))
        end
    end

    return vectors
end

"""
    smooth_field(field, iterations=1) → Matrix{Float32}

Apply a simple smoothing filter to a scalar field to reduce noise for visualization.
Funciona para magnitud de velocidad, vorticidad u otros campos escalares.
"""
function smooth_field(field::Matrix{Float32}, iterations::Int=1)
    smoothed = copy(field)
    nx, ny = size(field)

    for iter in 1:iterations
        temp = copy(smoothed)
        for j in 2:ny-1, i in 2:nx-1
            # Simple averaging filter
            smoothed[i,j] = 0.25 * temp[i,j] +
                           0.125 * (temp[i-1,j] + temp[i+1,j] + temp[i,j-1] + temp[i,j+1]) +
                           0.0625 * (temp[i-1,j-1] + temp[i-1,j+1] + temp[i+1,j-1] + temp[i+1,j+1])
        end
    end

    return smoothed
end

smooth_vorticity(field::Matrix{Float32}, iterations::Int=1) = smooth_field(field, iterations)

end # module