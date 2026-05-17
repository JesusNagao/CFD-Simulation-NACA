# examples/cavity_live.jl
# Ejecutar desde la raíz del proyecto:  julia --project examples/cavity_live.jl

include("../src/viz/engine.jl")
include("../src/viz/colormap.jl")
include("../src/helpers/grid.jl")
include("../src/profiles/NACA.jl")
include("../src/calculations/solver.jl")

using .VizEngine
using .Colormap
using .Grid
using .NACA
using .Solver
using Printf

# ── Parámetros de simulación ──────────────────────────────────────────────────
const NX, NY  = 256, 256  # Reduced for stability
const RE       = 100.0
const U_INF    = 1.0
const DX       = 100.0 / (NX - 1)
const DY       = 100.0 / (NY - 1)
const NU       = U_INF * 100.0 / RE  # kinematic viscosity from Reynolds number
const DT       = 0.01 * min(DX, DY) / U_INF  # Very conservative timestep

# ── Aquí irá tu solver real. Por ahora: datos de prueba animados ──────────────
# Reemplaza esta función con una llamada a tu módulo src/solver/
#=function fake_solver_step!(u, v, t, inds)
    for j in 1:NY+1, i in 1:NX+1
        if CartesianIndex(i, j) in inds
            u[i, j] = 0.0
            v[i, j] = 0.0
            continue
        end
        x = (i-1) * DX;  y = (j-1) * DY
        u[i, j] = sin(2π*x) * cos(2π*y) * cos(t)
        v[i, j] = -cos(2π*x) * sin(2π*y) * cos(t)
    end
end
=#

# ── Inicializar campos ────────────────────────────────────────────────────────
global u = fill(U_INF, NX+1, NY+1)
global v = zeros(NX+1, NY+1)
global p = zeros(NX+1, NY+1)

x = LinRange(0, 100, NX)
y = LinRange(0, 100, NY)

x_grid, y_grid = generate_grid(x, y)

inds = naca_profile_indices(x_grid, y_grid;m=0.04, p=0.4, t=0.12, chord=50.0)

# Build a closed polygon for the airfoil surface and render it in gray.
profile_x, profile_y_upper, profile_y_lower = naca_coordinates(0.04, 0.4, 0.12, 50.0; n_points=251)

# Center the profile inside the grid domain.
x_center = (minimum(x_grid) + maximum(x_grid)) / 2
y_center = (minimum(y_grid) + maximum(y_grid)) / 2
profile_x .-= 50.0/2
profile_x .+= x_center
profile_y_upper .+= y_center
profile_y_lower .+= y_center

profile_vertices = Float32[]
for i in 1:length(profile_x)
    push!(profile_vertices, Float32(profile_x[i]), Float32(profile_y_upper[i]))
end
for i in length(profile_x)-1:-1:2
    push!(profile_vertices, Float32(profile_x[i]), Float32(profile_y_lower[i]))
end

# ── Inicializar engine ────────────────────────────────────────────────────────
record_frames = true  # set to true to save every rendered frame to frames/frame_XXXXX.ppm
record_dir = "frames"
if record_frames
    mkpath(record_dir)
end
speed_min = 0.6f0
speed_max = 1.10f0
global frame_index = 0

eng = VizEngine.init(NX, NY; title="Fluid Flow — Speed Magnitude", width=700, height=700,
                     speed_min=speed_min, speed_max=speed_max)
VizEngine.upload_profile!(eng, profile_vertices, DX, DY)

running = true
global t = 0.0

while running

    # 1. Avanzar solver un paso de tiempo
    #fake_solver_step!(u, v, t, inds)
    global u, v, p = Solver.solver(u, v, NX, NY, inds, DT, DX, DY, 1.0, p, NU, U_INF)
    global t = t + DT
    println("Time step: t = ", t)

    # 2. Calcular magnitud de velocidad desde los campos u, v
    speed = Colormap.compute_speed(u, v, DX, DY)

    # 2.5. Suavizar campo de velocidad para visualización más fluida
    speed = Colormap.smooth_field(speed, 2)

    # 3. Subir magnitud de velocidad a la textura en GPU
    VizEngine.upload_scalar_field!(eng, speed)

    # 6. Normalizar con el máximo actual para el colormap de velocidad
    vmin = minimum(speed)
    vmax = maximum(speed)
    println("Time step: t = ", t, ", speed range = [", vmin, ", ", vmax, "]")

    # Stability check
    if isnan(vmax) || vmax > 1000.0
        println("Simulation became unstable! vmax = ", vmax)
        break
    end

    # 7. Dibujar frame — si devuelve false el usuario cerró la ventana
    frame_path = record_frames ? joinpath(record_dir, @sprintf("frame_%05d.ppm", frame_index)) : nothing
    global running = VizEngine.render_frame!(eng, Float32(vmin), Float32(vmax), false, frame_path)
    if record_frames
        global frame_index += 1
    end
end


VizEngine.shutdown!(eng)