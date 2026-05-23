# examples/cavity_live.jl
# Run from the project root: julia --project examples/cavity_live.jl

include("../src/viz/engine.jl")
include("../src/viz/colormap.jl")
include("../src/viz/coefficients.jl")
include("../src/helpers/grid.jl")
include("../src/profiles/NACA.jl")
include("../src/calculations/solver.jl")
include("../src/calculations/forces.jl")

using GLFW
using .VizEngine
using .Colormap
using .CoeffPlot
using .Grid
using .NACA
using .Solver
using .Forces
using Printf

# ── GPU detection ─────────────────────────────────────────────────────────────
# Try CUDA (NVIDIA) first, then Metal (Apple Silicon), then fall back to CPU.
_use_cuda  = false
_use_metal = false

try
    using CUDA
    if CUDA.functional()
        _use_cuda = true
        println("[GPU] CUDA (NVIDIA) GPU detected — running on GPU")
        println("[GPU] Device: ", CUDA.name(CUDA.device()))
    else
        println("[GPU] CUDA loaded but not functional — check NVIDIA driver")
        try; CUDA.versioninfo(); catch e; println("[GPU] CUDA versioninfo error: ", e); end
    end
catch e
    println("[GPU] CUDA unavailable: ", e)
end

if !_use_cuda
    try
        using Metal
        _use_metal = Metal.functional()
        _use_metal && println("[GPU] Metal (Apple) GPU detected — running on GPU")
    catch; end
end

(_use_cuda || _use_metal) || println("[GPU] No GPU detected — running on CPU")

# Move an array to the active compute device.
function to_device(x::AbstractArray)
    _use_cuda  && return CUDA.cu(x)
    _use_metal && return Metal.MtlArray(x)
    return x
end

# Bring an array back to CPU (no-op when already on CPU).
to_host(x::AbstractArray) = (_use_cuda || _use_metal) ? Array(x) : x

# ── Simulation parameters ─────────────────────────────────────────────────────
const NX, NY  = 256, 256
const RE       = 100.0
const U_INF    = 1.0
const ALPHA    = 8.0   # angle of attack in degrees (positive = nose up)
const DX       = 100.0 / (NX - 1)
const DY       = 100.0 / (NY - 1)
const NU       = U_INF * 100.0 / RE  # kinematic viscosity from Reynolds number
const DT       = 0.01 * min(DX, DY) / U_INF  # Very conservative timestep


# ── Initialize fields ─────────────────────────────────────────────────────────
global u = fill(U_INF, NX+1, NY+1)
global v = zeros(NX+1, NY+1)
global p = zeros(NX+1, NY+1)

x = LinRange(0, 100, NX)
y = LinRange(0, 100, NY)

x_grid, y_grid = generate_grid(x, y)

# Build a boolean obstacle mask sized to match u, v, p (NX+1 × NY+1).
# naca_profile_mask returns an NX×NY mask; pad with false for the extra border row/column.
obstacle = zeros(Bool, NX+1, NY+1)
obstacle[1:NX, 1:NY] .= naca_profile_mask(x_grid, y_grid; m=0.04, p=0.4, t=0.12, chord=50.0, alpha=ALPHA)

# Keep a CPU copy of the obstacle mask for force integration (the loop-based
# compute_forces function cannot run on a GPU array directly).
const obstacle_h = copy(obstacle)

# Move all simulation arrays to the compute device (GPU if available, CPU otherwise).
global u        = to_device(u)
global v        = to_device(v)
global p        = to_device(p)
global obstacle = to_device(obstacle)

# Build a closed polygon for the airfoil surface and render it in gray.
# naca_coordinates returns separate x arrays for upper/lower after rotation.
profile_xu, profile_yu, profile_xl, profile_yl = naca_coordinates(0.04, 0.4, 0.12, 50.0; n_points=251, alpha=ALPHA)

# Center using the same convention as naca_profile_mask:
# chord midpoint maps to (x_center, y_center).
x_center = (minimum(x_grid) + maximum(x_grid)) / 2
y_center = (minimum(y_grid) + maximum(y_grid)) / 2
cosA     = cosd(ALPHA)
sinA     = sind(ALPHA)
x_shift  = x_center - 50.0/2 * cosA
y_shift  = y_center + 50.0/2 * sinA

profile_xu .+= x_shift;  profile_yu .+= y_shift
profile_xl .+= x_shift;  profile_yl .+= y_shift

profile_vertices = Float32[]
for i in eachindex(profile_xu)
    push!(profile_vertices, Float32(profile_xu[i]), Float32(profile_yu[i]))
end
for i in reverse(axes(profile_xl, 1)[2:end-1])
    push!(profile_vertices, Float32(profile_xl[i]), Float32(profile_yl[i]))
end

# ── Initialize engine ─────────────────────────────────────────────────────────
record_frames = false  # set to true to save every rendered frame to frames/frame_XXXXX.ppm
record_dir = "frames"
if record_frames
    mkpath(record_dir)
end
speed_min = 0.6f0
speed_max = 1.10f0
global frame_index = 0

const VIEW_SPEED = 1
const VIEW_PRESSURE = 2
const VIEW_VORTICITY = 3
view_mode = VIEW_SPEED
view_labels = Dict(VIEW_SPEED => "SPEED", VIEW_PRESSURE => "PRESSURE", VIEW_VORTICITY => "VORTICITY")

prev_key_1 = false
prev_key_2 = false
prev_key_3 = false
prev_key_space = false

eng = VizEngine.init(NX, NY; title="Fluid Flow — Speed Magnitude", width=700, height=700,
                     speed_min=speed_min, speed_max=speed_max)
VizEngine.upload_profile!(eng, profile_vertices, DX, DY)

# Open the CL/CD plot in a separate window.  CoeffPlot.init() leaves its own
# context current, so we immediately restore the main window's context.
coeff_plot = CoeffPlot.init()
GLFW.MakeContextCurrent(eng.window)

running = true
global t = 0.0

while running

    # 1. Advance the solver by one time step
    global u, v, p = Solver.solver(u, v, NX, NY, obstacle, DT, DX, DY, 1.0, p, NU, U_INF)
    global t = t + DT

    # 2. Read keyboard input for view toggles before computing the current view
    prev_view_label = view_labels[view_mode]
    GLFW.PollEvents()
    current_key_1 = GLFW.GetKey(eng.window, GLFW.KEY_1) != GLFW.RELEASE
    current_key_2 = GLFW.GetKey(eng.window, GLFW.KEY_2) != GLFW.RELEASE
    current_key_3 = GLFW.GetKey(eng.window, GLFW.KEY_3) != GLFW.RELEASE
    current_key_space = GLFW.GetKey(eng.window, GLFW.KEY_SPACE) != GLFW.RELEASE

    global view_mode
    global prev_key_1, prev_key_2, prev_key_3, prev_key_space

    if current_key_1 && !prev_key_1
        println("[Input] KEY_1 pressed")
        view_mode = VIEW_SPEED
    elseif current_key_2 && !prev_key_2
        println("[Input] KEY_2 pressed")
        view_mode = VIEW_PRESSURE
    elseif current_key_3 && !prev_key_3
        println("[Input] KEY_3 pressed")
        view_mode = VIEW_VORTICITY
    elseif current_key_space && !prev_key_space
        println("[Input] SPACE pressed")
        view_mode = mod(view_mode, 3) + 1
    end

    # Log when the view actually changes
    if prev_view_label != view_labels[view_mode]
        println("[Input] switching view: ", prev_view_label, " -> ", view_labels[view_mode])
    end

    prev_key_1 = current_key_1
    prev_key_2 = current_key_2
    prev_key_3 = current_key_3
    prev_key_space = current_key_space

    # 3. Bring fields back to CPU for visualization (no-op when already on CPU).
    u_h = to_host(u)
    v_h = to_host(v)
    p_h = to_host(p)

    # 4. Choose the current scalar field based on the selected view mode
    field = nothing
    view_label = view_labels[view_mode]

    if view_mode == VIEW_SPEED
        field = Colormap.compute_speed(u_h, v_h, DX, DY)
        field = Colormap.smooth_field(field, 2)
    elseif view_mode == VIEW_PRESSURE
        field = Colormap.compute_pressure(p_h)
        field = Colormap.smooth_field(field, 2)
    elseif view_mode == VIEW_VORTICITY
        field = Colormap.compute_vorticity(u_h, v_h, DX, DY)
        field = Colormap.smooth_vorticity(field, 2)
    end

    # 5. Upload the selected scalar field to the GPU texture
    VizEngine.upload_scalar_field!(eng, field)

    # 6. Normalize using the current range for the selected field
    vmin = minimum(field)
    vmax = maximum(field)

    # Compute CL and CD from surface pressure + viscous stress integration.
    Fx, Fy = Forces.compute_forces(u_h, v_h, p_h, obstacle_h, DX, DY, 1.0, NU)
    CL, CD = Forces.compute_cl_cd(Fx, Fy, 1.0, U_INF, 50.0)

    println(@sprintf("t = %.4f  |  view = %-9s  |  range = [%+.3f, %+.3f]  |  CL = %+.4f  CD = %+.4f",
                     t, view_label, vmin, vmax, CL, CD))

    # Stability check
    if isnan(vmax) || vmax > 1000.0
        println("Simulation became unstable! vmax = ", vmax)
        break
    end

    # 7. Render the frame — returns false if the user closed the window
    frame_path = record_frames ? joinpath(record_dir, @sprintf("frame_%05d.ppm", frame_index)) : nothing
    ret = VizEngine.render_frame!(eng, Float32(vmin), Float32(vmax), true, frame_path, view_mode, view_label)
    global running = ret[1]
    clicked = ret[2]
    if clicked != 0
        prev_view_label = view_labels[view_mode]
        view_mode = clicked
        if prev_view_label != view_labels[view_mode]
            println("[Input] switching view: ", prev_view_label, " -> ", view_labels[view_mode])
        end
    end
    if record_frames
        global frame_index += 1
    end

    # Render the CL/CD history plot in its own window.
    # CoeffPlot.render! switches to the coeff GL context internally, so we
    # restore the main window's context immediately after for the next iteration.
    CoeffPlot.render!(coeff_plot, t, CL, CD)
    GLFW.MakeContextCurrent(eng.window)
end


CoeffPlot.shutdown!(coeff_plot)
GLFW.MakeContextCurrent(eng.window)
VizEngine.shutdown!(eng)
