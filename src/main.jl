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

# ── NACA 4-digit parameters — adjustable at runtime ───────────────────────────
# naca_alpha starts from ALPHA so the simulation opens with the configured angle.
# Key bindings:
#   Q / A  →  angle of attack  ± 1°
#   W / S  →  thickness        ± 0.01
#   E / D  →  max camber       ± 0.01
#   R / F  →  camber position  ± 0.1
naca_m     = 0.04    # max camber as fraction of chord
naca_p     = 0.4    # chordwise position of max camber
naca_t     = 0.12    # max thickness as fraction of chord
naca_chord = 50.0    # chord length in grid units
naca_alpha = Float64(ALPHA)  # angle of attack — initialized from ALPHA, then mutable

# ── Grid setup ────────────────────────────────────────────────────────────────
x = LinRange(0, 100, NX)
y = LinRange(0, 100, NY)
x_grid, y_grid = generate_grid(x, y)

# ── NACA profile builder ──────────────────────────────────────────────────────
# Returns (obstacle_mask_cpu, profile_vertices) for the given NACA parameters.
function build_naca_profile(m, p, t, chord, alpha)
    obs = zeros(Bool, NX+1, NY+1)
    obs[1:NX, 1:NY] .= naca_profile_mask(x_grid, y_grid;
                                          m=m, p=p, t=t, chord=chord, alpha=alpha)

    xu, yu, xl, yl = naca_coordinates(m, p, t, chord; n_points=251, alpha=alpha)

    x_center = (minimum(x_grid) + maximum(x_grid)) / 2
    y_center = (minimum(y_grid) + maximum(y_grid)) / 2
    x_shift  = x_center - chord/2 * cosd(alpha)
    y_shift  = y_center + chord/2 * sind(alpha)
    xu .+= x_shift;  yu .+= y_shift
    xl .+= x_shift;  yl .+= y_shift

    verts = Float32[]
    for i in eachindex(xu)
        push!(verts, Float32(xu[i]), Float32(yu[i]))
    end
    for i in reverse(axes(xl, 1)[2:end-1])
        push!(verts, Float32(xl[i]), Float32(yl[i]))
    end

    return obs, verts
end

# ── Initialize fields ─────────────────────────────────────────────────────────
global u = fill(U_INF, NX+1, NY+1)
global v = zeros(NX+1, NY+1)
global p = zeros(NX+1, NY+1)

obstacle_h, profile_vertices = build_naca_profile(naca_m, naca_p, naca_t, naca_chord, naca_alpha)

# Move all simulation arrays to the compute device (GPU if available, CPU otherwise).
global u        = to_device(u)
global v        = to_device(v)
global p        = to_device(p)
global obstacle = to_device(copy(obstacle_h))

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

prev_key_1     = false; prev_key_2 = false; prev_key_3 = false; prev_key_space = false
prev_key_q     = false; prev_key_a = false  # naca_alpha +/-
prev_key_w     = false; prev_key_s = false  # naca_t +/-
prev_key_e     = false; prev_key_d = false  # naca_m +/-
prev_key_r     = false; prev_key_f = false  # naca_p +/-

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
    current_key_1     = GLFW.GetKey(eng.window, GLFW.KEY_1) != GLFW.RELEASE
    current_key_2     = GLFW.GetKey(eng.window, GLFW.KEY_2) != GLFW.RELEASE
    current_key_3     = GLFW.GetKey(eng.window, GLFW.KEY_3) != GLFW.RELEASE
    current_key_space = GLFW.GetKey(eng.window, GLFW.KEY_SPACE) != GLFW.RELEASE

    cur_q = GLFW.GetKey(eng.window, GLFW.KEY_Q) != GLFW.RELEASE
    cur_a = GLFW.GetKey(eng.window, GLFW.KEY_A) != GLFW.RELEASE
    cur_w = GLFW.GetKey(eng.window, GLFW.KEY_W) != GLFW.RELEASE
    cur_s = GLFW.GetKey(eng.window, GLFW.KEY_S) != GLFW.RELEASE
    cur_e = GLFW.GetKey(eng.window, GLFW.KEY_E) != GLFW.RELEASE
    cur_d = GLFW.GetKey(eng.window, GLFW.KEY_D) != GLFW.RELEASE
    cur_r = GLFW.GetKey(eng.window, GLFW.KEY_R) != GLFW.RELEASE
    cur_f = GLFW.GetKey(eng.window, GLFW.KEY_F) != GLFW.RELEASE

    global view_mode
    global prev_key_1, prev_key_2, prev_key_3, prev_key_space
    global prev_key_q, prev_key_a, prev_key_w, prev_key_s
    global prev_key_e, prev_key_d, prev_key_r, prev_key_f
    global naca_m, naca_p, naca_t, naca_alpha

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

    # NACA parameter changes — rising-edge detection, clamped to valid ranges.
    # On any change: rebuild the obstacle mask, re-upload the profile geometry,
    # and reset the flow field so the solver starts fresh for the new shape.
    naca_changed = false
    if cur_q && !prev_key_q; naca_alpha = clamp(naca_alpha + 1.0, -25.0, 25.0); naca_changed = true; end
    if cur_a && !prev_key_a; naca_alpha = clamp(naca_alpha - 1.0, -25.0, 25.0); naca_changed = true; end
    if cur_w && !prev_key_w; naca_t = clamp(round(naca_t + 0.01, digits=2), 0.04, 0.30); naca_changed = true; end
    if cur_s && !prev_key_s; naca_t = clamp(round(naca_t - 0.01, digits=2), 0.04, 0.30); naca_changed = true; end
    if cur_e && !prev_key_e; naca_m = clamp(round(naca_m + 0.01, digits=2), 0.00, 0.09); naca_changed = true; end
    if cur_d && !prev_key_d; naca_m = clamp(round(naca_m - 0.01, digits=2), 0.00, 0.09); naca_changed = true; end
    if cur_r && !prev_key_r; naca_p = clamp(round(naca_p + 0.1,  digits=1), 0.1,  0.9 ); naca_changed = true; end
    if cur_f && !prev_key_f; naca_p = clamp(round(naca_p - 0.1,  digits=1), 0.1,  0.9 ); naca_changed = true; end

    if naca_changed
        println(@sprintf("[NACA] m=%.2f  p=%.1f  t=%.2f  α=%.1f°  →  rebuilding profile...",
                         naca_m, naca_p, naca_t, naca_alpha))
        obs_new, verts_new = build_naca_profile(naca_m, naca_p, naca_t, naca_chord, naca_alpha)
        copyto!(obstacle_h, obs_new)
        global u        = to_device(fill(U_INF, NX+1, NY+1))
        global v        = to_device(zeros(NX+1, NY+1))
        global p        = to_device(zeros(NX+1, NY+1))
        global obstacle = to_device(copy(obstacle_h))
        global t        = 0.0
        VizEngine.upload_profile!(eng, verts_new, DX, DY)
    end

    prev_key_1 = current_key_1; prev_key_2 = current_key_2
    prev_key_3 = current_key_3; prev_key_space = current_key_space
    prev_key_q = cur_q; prev_key_a = cur_a
    prev_key_w = cur_w; prev_key_s = cur_s
    prev_key_e = cur_e; prev_key_d = cur_d
    prev_key_r = cur_r; prev_key_f = cur_f

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
    CL, CD = Forces.compute_cl_cd(Fx, Fy, 1.0, U_INF, naca_chord)

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

    # 8. Update window title with current NACA parameters and key hints
    GLFW.SetWindowTitle(eng.window, @sprintf(
        "Fluid — %s  |  NACA m=%.2f p=%.1f t=%.2f α=%.1f°  |  Q/A:α  W/S:t  E/D:m  R/F:p",
        view_label, naca_m, naca_p, naca_t, naca_alpha))

    # Render the CL/CD history plot in its own window.
    # CoeffPlot.render! switches to the coeff GL context internally, so we
    # restore the main window's context immediately after for the next iteration.
    CoeffPlot.render!(coeff_plot, t, CL, CD)
    GLFW.MakeContextCurrent(eng.window)
end


CoeffPlot.shutdown!(coeff_plot)
GLFW.MakeContextCurrent(eng.window)
VizEngine.shutdown!(eng)
