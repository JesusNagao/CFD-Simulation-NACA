# 2D CFD Airfoil Simulation

A real-time 2D incompressible Navier-Stokes solver for flow around a NACA 4-digit airfoil, written in Julia with live OpenGL visualization and aerodynamic coefficient tracking.

![Simulation](simulation.mp4)

---

## Features

- **Fractional-step projection solver** — upwind advection, central diffusion, and a Red-Black SOR Poisson pressure solve in a single timestep loop
- **Immersed boundary method** — NACA 4-digit airfoils at arbitrary angle of attack, analytically masked onto the Cartesian grid
- **GPU acceleration** — automatically uses CUDA (NVIDIA) or Metal (Apple Silicon) when available; falls back to CPU transparently
- **Three scalar field views** — speed magnitude, pressure, and vorticity, each with its own colormap and colorbar
- **Real-time CL/CD plot** — a second window shows the lift and drag coefficient history as the simulation runs
- **Aerodynamic force integration** — surface integral of pressure and viscous traction over every immersed-boundary face

---

## Project Structure

```
.
├── Project.toml                  # Julia package manifest
├── src/
│   ├── main.jl                   # Entry point — parameters, init, simulation loop
│   ├── calculations/
│   │   ├── solver.jl             # Navier-Stokes fractional-step solver
│   │   └── forces.jl             # Surface force integration → CL / CD
│   ├── helpers/
│   │   └── grid.jl               # 2D Cartesian grid generation
│   ├── profiles/
│   │   └── NACA.jl               # NACA 4-digit airfoil geometry and masking
│   └── viz/
│       ├── engine.jl             # Main OpenGL window — field rendering, buttons, colorbar
│       ├── colormap.jl           # Scalar field computation (speed, pressure, vorticity)
│       └── coefficients.jl       # CL / CD history plot (second GLFW window)
```

---

## Requirements

- Julia ≥ 1.9
- **GLFW.jl** — window management and input
- **ModernGL.jl** — OpenGL bindings
- *(optional)* **CUDA.jl** — NVIDIA GPU acceleration
- *(optional)* **Metal.jl** — Apple Silicon GPU acceleration

Install dependencies from the project root:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

---

## Running

```bash
julia --project src/main.jl
```

Two windows open on startup:

| Window | Size | Content |
|--------|------|---------|
| **Fluid Flow** | 700 × 700 | Scalar field + airfoil + colorbar + view buttons |
| **CL / CD** | 520 × 520 | Time-history plot of lift and drag coefficients |

---

## Controls

### Keyboard (Fluid Flow window)

| Key | Action |
|-----|--------|
| `1` | Speed magnitude view |
| `2` | Pressure view |
| `3` | Vorticity view |
| `Space` | Cycle through views |
| `Esc` | Quit |

### Mouse

Click the **SPEED**, **PRESSURE**, or **VORTICITY** buttons in the top-left corner of the main window to switch views.

---

## Simulation Parameters

All parameters are defined at the top of [`src/main.jl`](src/main.jl).

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NX`, `NY` | 256, 256 | Grid resolution |
| `RE` | 100 | Reynolds number |
| `U_INF` | 1.0 | Freestream velocity |
| `ALPHA` | 8.0° | Angle of attack |
| `DX`, `DY` | 100 / (NX−1) | Grid spacing |
| `NU` | U_INF × 100 / RE | Kinematic viscosity |
| `DT` | 0.01 × min(DX, DY) / U_INF | Timestep (CFL-conservative) |
| Chord | 50.0 | Airfoil chord length (grid units) |
| NACA profile | 2412 | m=0.04, p=0.4, t=0.12 |

---

## Numerical Methods

### Solver (`solver.jl`)

A **fractional-step projection method** advances the solution one timestep in three stages:

1. **Predictor** — explicit advection (first-order upwind) and diffusion (central differences) give an intermediate velocity field `u*` that is not yet divergence-free.
2. **Pressure Poisson** — a Red-Black SOR iteration solves `∇²p = (ρ/Δt) ∇·u*` with Neumann BCs on all walls and a Dirichlet anchor at the outlet (`p = 0`). Convergence is checked every 25 iterations against a residual threshold of 10⁻⁵.
3. **Velocity correction** — `u = u* − (Δt/ρ) ∇p` projects `u*` onto the divergence-free subspace.

Boundary conditions:
- **Inlet** (left): uniform `u = U_INF`, `v = 0`
- **Walls** (top/bottom): no-penetration (`v = 0`), zero normal gradient for `u`
- **Outlet** (right): zero normal gradient (convective outflow), `p = 0`
- **Obstacle** interior: velocity forced to zero via a boolean mask (`free = !obstacle`)

### Airfoil geometry (`NACA.jl`)

The NACA 4-digit thickness and camber distributions are evaluated analytically. Each grid point is rotated into the airfoil's local frame and tested against the camber-line ± half-thickness envelope — no surface sampling or tolerance tuning required.

`naca_profile_mask` returns a boolean matrix; `naca_coordinates` returns the upper and lower surface contours for the polygon overlay.

### Aerodynamic forces (`forces.jl`)

For each obstacle cell that shares a face with a fluid cell, the discrete traction

```
t = σ · n̂  =  (−p I + 2μ E) · n̂
```

is integrated over that face. Pressure and viscous-normal terms are one-sided differences evaluated at the adjacent fluid node; the shear term uses centered differences along the face direction. The net force vector `(Fx, Fy)` is non-dimensionalised as:

```
CL = Fy / q_ref
CD = Fx / q_ref
q_ref = ½ ρ U_∞² chord
```

### Visualization (`engine.jl`, `colormap.jl`, `coefficients.jl`)

The main window renders a full-screen quad textured with the scalar field (uploaded each frame as an `R32F` texture), with the active colormap applied in the fragment shader. The airfoil polygon is drawn as a `GL_TRIANGLE_FAN` filled shape with a `GL_LINE_LOOP` outline. The colorbar gradient and view-toggle buttons use a per-vertex color shader with gradient fills.

The CL/CD window maintains a ring buffer of up to 4 000 samples and redraws the history using `GL_LINE_STRIP`. Y-axis grid lines and tick labels use an adaptive step size (`_nice_step`) that keeps approximately five grid lines regardless of the coefficient range. Axis labels are rendered from a built-in bitmap font atlas baked into a `GL_R8` texture.

### GPU acceleration (`main.jl`)

At startup the code probes for CUDA then Metal:

```julia
try; using CUDA;  _use_cuda  = CUDA.functional();  catch; end
try; using Metal; _use_metal = Metal.functional();  catch; end
```

`to_device` moves arrays to the active device; `to_host` copies them back for the force-integration loop (which runs on CPU) and for OpenGL uploads.

---

## Output

Setting `record_frames = true` in `main.jl` saves every rendered frame to `frames/frame_XXXXX.ppm`. These can be assembled into a video with FFmpeg:

```bash
ffmpeg -framerate 30 -i frames/frame_%05d.ppm -c:v libx264 -pix_fmt yuv420p output.mp4
```

---

## License

MIT
