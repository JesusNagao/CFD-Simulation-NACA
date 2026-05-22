module NACA

export naca_profile_indices, naca_profile_mask, naca_coordinates

"""
    naca_coordinates(m, p, t, chord; n_points, alpha) → (x_upper, y_upper, x_lower, y_lower)

Compute surface coordinates of a NACA 4-digit airfoil, optionally rotated by an angle
of attack `alpha`.

Arguments:
- `m`: maximum camber as a fraction of chord (e.g. 0.04 for 4 %).
- `p`: chordwise position of maximum camber (e.g. 0.4 for 40 % chord).
- `t`: maximum thickness as a fraction of chord (e.g. 0.12 for 12 %).
- `chord`: chord length.
- `n_points`: number of sample points along the chord.
- `alpha`: angle of attack in degrees. Positive values rotate the nose upward
  (clockwise rotation when the freestream flows left-to-right). Default 0.

Returns `(x_upper, y_upper, x_lower, y_lower)`.
When `alpha ≠ 0`, `x_upper` and `x_lower` differ.
"""
function naca_coordinates(m::Float64, p::Float64, t::Float64, chord::Float64;
                          n_points::Int=401, alpha::Float64=0.0)

    x_ref = range(0.0, chord, length=n_points)

    yc = similar(x_ref)
    for (k, x) in enumerate(x_ref)
        if x <= p * chord
            yc[k] = m / p^2       * (2p * x/chord - (x/chord)^2) * chord
        else
            yc[k] = m / (1-p)^2  * ((1 - 2p) + 2p * x/chord - (x/chord)^2) * chord
        end
    end

    yt = similar(x_ref)
    for (k, x) in enumerate(x_ref)
        ξ = x / chord
        yt[k] = 5.0 * t * chord * (
            0.2969*sqrt(ξ) - 0.1260*ξ - 0.3516*ξ^2 + 0.2843*ξ^3 - 0.1015*ξ^4
        )
    end

    x_raw  = collect(x_ref)
    yu_raw = yc .+ yt
    yl_raw = yc .- yt

    alpha == 0.0 && return copy(x_raw), yu_raw, copy(x_raw), yl_raw

    # Clockwise rotation by alpha (positive alpha = nose up for left-to-right flow).
    # R(-alpha) = [[cosα, sinα], [-sinα, cosα]]
    cosα = cosd(alpha)
    sinα = sind(alpha)

    xu = @. x_raw * cosα + yu_raw * sinα
    yu = @. -x_raw * sinα + yu_raw * cosα
    xl = @. x_raw * cosα + yl_raw * sinα
    yl = @. -x_raw * sinα + yl_raw * cosα

    return xu, yu, xl, yl
end

"""
    naca_profile_mask(x_grid, y_grid; m, p, t, chord, alpha, ...) → BitMatrix

Build a boolean mask marking all grid points that lie **inside** the NACA airfoil body
(including the full interior, not just the surface).

Each grid point is transformed into the airfoil's local frame via an inverse rotation,
then tested analytically against the camber-line and thickness formula — no tolerance
or surface sampling needed.

Arguments:
- `x_grid`, `y_grid`: grid matrices of identical shape.
- `m`, `p`, `t`: NACA profile parameters.
- `chord`: chord length; if `nothing`, scaled to 60 % of the smaller grid extent.
- `alpha`: angle of attack in degrees (positive = nose up). Default 0.
"""
function naca_profile_mask(x_grid::AbstractMatrix, y_grid::AbstractMatrix;
        m::Float64=0.04, p::Float64=0.4, t::Float64=0.12,
        chord::Union{Nothing,Float64}=nothing,
        alpha::Float64=0.0)

    size(x_grid) == size(y_grid) || throw(ArgumentError("x and y must have the same shape"))

    x_center = (minimum(x_grid) + maximum(x_grid)) / 2
    y_center = (minimum(y_grid) + maximum(y_grid)) / 2

    if chord === nothing
        grid_width  = maximum(x_grid) - minimum(x_grid)
        grid_height = maximum(y_grid) - minimum(y_grid)
        chord = 0.6 * min(grid_width, grid_height)
    end

    # The chord midpoint maps to (x_center, y_center) in world coordinates.
    # Forward transform: world = R(-alpha) * local + origin
    # where R(-alpha) = [[cosα, sinα], [-sinα, cosα]]
    # Chord midpoint (chord/2, 0) in local → (chord/2·cosα, -chord/2·sinα) rotated.
    # Adding origin puts it at (x_center, y_center):
    cosα    = cosd(alpha)
    sinα    = sind(alpha)
    x_shift = x_center - chord/2 * cosα
    y_shift = y_center + chord/2 * sinα

    mask = falses(size(x_grid))

    for i in axes(x_grid, 1), j in axes(x_grid, 2)
        # Inverse rotation R(+alpha) maps world → local airfoil frame.
        u     = x_grid[i, j] - x_shift
        v     = y_grid[i, j] - y_shift
        x_loc = u * cosα - v * sinα
        y_loc = u * sinα + v * cosα

        0.0 <= x_loc <= chord || continue   # outside chord span

        ξ = x_loc / chord

        # Camber line at this chord station.
        yc_loc = if ξ <= p
            m / p^2 * (2p*ξ - ξ^2) * chord
        else
            m / (1-p)^2 * ((1 - 2p) + 2p*ξ - ξ^2) * chord
        end

        # Half-thickness at this chord station.
        yt_loc = 5.0 * t * chord * (
            0.2969*sqrt(ξ) - 0.1260*ξ - 0.3516*ξ^2 + 0.2843*ξ^3 - 0.1015*ξ^4
        )

        mask[i, j] = yc_loc - yt_loc <= y_loc <= yc_loc + yt_loc
    end

    return mask
end

"""
    naca_profile_indices(x_grid, y_grid; kwargs...) → Vector{CartesianIndex}

Return grid indices of all points inside the NACA airfoil body.
Keyword arguments are forwarded to `naca_profile_mask`.
"""
function naca_profile_indices(x_grid::AbstractMatrix, y_grid::AbstractMatrix; kwargs...)
    mask = naca_profile_mask(x_grid, y_grid; kwargs...)
    return findall(mask)
end

end # module
