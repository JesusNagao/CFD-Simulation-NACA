module NACA

export naca_profile_indices, naca_profile_mask, naca_coordinates

# A simple type placeholder for future profile configuration.
mutable struct profile_NACA
    P::Float64
end

"""
Compute the coordinates of a NACA 4-digit airfoil profile.

Arguments:
- `m`: maximum camber as a fraction of chord (e.g. 0.04 for 4%% camber).
- `p`: camber position as a fraction of chord (e.g. 0.4 for 40%% chord).
- `t`: maximum thickness as a fraction of chord (e.g. 0.12 for 12%% thickness).
- `chord`: chord length of the airfoil.
- `n_points`: number of points to sample along the chord.

Returns:
- `x_coords`: x positions along the chord.
- `y_upper`: y positions of the upper surface.
- `y_lower`: y positions of the lower surface.
"""
function naca_coordinates(m::Float64, p::Float64, t::Float64, chord::Float64; n_points::Int=401)
    # Generate a linear distribution of x along the chord line.
    x_ref = range(0.0, chord, length=n_points)

    # Compute mean camber line y-coordinates for upper/lower surfaces.
    yc = similar(x_ref)
    for (k, x) in enumerate(x_ref)
        if x <= p*chord
            yc[k] = m/(p^2) * (2p*x/chord - (x/chord)^2) * chord
        else
            yc[k] = m/((1-p)^2) * ((1 - 2p) + 2p*x/chord - (x/chord)^2) * chord
        end
    end

    # Compute thickness distribution around the mean camber line.
    yt = similar(x_ref)
    for (k, x) in enumerate(x_ref)
        ξ = x/chord
        yt[k] = 5.0 * t * chord * (
            0.2969*sqrt(ξ) -
            0.1260*ξ -
            0.3516*ξ^2 +
            0.2843*ξ^3 -
            0.1015*ξ^4
        )
    end

    # Build the upper and lower surface coordinates.
    x_coords = collect(x_ref)
    y_upper = yc .+ yt
    y_lower = yc .- yt
    return x_coords, y_upper, y_lower
end

"""
Build a boolean mask for grid points that lie on the centered NACA airfoil surface.

Arguments:
- `x_grid`, `y_grid`: grid matrices of identical shape.
- `m`, `p`, `t`: NACA profile parameters.
- `chord`: chord length of the airfoil. If `nothing`, the function scales the profile to roughly 60%% of the smaller grid extent.
- `n_points`: sample resolution along the airfoil chord.
- `tolerance`: optional distance threshold; if omitted, it is estimated from grid spacing.

Returns:
- `mask`: a boolean matrix of the same shape as the input grid.
"""
function naca_profile_mask(x_grid::AbstractMatrix, y_grid::AbstractMatrix;
        m::Float64=0.04, p::Float64=0.4, t::Float64=0.12, chord::Union{Nothing,Float64}=nothing,
        n_points::Int=401, tolerance::Union{Nothing,Float64}=nothing)

    # Require both grid matrices to have the same dimensions.
    size(x_grid) == size(y_grid) || throw(ArgumentError("x and y must have the same shape"))

    # Center the profile in the middle of the grid extent.
    x_center = (minimum(x_grid) + maximum(x_grid)) / 2
    y_center = (minimum(y_grid) + maximum(y_grid)) / 2

    # If no chord is given, scale the profile to the grid size.
    if chord === nothing
        grid_width = maximum(x_grid) - minimum(x_grid)
        grid_height = maximum(y_grid) - minimum(y_grid)
        chord = 0.6 * min(grid_width, grid_height)
    end

    # Compute centered airfoil coordinates.
    x_coords, y_upper, y_lower = naca_coordinates(m, p, t, chord; n_points=n_points)
    x_coords .-= chord/2          # shift chord origin to profile center
    x_coords .+= x_center         # center airfoil horizontally
    y_upper .+= y_center          # center upper surface vertically
    y_lower .+= y_center          # center lower surface vertically

    # Estimate tolerance from grid spacing if none was provided.
    if tolerance === nothing
        dx = sum(abs.(diff(x_grid[:,1]))) / max(1, length(x_grid[:,1]) - 1)
        dy = sum(abs.(diff(y_grid[1,:]))) / max(1, length(y_grid[1,:]) - 1)
        tolerance = 0.75 * max(dx, dy)
    end
    tol2 = tolerance^2

    mask = falses(size(x_grid))

    # Mark cells whose distance to either surface is within the tolerance.
    for i in axes(x_grid, 1), j in axes(x_grid, 2)
        xg = x_grid[i, j]
        yg = y_grid[i, j]
        for k in eachindex(x_coords)
            dx = xg - x_coords[k]
            dist2_up = dx^2 + (yg - y_upper[k])^2
            dist2_low = dx^2 + (yg - y_lower[k])^2
            if dist2_up <= tol2 || dist2_low <= tol2
                mask[i, j] = true
                break
            end
        end
    end

    return mask
end

"""
Return the list of grid indices where the NACA profile is present.

Arguments:
- `x_grid`, `y_grid`: grid matrices of identical shape.
- `kwargs...`: forwarded keyword arguments to `naca_profile_mask`.

Returns:
- vector of `CartesianIndex` values corresponding to profile points.
"""
function naca_profile_indices(x_grid::AbstractMatrix, y_grid::AbstractMatrix; kwargs...)
    mask = naca_profile_mask(x_grid, y_grid; kwargs...)
    return findall(mask)
end

end # module