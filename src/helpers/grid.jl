module Grid

export generate_grid

"""
    generate_grid(x, y)

Create a full 2D coordinate grid from separate x and y ranges.
The returned matrices satisfy
`grid_space_x[i,j] == x[i]` and `grid_space_y[i,j] == y[j]`.
"""

function generate_grid(x::LinRange, y::LinRange)
    
    # Determine sizes for the x and y axes
    size_x = length(x)
    size_y = length(y)

    #Initialize empty grid
    grid_space_x = zeros(size_x, size_y)
    grid_space_y = zeros(size_x, size_y)

    #iterate through the entire x and y axis
    for i in range(1,size_x)
        for j in range(1, size_y)

            #fill grid values
            grid_space_x[i,j] = x[i]
            grid_space_y[i,j] = y[j]

        end
    end

    #return x and y grid
    return grid_space_x, grid_space_y
end


end