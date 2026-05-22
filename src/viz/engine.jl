# src/viz/engine.jl
module VizEngine

using GLFW
using ModernGL

const SPEED_MIN = 0.5f0
const SPEED_MAX = 1.10f0

# ─── Shaders ──────────────────────────────────────────────────────────────────
# A fullscreen quad is rendered. The scalar field arrives as a 2D texture.

const VERT_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    TexCoord    = aTexCoord;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

# Colormap "speed": dark blue → cyan → yellow → red for increasing speed magnitude
const FRAG_SRC = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D u_field;
uniform float     u_vmin;
uniform float     u_vmax;        // maximum value for normalization
uniform int       u_view_mode;

vec3 speed_colormap(float t) {
    vec3 dark_blue = vec3(0.05, 0.1, 0.35);
    vec3 cyan      = vec3(0.2, 0.7, 0.9);
    vec3 yellow    = vec3(0.98, 0.9, 0.3);
    vec3 red       = vec3(0.85, 0.2, 0.1);

    if (t < 0.33) return mix(dark_blue, cyan, t / 0.33);
    else if (t < 0.66) return mix(cyan, yellow, (t - 0.33) / 0.33);
    else return mix(yellow, red, (t - 0.66) / 0.34);
}

vec3 pressure_colormap(float t) {
    vec3 blue  = vec3(0.05, 0.25, 0.7);
    vec3 white = vec3(0.95, 0.95, 0.95);
    vec3 red   = vec3(0.8, 0.15, 0.15);
    if (t < 0.5) return mix(blue, white, t * 2.0);
    return mix(white, red, (t - 0.5) * 2.0);
}

vec3 vorticity_colormap(float t) {
    vec3 purple = vec3(0.5, 0.1, 0.65);
    vec3 cyan   = vec3(0.15, 0.9, 0.9);
    vec3 yellow = vec3(0.98, 0.9, 0.3);
    if (t < 0.5) return mix(purple, cyan, t * 2.0);
    return mix(cyan, yellow, (t - 0.5) * 2.0);
}

void main() {
    float raw = texture(u_field, TexCoord).r;
    float t = 0.0;
    if (u_vmax > u_vmin) {
        t = clamp((raw - u_vmin) / (u_vmax - u_vmin), 0.0, 1.0);
    }

    vec3 color;
    if (u_view_mode == 1) {
        t = pow(t, 0.5);
        color = speed_colormap(t);
    } else if (u_view_mode == 2) {
        color = pressure_colormap(t);
    } else {
        color = vorticity_colormap(t);
    }

    if (u_view_mode == 1 && raw < 1e-6) {
        color = vec3(0.05, 0.1, 0.15);
    }

    FragColor = vec4(color, 1.0);
}
"""

# Streamlines shader - renders streamlines as smooth curves
const VERT_STREAMLINES_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
uniform vec3 u_color;
out vec3 Color;

void main() {
    Color = u_color;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

const FRAG_STREAMLINES_SRC = """
#version 330 core
in vec3 Color;
out vec4 FragColor;

void main() {
    FragColor = vec4(Color, 0.8);
}
"""

const VERT_LEGEND_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;
out vec3 Color;
void main() {
    Color = aColor;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

const FRAG_LEGEND_SRC = """
#version 330 core
in vec3 Color;
out vec4 FragColor;
void main() {
    FragColor = vec4(Color, 1.0);
}
"""

const TEXT_FONT_WIDTH = 5
const TEXT_FONT_HEIGHT = 7
const TEXT_FONT_GAP = 1
const TEXT_ATLAS_CHARS = "0123456789.-ms/ SPEEDRUVOTICY "

const TEXT_FONT_BITMAPS = Dict(
    '0' => [" 111 ", "1   1", "1  11", "1 1 1", "11  1", "1   1", " 111 "],
    '1' => ["  1  ", " 11  ", "1 1  ", "  1  ", "  1  ", "  1  ", "11111"],
    '2' => [" 111 ", "1   1", "    1", "  11 ", " 1   ", "1    ", "11111"],
    '3' => [" 111 ", "1   1", "    1", "  11 ", "    1", "1   1", " 111 "],
    '4' => ["   1 ", "  11 ", " 1 1 ", "1  1 ", "11111", "   1 ", "   1 "],
    '5' => ["11111", "1    ", "1111 ", "    1", "    1", "1   1", " 111 "],
    '6' => [" 111 ", "1   1", "1    ", "1111 ", "1   1", "1   1", " 111 "],
    '7' => ["11111", "    1", "   1 ", "  1  ", " 1   ", "1    ", "1    "],
    '8' => [" 111 ", "1   1", "1   1", " 111 ", "1   1", "1   1", " 111 "],
    '9' => [" 111 ", "1   1", "1   1", " 1111", "    1", "1   1", " 111 "],
    '.' => ["     ", "     ", "     ", "     ", "     ", "  11 ", "  11 "],
    '-' => ["     ", "     ", "     ", "11111", "     ", "     ", "     "],
    'm' => ["     ", "     ", "     ", "1   1", "11 11", "1 1 1", "1   1"],
    's' => [" 111 ", "1   1", "1    ", " 111 ", "    1", "1   1", " 111 "],
    '/' => ["    1", "   1 ", "   1 ", "  1  ", " 1   ", " 1   ", "1    "],
    'S' => [" 111 ", "1    ", "1    ", " 111 ", "    1", "1   1", " 111 "],
    'P' => ["1111 ", "1   1", "1   1", "1111 ", "1    ", "1    ", "1    "],
    'R' => ["1111 ", "1   1", "1   1", "1111 ", "1  1 ", "1   1", "1   1"],
    'E' => ["11111", "1    ", "1    ", "1111 ", "1    ", "1    ", "11111"],
    'D' => ["1111 ", "1   1", "1   1", "1   1", "1   1", "1   1", "1111 "],
    'U' => ["1   1", "1   1", "1   1", "1   1", "1   1", "1   1", " 111 "],
    'V' => ["1   1", "1   1", "1   1", "1   1", " 1 1 ", " 1 1 ", "  1  "],
    'O' => [" 111 ", "1   1", "1   1", "1   1", "1   1", "1   1", " 111 "],
    'T' => ["11111", "  1  ", "  1  ", "  1  ", "  1  ", "  1  ", "  1  "],
    'I' => ["11111", "  1  ", "  1  ", "  1  ", "  1  ", "  1  ", "11111"],
    'C' => [" 1111", "1    ", "1    ", "1    ", "1    ", "1    ", " 1111"],
    'Y' => ["1   1", "1   1", " 1 1 ", "  1  ", "  1  ", "  1  ", "  1  "],
    ' ' => ["     ", "     ", "     ", "     ", "     ", "     ", "     "]
)

const VERT_TEXT_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    TexCoord = aTexCoord;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

const FRAG_TEXT_SRC = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D u_text;
void main() {
    float alpha = texture(u_text, TexCoord).r;
    FragColor = vec4(1.0, 1.0, 1.0, alpha);
}
"""

# ─── Compilation helpers ───────────────────────────────────────────────────

function compile_shader(src::String, type::GLenum)
    shader = glCreateShader(type)
    glShaderSource(shader, 1, [src], C_NULL)
    glCompileShader(shader)
    status = Ref{GLint}(0)
    glGetShaderiv(shader, GL_COMPILE_STATUS, status)
    if status[] == GL_FALSE
        len = Ref{GLint}(0)
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, len)
        log = Vector{UInt8}(undef, len[])
        glGetShaderInfoLog(shader, len[], C_NULL, log)
        error("Shader error:\n$(String(log))")
    end
    return shader
end

function create_program(vert_src, frag_src)
    vert = compile_shader(vert_src, GL_VERTEX_SHADER)
    frag = compile_shader(frag_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vert); glAttachShader(prog, frag)
    glLinkProgram(prog)
    glDeleteShader(vert); glDeleteShader(frag)
    return prog
end
function build_text_atlas()
    chars = collect(TEXT_ATLAS_CHARS)
    atlas_w = length(chars) * (TEXT_FONT_WIDTH + TEXT_FONT_GAP)
    atlas_h = TEXT_FONT_HEIGHT
    data = fill(UInt8(0), atlas_w * atlas_h)

    for (i, ch) in enumerate(chars)
        rows = TEXT_FONT_BITMAPS[ch]
        x0 = (i - 1) * (TEXT_FONT_WIDTH + TEXT_FONT_GAP)
        for y in 1:TEXT_FONT_HEIGHT
            row = rows[y]
            for x in 1:TEXT_FONT_WIDTH
                if row[x] == '1'
                    data[(y - 1) * atlas_w + x0 + x] = 255
                end
            end
        end
    end

    return atlas_w, atlas_h, data
end

function text_width(text::String, scale::Float32)
    char_spacing = scale * 0.2f0
    return sum(scale * TEXT_FONT_WIDTH + char_spacing for _ in text) - char_spacing
end

function text_quad_vertices(text::String, x::Float32, y::Float32, scale::Float32)
    atlas_w = length(TEXT_ATLAS_CHARS) * (TEXT_FONT_WIDTH + TEXT_FONT_GAP)
    verts = Float32[]
    cursor = x
    char_spacing = scale * 0.2f0

    for ch in text
        idx = findfirst(==(ch), TEXT_ATLAS_CHARS)
        if idx === nothing
            cursor += scale * (TEXT_FONT_WIDTH + 1)
            continue
        end

        u0 = (Float32(idx - 1) * (TEXT_FONT_WIDTH + TEXT_FONT_GAP)) / atlas_w
        u1 = (Float32((idx - 1) * (TEXT_FONT_WIDTH + TEXT_FONT_GAP) + TEXT_FONT_WIDTH)) / atlas_w
        v0 = 1f0
        v1 = 0f0
        w = scale * TEXT_FONT_WIDTH
        h = scale * TEXT_FONT_HEIGHT
        x1 = cursor + w
        y1 = y + h

        append!(verts, Float32[
            cursor, y,   u0, v0,
            x1,     y,   u1, v0,
            x1,     y1,  u1, v1,
            cursor, y,   u0, v0,
            x1,     y1,  u1, v1,
            cursor, y1,  u0, v1
        ])

        cursor += w + char_spacing
    end

    return verts
end
# ─── Estado del engine ────────────────────────────────────────────────────────

mutable struct FluidEngine
    window::GLFW.Window
    prog::GLuint
    prog_streamlines::GLuint  # New program for streamlines
    vao::GLuint
    vao_streamlines::GLuint   # New VAO for streamlines
    vbo_streamlines::GLuint   # New VBO for streamlines
    vao_vectors::GLuint       # VAO for velocity vectors
    vbo_vectors::GLuint       # VBO for velocity vectors
    texture::GLuint
    loc_vmin::GLint          # location of the uniform u_vmin
    loc_vmax::GLint          # location of the uniform u_vmax
    loc_view_mode::GLint     # location of the uniform u_view_mode
    loc_streamline_color::GLint  # color for streamlines and vectors
    speed_min::Float32        # lower bound for color normalization
    speed_max::Float32        # upper bound for color normalization
    nx::Int
    ny::Int
    streamlines_data::Vector{Float32}  # Store streamline vertices
    vector_data::Vector{Float32}       # Store velocity vector vertices
    vao_profile::GLuint               # VAO for airfoil profile
    vbo_profile::GLuint               # VBO for airfoil profile
    profile_data::Vector{Float32}     # Store profile vertices with centroid prepended
    prog_legend::GLuint               # Shader program for the legend
    vao_legend::GLuint                # VAO for the legend bar
    vbo_legend::GLuint                # VBO for the legend bar
    vao_buttons::GLuint               # VAO for HUD buttons
    vbo_buttons::GLuint               # VBO for HUD buttons
    vao_legend_outline::GLuint        # VAO for legend outline
    vbo_legend_outline::GLuint        # VBO for legend outline
    legend_count::Int                 # Vertex count for legend rendering
    legend_outline_count::Int         # Outline vertex count
    prog_text::GLuint                 # Shader program for text labels
    vao_text::GLuint                  # VAO for text quads
    vbo_text::GLuint                  # VBO for text quads
    text_texture::GLuint              # Texture atlas for text rendering
    text_vertex_count::Int            # Vertex count for text rendering
    prev_left_pressed::Bool           # previous mouse left button state
end

# ─── Initialization ─────────────────────────────────────────────────────────

function init(nx::Int, ny::Int; title="Fluid Visualizer", width=900, height=900,
              speed_min::Float32=SPEED_MIN, speed_max::Float32=SPEED_MAX)
    GLFW.Init()
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 3)
    GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
    GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, true)  # required on macOS

    window = GLFW.CreateWindow(width, height, title)
    GLFW.MakeContextCurrent(window)
    GLFW.SetInputMode(window, GLFW.STICKY_KEYS, 1)
    GLFW.SwapInterval(1)  # vsync

    # Fullscreen quad: two triangles, NDC coordinates + UV coordinates
    #
    #  (-1,1)──────(1,1)
    #    │    ╲  B  │
    #    │  A  ╲    │
    #  (-1,-1)─────(1,-1)
    #
    vertices = Float32[
    #   x      y     u     v
       -1f0,  -1f0,  0f0,  0f0,
        1f0,  -1f0,  1f0,  0f0,
        1f0,   1f0,  1f0,  1f0,
       -1f0,   1f0,  0f0,  1f0,
    ]
    indices = UInt32[0, 1, 2,  2, 3, 0]

    vao_ref = Ref{GLuint}(0); vbo_ref = Ref{GLuint}(0); ebo_ref = Ref{GLuint}(0)
    glGenVertexArrays(1, vao_ref)
    glGenBuffers(1, vbo_ref)
    glGenBuffers(1, ebo_ref)

    glBindVertexArray(vao_ref[])
      glBindBuffer(GL_ARRAY_BUFFER, vbo_ref[])
      glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_ref[])
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW)

      stride = 4 * sizeof(Float32)
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, Ptr{Nothing}(0))
      glEnableVertexAttribArray(0)
      glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, Ptr{Nothing}(2*sizeof(Float32)))
      glEnableVertexAttribArray(1)
    glBindVertexArray(0)

    # 2D single-channel texture (R32F) for a scalar field such as speed
    tex_ref = Ref{GLuint}(0)
    glGenTextures(1, tex_ref)
    glBindTexture(GL_TEXTURE_2D, tex_ref[])
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
      # Reserve storage: nx×ny pixels, one float channel
      glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, nx, ny, 0,
                   GL_RED, GL_FLOAT, C_NULL)
    glBindTexture(GL_TEXTURE_2D, 0)

    prog = create_program(VERT_SRC, FRAG_SRC)
    glUseProgram(prog)
    glUniform1i(glGetUniformLocation(prog, "u_field"), 0)  # texture unit 0
    loc_vmin = glGetUniformLocation(prog, "u_vmin")
    loc_vmax = glGetUniformLocation(prog, "u_vmax")
    loc_view_mode = glGetUniformLocation(prog, "u_view_mode")

    # Initialize streamlines shader program
    prog_streamlines = create_program(VERT_STREAMLINES_SRC, FRAG_STREAMLINES_SRC)
    loc_streamline_color = glGetUniformLocation(prog_streamlines, "u_color")

    # Initialize legend shader program
    prog_legend = create_program(VERT_LEGEND_SRC, FRAG_LEGEND_SRC)

    # Initialize streamlines VAO and VBO
    vao_streamlines_ref = Ref{GLuint}(0)
    vbo_streamlines_ref = Ref{GLuint}(0)
    glGenVertexArrays(1, vao_streamlines_ref)
    glGenBuffers(1, vbo_streamlines_ref)

    glBindVertexArray(vao_streamlines_ref[])
    glBindBuffer(GL_ARRAY_BUFFER, vbo_streamlines_ref[])
    # Initially empty buffer - will be updated with streamlines data
    glBufferData(GL_ARRAY_BUFFER, 0, C_NULL, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, Ptr{Nothing}(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)

    # Initialize velocity vectors VAO and VBO
    vao_vectors_ref = Ref{GLuint}(0)
    vbo_vectors_ref = Ref{GLuint}(0)
    glGenVertexArrays(1, vao_vectors_ref)
    glGenBuffers(1, vbo_vectors_ref)

    glBindVertexArray(vao_vectors_ref[])
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vectors_ref[])
    glBufferData(GL_ARRAY_BUFFER, 0, C_NULL, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, Ptr{Nothing}(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)

    # Initialize profile VAO and VBO for the airfoil solid
    vao_profile_ref = Ref{GLuint}(0)
    vbo_profile_ref = Ref{GLuint}(0)
    glGenVertexArrays(1, vao_profile_ref)
    glGenBuffers(1, vbo_profile_ref)

    glBindVertexArray(vao_profile_ref[])
    glBindBuffer(GL_ARRAY_BUFFER, vbo_profile_ref[])
    glBufferData(GL_ARRAY_BUFFER, 0, C_NULL, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, Ptr{Nothing}(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)

    # Initialize legend VAO and VBO for the horizontal colorbar
    legend_vertices = Float32[
        # x      y       r      g      b
        -0.75f0, -0.85f0, 0.05f0, 0.1f0, 0.35f0,
        -0.75f0, -0.78f0, 0.05f0, 0.1f0, 0.35f0,
        -0.25f0, -0.85f0, 0.2f0, 0.7f0, 0.9f0,
        -0.25f0, -0.78f0, 0.2f0, 0.7f0, 0.9f0,
         0.25f0, -0.85f0, 0.98f0, 0.9f0, 0.3f0,
         0.25f0, -0.78f0, 0.98f0, 0.9f0, 0.3f0,
         0.75f0, -0.85f0, 0.85f0, 0.2f0, 0.1f0,
         0.75f0, -0.78f0, 0.85f0, 0.2f0, 0.1f0,
    ]

    vao_legend_ref = Ref{GLuint}(0)
    vbo_legend_ref = Ref{GLuint}(0)
    glGenVertexArrays(1, vao_legend_ref)
    glGenBuffers(1, vbo_legend_ref)

    glBindVertexArray(vao_legend_ref[])
    glBindBuffer(GL_ARRAY_BUFFER, vbo_legend_ref[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(legend_vertices), legend_vertices, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5*sizeof(Float32), Ptr{Nothing}(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5*sizeof(Float32), Ptr{Nothing}(2*sizeof(Float32)))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)

    # Initialize buttons VAO/VBO (empty for now)
    vao_buttons_ref = Ref{GLuint}(0)
    vbo_buttons_ref = Ref{GLuint}(0)
    glGenVertexArrays(1, vao_buttons_ref)
    glGenBuffers(1, vbo_buttons_ref)

    glBindVertexArray(vao_buttons_ref[])
    glBindBuffer(GL_ARRAY_BUFFER, vbo_buttons_ref[])
    glBufferData(GL_ARRAY_BUFFER, 0, C_NULL, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5*sizeof(Float32), Ptr{Nothing}(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5*sizeof(Float32), Ptr{Nothing}(2*sizeof(Float32)))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)

    legend_outline_vertices = Float32[
        -0.75f0, -0.85f0,
         0.75f0, -0.85f0,
         0.75f0, -0.78f0,
        -0.75f0, -0.78f0,
    ]

    vao_legend_outline_ref = Ref{GLuint}(0)
    vbo_legend_outline_ref = Ref{GLuint}(0)
    glGenVertexArrays(1, vao_legend_outline_ref)
    glGenBuffers(1, vbo_legend_outline_ref)

    glBindVertexArray(vao_legend_outline_ref[])
    glBindBuffer(GL_ARRAY_BUFFER, vbo_legend_outline_ref[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(legend_outline_vertices), legend_outline_vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, Ptr{Nothing}(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)

    prog_text = create_program(VERT_TEXT_SRC, FRAG_TEXT_SRC)
    glUseProgram(prog_text)
    glUniform1i(glGetUniformLocation(prog_text, "u_text"), 0)

    text_texture_ref = Ref{GLuint}(0)
    glGenTextures(1, text_texture_ref)
    glBindTexture(GL_TEXTURE_2D, text_texture_ref[])
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
      atlas_w, atlas_h, atlas_data = build_text_atlas()
      glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, atlas_w, atlas_h, 0,
                   GL_RED, GL_UNSIGNED_BYTE, atlas_data)
    glBindTexture(GL_TEXTURE_2D, 0)

    vao_text_ref = Ref{GLuint}(0)
    vbo_text_ref = Ref{GLuint}(0)
    glGenVertexArrays(1, vao_text_ref)
    glGenBuffers(1, vbo_text_ref)

    glBindVertexArray(vao_text_ref[])
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text_ref[])
    glBufferData(GL_ARRAY_BUFFER, 0, C_NULL, GL_DYNAMIC_DRAW)
    stride_text = 4 * sizeof(Float32)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride_text, Ptr{Nothing}(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride_text, Ptr{Nothing}(2*sizeof(Float32)))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)

    return FluidEngine(window, prog, prog_streamlines,
                       vao_ref[], vao_streamlines_ref[], vbo_streamlines_ref[],
                       vao_vectors_ref[], vbo_vectors_ref[], tex_ref[],
                       loc_vmin, loc_vmax, loc_view_mode, loc_streamline_color,
                       speed_min, speed_max,
                       nx, ny,
                       Float32[], Float32[],
                       vao_profile_ref[], vbo_profile_ref[], Float32[],
                       prog_legend, vao_legend_ref[], vbo_legend_ref[],
                       vao_buttons_ref[], vbo_buttons_ref[],
                       vao_legend_outline_ref[], vbo_legend_outline_ref[], 8, 4,
                       prog_text, vao_text_ref[], vbo_text_ref[], text_texture_ref[], 0, false)
end

# ─── Upload texture with new solver data ───────────────────────────────────

"""
    upload_scalar_field!(engine, field)

Upload the scalar matrix `field` (nx × ny, Float32) to the GPU.
Call this each time the solver produces a new timestep.
"""
function upload_scalar_field!(eng::FluidEngine, field::Matrix{Float32})
    @assert size(field) == (eng.nx, eng.ny) "field must be $(eng.nx)×$(eng.ny)"

    glBindTexture(GL_TEXTURE_2D, eng.texture)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, eng.nx, eng.ny,
                    GL_RED, GL_FLOAT, field)
    glBindTexture(GL_TEXTURE_2D, 0)
end

upload_vorticity!(eng::FluidEngine, omega::Matrix{Float32}) = upload_scalar_field!(eng, omega)

# ─── Render a frame ─────────────────────────────────────────────────────────

"""
    upload_streamlines!(engine, streamlines, dx, dy)

Upload streamlines data to GPU for rendering.
`streamlines` is a vector of vectors of (x,y) tuples.
"""
function upload_streamlines!(eng::FluidEngine, streamlines::Vector{Vector{Tuple{Float64,Float64}}}, dx::Float64, dy::Float64)
    # Convert streamlines to NDC coordinates (-1 to 1)
    eng.streamlines_data = Float32[]

    for streamline in streamlines
        for (x, y) in streamline
            # Convert to NDC: domain is 0..nx*dx, 0..ny*dy
            # Map to -1..1
            ndc_x = 2.0 * (x / (eng.nx * dx)) - 1.0
            ndc_y = 2.0 * (y / (eng.ny * dy)) - 1.0
            push!(eng.streamlines_data, ndc_x, ndc_y)
        end
    end

    # Update VBO with new data
    glBindBuffer(GL_ARRAY_BUFFER, eng.vbo_streamlines)
    glBufferData(GL_ARRAY_BUFFER, sizeof(eng.streamlines_data), eng.streamlines_data, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
end

function upload_velocity_vectors!(eng::FluidEngine, vectors::Vector{Float32}, dx::Float64, dy::Float64)
    eng.vector_data = Float32[]

    for idx in 1:4:length(vectors)
        x0 = Float64(vectors[idx])
        y0 = Float64(vectors[idx+1])
        x1 = Float64(vectors[idx+2])
        y1 = Float64(vectors[idx+3])

        ndc_x0 = 2.0 * (x0 / (eng.nx * dx)) - 1.0
        ndc_y0 = 2.0 * (y0 / (eng.ny * dy)) - 1.0
        ndc_x1 = 2.0 * (x1 / (eng.nx * dx)) - 1.0
        ndc_y1 = 2.0 * (y1 / (eng.ny * dy)) - 1.0

        push!(eng.vector_data, Float32(ndc_x0), Float32(ndc_y0), Float32(ndc_x1), Float32(ndc_y1))
    end

    glBindBuffer(GL_ARRAY_BUFFER, eng.vbo_vectors)
    glBufferData(GL_ARRAY_BUFFER, sizeof(eng.vector_data), eng.vector_data, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
end

function upload_profile!(eng::FluidEngine, profile_vertices::Vector{Float32}, dx::Float64, dy::Float64)
    # profile_vertices is a closed boundary contour in physical coordinates [x0,y0, x1,y1, ...]
    n_points = length(profile_vertices) ÷ 2
    if n_points < 3
        eng.profile_data = Float32[]
        return
    end

    eng.profile_data = Float32[]
    for idx in 1:2:length(profile_vertices)
        x = Float64(profile_vertices[idx])
        y = Float64(profile_vertices[idx+1])
        ndc_x = 2.0 * (x / (eng.nx * dx)) - 1.0
        ndc_y = 2.0 * (y / (eng.ny * dy)) - 1.0
        push!(eng.profile_data, Float32(ndc_x), Float32(ndc_y))
    end

    glBindBuffer(GL_ARRAY_BUFFER, eng.vbo_profile)
    glBufferData(GL_ARRAY_BUFFER, sizeof(eng.profile_data), eng.profile_data, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
end

function update_text_buffer!(eng::FluidEngine, text_vertices::Vector{Float32})
    eng.text_vertex_count = length(text_vertices) ÷ 4
    glBindBuffer(GL_ARRAY_BUFFER, eng.vbo_text)
    glBufferData(GL_ARRAY_BUFFER, sizeof(text_vertices), text_vertices, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
end

function build_legend_vertices(view_mode::Int)
    y0 = -0.85f0; y1 = -0.78f0
    if view_mode == 2  # Pressure: blue → white → red
        stops = [(-0.75f0, 0.05f0, 0.25f0, 0.7f0),
                 ( 0.0f0,  0.95f0, 0.95f0, 0.95f0),
                 ( 0.75f0, 0.8f0,  0.15f0, 0.15f0)]
    elseif view_mode == 3  # Vorticity: purple → cyan → yellow
        stops = [(-0.75f0, 0.5f0,  0.1f0,  0.65f0),
                 ( 0.0f0,  0.15f0, 0.9f0,  0.9f0),
                 ( 0.75f0, 0.98f0, 0.9f0,  0.3f0)]
    else  # Speed: dark blue → cyan → yellow → red
        stops = [(-0.75f0, 0.05f0, 0.1f0,  0.35f0),
                 (-0.25f0, 0.2f0,  0.7f0,  0.9f0),
                 ( 0.25f0, 0.98f0, 0.9f0,  0.3f0),
                 ( 0.75f0, 0.85f0, 0.2f0,  0.1f0)]
    end
    verts = Float32[]
    for (x, r, g, b) in stops
        append!(verts, [x, y0, r, g, b, x, y1, r, g, b])
    end
    return verts
end

"""
    render_frame!(engine, vmin, vmax, show_streamlines=false, capture_path=nothing) → Bool

Draw one frame and return `false` if the user closed the window (ESC or close button).
`vmin` and `vmax` normalize the scalar field colormap.
`show_streamlines` enables streamline rendering.
`capture_path` optionally saves the frame to an image file.
"""
function render_frame!(eng::FluidEngine, vmin::Float32, vmax::Float32, show_streamlines::Bool=false, capture_path::Union{Nothing,String}=nothing, view_mode::Int=1, view_name::String="")
    GLFW.PollEvents()

    if GLFW.WindowShouldClose(eng.window) ||
       GLFW.GetKey(eng.window, GLFW.KEY_ESCAPE) == GLFW.PRESS
        return (false, 0)
    end

    clicked = 0

    # Handle mouse clicks for HUD buttons
    left_pressed = GLFW.GetMouseButton(eng.window, GLFW.MOUSE_BUTTON_LEFT)
    #println("Left pressed: ", left_pressed, " (previous: ", eng.prev_left_pressed, ")")
    if left_pressed && !eng.prev_left_pressed
        # New click: obtain cursor position and map to NDC
        cpos  = GLFW.GetCursorPos(eng.window)
        wsize = GLFW.GetWindowSize(eng.window)
        w = Float64(wsize.width); h = Float64(wsize.height)
        ndc_x = Float32((cpos.x / w) * 2.0 - 1.0)
        ndc_y = Float32(-((cpos.y / h) * 2.0 - 1.0))
        println("[Click] cursor NDC: (", round(ndc_x, digits=3), ", ", round(ndc_y, digits=3), ")")

        # Define button layout (top-left area)
        btn_w = 0.22f0
        btn_h = 0.08f0
        gap = 0.02f0
        start_x = -0.96f0
        start_y = 0.92f0

        # Buttons: SPEED(1), PRESSURE(2), VORTICITY(3)
        btns = [ (start_x,               btn_w, btn_h, 1),
                 (start_x + (btn_w+gap), btn_w, btn_h, 2),
                 (start_x + 2*(btn_w+gap), btn_w, btn_h, 3) ]

        for (bx, bw, bh, mode) in btns
            if ndc_x >= bx && ndc_x <= bx+bw && ndc_y <= start_y && ndc_y >= start_y-bh
                clicked = mode
                break
            end
        end
    end


    eng.prev_left_pressed = left_pressed

    glClearColor(0f0, 0f0, 0f0, 1f0)
    glClear(GL_COLOR_BUFFER_BIT)

    # Render scalar field background
    glUseProgram(eng.prog)
    glUniform1f(eng.loc_vmin, vmin)
    glUniform1f(eng.loc_vmax, vmax)
    glUniform1i(eng.loc_view_mode, view_mode)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, eng.texture)

    glBindVertexArray(eng.vao)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
    glBindVertexArray(0)

    # Render the NACA airfoil as a gray filled shape with a dark outline
    if length(eng.profile_data) > 4
        glUseProgram(eng.prog_streamlines)
        glUniform3f(eng.loc_streamline_color, 0.65f0, 0.65f0, 0.65f0)  # Light gray fill
        glBindVertexArray(eng.vao_profile)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawArrays(GL_TRIANGLE_FAN, 0, length(eng.profile_data) ÷ 2)
        glDisable(GL_BLEND)

        glUniform3f(eng.loc_streamline_color, 0.25f0, 0.25f0, 0.25f0)  # Dark gray outline
        glLineWidth(2.0f0)
        glDrawArrays(GL_LINE_LOOP, 0, length(eng.profile_data) ÷ 2)
        glLineWidth(1.0f0)

        glBindVertexArray(0)
    end

    # Rebuild and upload the legend gradient to match the current view mode
    legend_verts = build_legend_vertices(view_mode)
    glBindBuffer(GL_ARRAY_BUFFER, eng.vbo_legend)
    glBufferData(GL_ARRAY_BUFFER, sizeof(legend_verts), legend_verts, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    eng.legend_count = length(legend_verts) ÷ 5

    # Render the legend colorbar in the lower part of the screen
    glUseProgram(eng.prog_legend)
    glBindVertexArray(eng.vao_legend)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, eng.legend_count)
    glDisable(GL_BLEND)

    glUseProgram(eng.prog_streamlines)
    glBindVertexArray(eng.vao_legend_outline)
    glUniform3f(eng.loc_streamline_color, 0.25f0, 0.25f0, 0.25f0)
    glLineWidth(2.0f0)
    glDrawArrays(GL_LINE_LOOP, 0, eng.legend_outline_count)
    glLineWidth(1.0f0)
    glBindVertexArray(0)

    # Draw numeric labels below the legend bar
    if vmax > vmin
        min_label = string(round(vmin, digits=2))
        mid_val = (vmin + vmax) / 2f0
        mid_label = string(round(mid_val, digits=2))
        max_label = string(round(vmax, digits=2))
        label_scale = 0.0035f0

        min_x = -0.75f0
        mid_x = -text_width(mid_label, label_scale) / 2f0
        max_x = 0.75f0 - text_width(max_label, label_scale)
        label_y = -0.94f0
        unit_y = -0.88f0

        text_data = Float32[]
        append!(text_data, text_quad_vertices(min_label, min_x, label_y, label_scale))
        append!(text_data, text_quad_vertices(mid_label, mid_x, label_y, label_scale))
        append!(text_data, text_quad_vertices(max_label, max_x, label_y, label_scale))
        append!(text_data, text_quad_vertices(view_name, -0.75f0, unit_y, label_scale * 0.9f0))

        update_text_buffer!(eng, text_data)
        glUseProgram(eng.prog_text)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, eng.text_texture)
        glBindVertexArray(eng.vao_text)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawArrays(GL_TRIANGLES, 0, eng.text_vertex_count)
        glDisable(GL_BLEND)
        glBindVertexArray(0)
    end

    # ── HUD buttons: gradient fill + border outline + centred label ──────────
    let
        btn_w  = 0.22f0
        btn_h  = 0.08f0
        gap    = 0.02f0
        sx     = -0.96f0
        sy     =  0.92f0
        lsc    = 0.0038f0                          # label text scale
        lh     = lsc * Float32(TEXT_FONT_HEIGHT)   # label height in NDC

        # (mode, label, x-origin, base fill colour r/g/b)
        defs = [
            (1, "SPEED",     sx,                   (0.04f0, 0.10f0, 0.50f0)),
            (2, "PRESSURE",  sx + (btn_w + gap),   (0.05f0, 0.32f0, 0.12f0)),
            (3, "VORTICITY", sx + 2*(btn_w + gap), (0.38f0, 0.05f0, 0.42f0)),
        ]

        lbl_verts = Float32[]   # accumulate all label quads for one text draw

        for (mode, label, bx, (br, bg, bb)) in defs
            x0 = bx; x1 = bx + btn_w
            y0 = sy - btn_h; y1 = sy
            act = (mode == view_mode)

            # ── gradient fill (bottom darker → top lighter) ───────────────
            if act
                top_r = min(br * 4.0f0, 0.55f0)
                top_g = min(bg * 3.5f0, 0.68f0)
                top_b = min(bb * 3.0f0, 0.90f0)
                bot_r = min(br * 2.0f0, 0.35f0)
                bot_g = min(bg * 2.0f0, 0.50f0)
                bot_b = min(bb * 2.0f0, 0.75f0)
            else
                top_r, top_g, top_b = br * 0.85f0, bg * 0.85f0, bb * 0.85f0
                bot_r, bot_g, bot_b = br * 0.50f0, bg * 0.50f0, bb * 0.50f0
            end
            fill_v = Float32[
                x0, y0, bot_r, bot_g, bot_b,
                x1, y0, bot_r, bot_g, bot_b,
                x1, y1, top_r, top_g, top_b,
                x0, y0, bot_r, bot_g, bot_b,
                x1, y1, top_r, top_g, top_b,
                x0, y1, top_r, top_g, top_b,
            ]
            glBindBuffer(GL_ARRAY_BUFFER, eng.vbo_buttons)
            glBufferData(GL_ARRAY_BUFFER, sizeof(fill_v), fill_v, GL_DYNAMIC_DRAW)
            glUseProgram(eng.prog_legend)
            glBindVertexArray(eng.vao_buttons)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glDisable(GL_BLEND)
            glBindVertexArray(0)

            # ── border outline ────────────────────────────────────────────
            out_v = Float32[x0, y0, x1, y0, x1, y1, x0, y1]
            glBindBuffer(GL_ARRAY_BUFFER, eng.vbo_streamlines)
            glBufferData(GL_ARRAY_BUFFER, sizeof(out_v), out_v, GL_DYNAMIC_DRAW)
            glUseProgram(eng.prog_streamlines)
            if act
                glUniform3f(eng.loc_streamline_color, 1.0f0, 1.0f0, 1.0f0)
                glLineWidth(2.5f0)
            else
                glUniform3f(eng.loc_streamline_color, 0.38f0, 0.38f0, 0.44f0)
            end
            glBindVertexArray(eng.vao_streamlines)
            glDrawArrays(GL_LINE_LOOP, 0, 4)
            glLineWidth(1.0f0)
            glBindVertexArray(0)

            # ── collect centred label geometry ────────────────────────────
            tw = text_width(label, lsc)
            append!(lbl_verts, text_quad_vertices(label,
                x0 + (btn_w - tw) / 2f0,
                y0 + (btn_h - lh) / 2f0,
                lsc))
        end

        # Single text draw for all three labels
        update_text_buffer!(eng, lbl_verts)
        glUseProgram(eng.prog_text)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, eng.text_texture)
        glBindVertexArray(eng.vao_text)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDrawArrays(GL_TRIANGLES, 0, eng.text_vertex_count)
        glDisable(GL_BLEND)
        glBindVertexArray(0)
    end


    if isempty(view_name)
        view_name = "Scalar"
    end
    GLFW.SetWindowTitle(eng.window, string("Fluid Flow — ", view_name, " [", round(vmin, digits=3), " → ", round(vmax, digits=3), "]"))

    if capture_path !== nothing
        capture_frame!(eng, capture_path)
    end

    GLFW.SwapBuffers(eng.window)
    return (true, clicked)
end

function capture_frame!(eng::FluidEngine, filename::AbstractString)
    width, height = GLFW.GetFramebufferSize(eng.window)

    pixels = Vector{UInt8}(undef, width * height * 3)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pointer(pixels))

    open(filename, "w") do io
        write(io, "P6\n", string(width), " ", string(height), "\n255\n")
        rowbytes = width * 3
        for row in 0:height-1
            off = (height - 1 - row) * rowbytes + 1
            write(io, pixels[off:off+rowbytes-1])
        end
    end
end

# ─── Cleanup ─────────────────────────────────────────────────────────────────

function shutdown!(eng::FluidEngine)
    tex = Ref(eng.texture); glDeleteTextures(1, tex)
    vao = Ref(eng.vao);     glDeleteVertexArrays(1, vao)
    vao_s = Ref(eng.vao_streamlines); glDeleteVertexArrays(1, vao_s)
    vao_v = Ref(eng.vao_vectors); glDeleteVertexArrays(1, vao_v)
    vao_p = Ref(eng.vao_profile); glDeleteVertexArrays(1, vao_p)
    vao_l = Ref(eng.vao_legend); glDeleteVertexArrays(1, vao_l)
    vao_lo = Ref(eng.vao_legend_outline); glDeleteVertexArrays(1, vao_lo)
    vao_t = Ref(eng.vao_text); glDeleteVertexArrays(1, vao_t)
    vbo_p = Ref(eng.vbo_profile); glDeleteBuffers(1, vbo_p)
    vbo_l = Ref(eng.vbo_legend); glDeleteBuffers(1, vbo_l)
    vbo_lo = Ref(eng.vbo_legend_outline); glDeleteBuffers(1, vbo_lo)
    vbo_t = Ref(eng.vbo_text); glDeleteBuffers(1, vbo_t)
    txt = Ref(eng.text_texture); glDeleteTextures(1, txt)
    glDeleteProgram(eng.prog)
    glDeleteProgram(eng.prog_streamlines)
    glDeleteProgram(eng.prog_legend)
    glDeleteProgram(eng.prog_text)
    GLFW.DestroyWindow(eng.window)
    GLFW.Terminate()
end

export FluidEngine, init, upload_scalar_field!, upload_profile!, upload_vorticity!, upload_streamlines!, upload_velocity_vectors!, render_frame!, capture_frame!, shutdown!

end # module