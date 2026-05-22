module CoeffPlot

using GLFW
using ModernGL
using Printf

# ─── Shaders ──────────────────────────────────────────────────────────────────

const VERT_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
"""

const FRAG_SRC = """
#version 330 core
out vec4 FragColor;
uniform vec4 u_color;
void main() { FragColor = u_color; }
"""

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
uniform vec3 u_text_color;
void main() {
    float alpha = texture(u_text, TexCoord).r;
    FragColor = vec4(u_text_color, alpha);
}
"""

# ─── Font atlas (digits + punctuation for axis labels) ────────────────────────

const TEXT_FONT_WIDTH  = 5
const TEXT_FONT_HEIGHT = 7
const TEXT_FONT_GAP    = 1
const TEXT_ATLAS_CHARS = "0123456789.- "

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
    ' ' => ["     ", "     ", "     ", "     ", "     ", "     ", "     "],
)

# ─── Plot region in NDC [-1, 1]² ─────────────────────────────────────────────
# Shifted inward from the edges to leave margins for axis tick labels.
const PX0 = -0.68f0   # left
const PX1 =  0.92f0   # right
const PY0 = -0.70f0   # bottom
const PY1 =  0.88f0   # top

# ─── State ────────────────────────────────────────────────────────────────────

mutable struct CoeffWindow
    window::GLFW.Window
    prog::GLuint
    loc_color::GLint
    vao::GLuint
    vbo::GLuint
    cl_hist::Vector{Float32}
    cd_hist::Vector{Float32}
    t_hist::Vector{Float32}
    max_pts::Int
    alive::Bool
    prog_text::GLuint
    loc_text_color::GLint
    vao_text::GLuint
    vbo_text::GLuint
    text_texture::GLuint
end

# ─── Internal helpers ─────────────────────────────────────────────────────────

function _build_atlas()
    chars   = collect(TEXT_ATLAS_CHARS)
    atlas_w = length(chars) * (TEXT_FONT_WIDTH + TEXT_FONT_GAP)
    atlas_h = TEXT_FONT_HEIGHT
    data    = fill(UInt8(0), atlas_w * atlas_h)
    for (i, ch) in enumerate(chars)
        rows = get(TEXT_FONT_BITMAPS, ch, fill("     ", TEXT_FONT_HEIGHT))
        x0   = (i - 1) * (TEXT_FONT_WIDTH + TEXT_FONT_GAP)
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

function _tw(text::String, sc::Float32)
    isempty(text) && return 0f0
    sp = sc * 0.2f0
    length(text) * (sc * TEXT_FONT_WIDTH + sp) - sp
end

function _text_quads(text::String, x::Float32, y::Float32, sc::Float32)
    atlas_w = length(TEXT_ATLAS_CHARS) * (TEXT_FONT_WIDTH + TEXT_FONT_GAP)
    verts   = Float32[]
    cursor  = x
    sp      = sc * 0.2f0
    for ch in text
        idx = findfirst(==(ch), TEXT_ATLAS_CHARS)
        if idx === nothing
            cursor += sc * TEXT_FONT_WIDTH + sp
            continue
        end
        u0 = Float32((idx - 1) * (TEXT_FONT_WIDTH + TEXT_FONT_GAP)) / Float32(atlas_w)
        u1 = Float32((idx - 1) * (TEXT_FONT_WIDTH + TEXT_FONT_GAP) + TEXT_FONT_WIDTH) / Float32(atlas_w)
        w  = sc * TEXT_FONT_WIDTH
        h  = sc * TEXT_FONT_HEIGHT
        x1 = cursor + w
        y1 = y + h
        append!(verts, Float32[
            cursor, y,   u0, 1f0,
            x1,     y,   u1, 1f0,
            x1,     y1,  u1, 0f0,
            cursor, y,   u0, 1f0,
            x1,     y1,  u1, 0f0,
            cursor, y1,  u0, 0f0,
        ])
        cursor += w + sp
    end
    return verts
end

function _draw_labels!(cw::CoeffWindow, verts::Vector{Float32}, r::Real, g::Real, b::Real)
    isempty(verts) && return
    glBindBuffer(GL_ARRAY_BUFFER, cw.vbo_text)
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glUseProgram(cw.prog_text)
    glUniform3f(cw.loc_text_color, Float32(r), Float32(g), Float32(b))
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, cw.text_texture)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBindVertexArray(cw.vao_text)
    glDrawArrays(GL_TRIANGLES, 0, length(verts) ÷ 4)
    glBindVertexArray(0)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_BLEND)
end

function _upload_draw!(cw::CoeffWindow, verts::Vector{Float32},
                       r::Real, g::Real, b::Real;
                       mode=GL_LINE_STRIP, lw::Float32=1f0)
    isempty(verts) && return
    glBindBuffer(GL_ARRAY_BUFFER, cw.vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glUseProgram(cw.prog)
    glUniform4f(cw.loc_color, Float32(r), Float32(g), Float32(b), 1f0)
    lw != 1f0 && glLineWidth(lw)
    glBindVertexArray(cw.vao)
    glDrawArrays(mode, 0, length(verts) ÷ 2)
    glBindVertexArray(0)
    lw != 1f0 && glLineWidth(1f0)
end

function _ndc(t_val, coeff_val, t0, t1, v0, v1)
    x = PX0 + (PX1 - PX0) * (Float32(t_val)    - t0) / max(t1 - t0, 1f-6)
    y = PY0 + (PY1 - PY0) * (Float32(coeff_val) - v0) / max(v1 - v0, 1f-6)
    clamp(x, -1f0, 1f0), clamp(y, -1f0, 1f0)
end

function _square_verts(x0::Float32, y0::Float32, s::Float32)
    x1 = x0 + s; y1 = y0 + s
    Float32[x0, y0,  x1, y0,  x1, y1,
            x0, y0,  x1, y1,  x0, y1]
end

# ─── Init ─────────────────────────────────────────────────────────────────────

"""
    init(; title, width, height, max_pts) → CoeffWindow

Create a second GLFW window with its own OpenGL context for plotting CL/CD history.
GLFW must already be initialised (by VizEngine.init).
This call leaves the coeff window's context current; the caller must restore the
main window's context via `GLFW.MakeContextCurrent(eng.window)` afterwards.
"""
function init(; title::String="CL / CD", width::Int=520, height::Int=520,
              max_pts::Int=4000)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 3)
    GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
    GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, true)

    win = GLFW.CreateWindow(width, height, title)
    GLFW.MakeContextCurrent(win)
    GLFW.SwapInterval(0)

    function _compile(src, type)
        s = glCreateShader(type)
        glShaderSource(s, 1, [src], C_NULL)
        glCompileShader(s)
        s
    end

    # Line/geometry shader
    v = _compile(VERT_SRC, GL_VERTEX_SHADER)
    f = _compile(FRAG_SRC, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, v); glAttachShader(prog, f)
    glLinkProgram(prog)
    glDeleteShader(v); glDeleteShader(f)
    loc = glGetUniformLocation(prog, "u_color")

    vao_r = Ref{GLuint}(0); vbo_r = Ref{GLuint}(0)
    glGenVertexArrays(1, vao_r); glGenBuffers(1, vbo_r)
    glBindVertexArray(vao_r[])
      glBindBuffer(GL_ARRAY_BUFFER, vbo_r[])
      glBufferData(GL_ARRAY_BUFFER, 0, C_NULL, GL_DYNAMIC_DRAW)
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, Ptr{Nothing}(0))
      glEnableVertexAttribArray(0)
    glBindVertexArray(0)

    # Text shader
    vt = _compile(VERT_TEXT_SRC, GL_VERTEX_SHADER)
    ft = _compile(FRAG_TEXT_SRC, GL_FRAGMENT_SHADER)
    prog_text = glCreateProgram()
    glAttachShader(prog_text, vt); glAttachShader(prog_text, ft)
    glLinkProgram(prog_text)
    glDeleteShader(vt); glDeleteShader(ft)
    glUseProgram(prog_text)
    glUniform1i(glGetUniformLocation(prog_text, "u_text"), 0)
    loc_tc = glGetUniformLocation(prog_text, "u_text_color")

    # Font atlas texture
    atlas_w, atlas_h, atlas_data = _build_atlas()
    tex_r = Ref{GLuint}(0)
    glGenTextures(1, tex_r)
    glBindTexture(GL_TEXTURE_2D, tex_r[])
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
      glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, atlas_w, atlas_h, 0,
                   GL_RED, GL_UNSIGNED_BYTE, atlas_data)
    glBindTexture(GL_TEXTURE_2D, 0)

    # Text VAO/VBO  (layout: x, y, u, v — 4 floats per vertex)
    vao_t = Ref{GLuint}(0); vbo_t = Ref{GLuint}(0)
    glGenVertexArrays(1, vao_t); glGenBuffers(1, vbo_t)
    glBindVertexArray(vao_t[])
      glBindBuffer(GL_ARRAY_BUFFER, vbo_t[])
      glBufferData(GL_ARRAY_BUFFER, 0, C_NULL, GL_DYNAMIC_DRAW)
      stride_t = 4 * sizeof(Float32)
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride_t, Ptr{Nothing}(0))
      glEnableVertexAttribArray(0)
      glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride_t, Ptr{Nothing}(2*sizeof(Float32)))
      glEnableVertexAttribArray(1)
    glBindVertexArray(0)

    return CoeffWindow(win, prog, loc, vao_r[], vbo_r[],
                       Float32[], Float32[], Float32[], max_pts, true,
                       prog_text, loc_tc, vao_t[], vbo_t[], tex_r[])
end

# ─── Public API ───────────────────────────────────────────────────────────────

"""
    render!(cw, t, CL, CD) → Bool

Append the new (t, CL, CD) sample, redraw the history plot, and return false if
the window has been closed.

This function switches to the coeff window's GL context.  The caller is
responsible for restoring the main window's context afterwards with:
    GLFW.MakeContextCurrent(eng.window)
"""
function render!(cw::CoeffWindow, t::Real, CL::Real, CD::Real)
    !cw.alive && return false

    if GLFW.WindowShouldClose(cw.window)
        _destroy!(cw)
        return false
    end

    # Append sample, trimming the oldest point when at capacity.
    push!(cw.t_hist,  Float32(t))
    push!(cw.cl_hist, Float32(CL))
    push!(cw.cd_hist, Float32(CD))
    if length(cw.t_hist) > cw.max_pts
        popfirst!(cw.t_hist)
        popfirst!(cw.cl_hist)
        popfirst!(cw.cd_hist)
    end

    GLFW.MakeContextCurrent(cw.window)
    glClearColor(0.07f0, 0.07f0, 0.10f0, 1f0)
    glClear(GL_COLOR_BUFFER_BIT)

    n = length(cw.t_hist)
    if n >= 2
        t0 = cw.t_hist[1];  t1 = cw.t_hist[end]

        # Auto-scale y to include all data with a small padding.
        all_v  = vcat(cw.cl_hist, cw.cd_hist)
        lo, hi = minimum(all_v), maximum(all_v)
        pad    = max((hi - lo) * 0.12f0, 0.10f0)
        v0 = lo - pad
        v1 = hi + pad

        # ── Grid lines every 0.5 coefficient units ─────────────────────────
        gs = 0.5f0
        gv = ceil(v0 / gs) * gs
        while gv <= v1
            _, gy = _ndc(t0, gv, t0, t1, v0, v1)
            if abs(gv) < 1f-5   # zero line is brighter
                _upload_draw!(cw, Float32[PX0, gy, PX1, gy],
                              0.45, 0.45, 0.50; mode=GL_LINES, lw=1.5f0)
            else
                _upload_draw!(cw, Float32[PX0, gy, PX1, gy],
                              0.18, 0.18, 0.22; mode=GL_LINES)
            end
            gv += gs
        end

        # ── Axis border (left + bottom) ────────────────────────────────────
        _upload_draw!(cw, Float32[PX0, PY0, PX1, PY0],
                      0.45, 0.45, 0.50; mode=GL_LINES)
        _upload_draw!(cw, Float32[PX0, PY0, PX0, PY1],
                      0.45, 0.45, 0.50; mode=GL_LINES)

        # ── CL history — blue ──────────────────────────────────────────────
        cl_v = Float32[]
        for i in 1:n
            x, y = _ndc(cw.t_hist[i], cw.cl_hist[i], t0, t1, v0, v1)
            push!(cl_v, x, y)
        end
        _upload_draw!(cw, cl_v, 0.25, 0.65, 1.00; lw=2f0)

        # ── CD history — orange ────────────────────────────────────────────
        cd_v = Float32[]
        for i in 1:n
            x, y = _ndc(cw.t_hist[i], cw.cd_hist[i], t0, t1, v0, v1)
            push!(cd_v, x, y)
        end
        _upload_draw!(cw, cd_v, 1.00, 0.52, 0.15; lw=2f0)

        # ── Legend: two small coloured squares in the top-right corner ─────
        sq  = 0.040f0
        gap = 0.085f0
        lx  = PX1 - 0.06f0 - sq
        ly  = PY1 - 0.05f0 - sq
        _upload_draw!(cw, _square_verts(lx - gap, ly, sq),
                      0.25, 0.65, 1.00; mode=GL_TRIANGLES)
        _upload_draw!(cw, _square_verts(lx, ly, sq),
                      1.00, 0.52, 0.15; mode=GL_TRIANGLES)

        # ── Axis tick labels ───────────────────────────────────────────────
        lsc  = 0.013f0
        lh   = lsc * Float32(TEXT_FONT_HEIGHT)
        lbls = Float32[]

        # Y-axis: one numeric label per grid line, right-aligned left of PX0
        tick = ceil(v0 / gs) * gs
        while tick <= v1
            _, gy = _ndc(t0, tick, t0, t1, v0, v1)
            lbl = @sprintf("%.1f", tick)
            lw  = _tw(lbl, lsc)
            append!(lbls, _text_quads(lbl, PX0 - lw - 0.022f0, gy - lh * 0.5f0, lsc))
            tick += gs
        end

        # X-axis: t0 left-aligned, tmid centered, t1 right-aligned.
        # Left-aligning t0 (instead of centering) avoids collision with Y-axis labels.
        for (i, frac) in enumerate((0.0f0, 0.5f0, 1.0f0))
            tv  = t0 + frac * (t1 - t0)
            gx, _ = _ndc(tv, v0, t0, t1, v0, v1)
            lbl = @sprintf("%.1f", tv)
            lw  = _tw(lbl, lsc)
            tx  = if i == 1; gx             # left-align at t0
                  elseif i == 3; gx - lw    # right-align at t1
                  else; gx - lw * 0.5f0     # center at tmid
                  end
            append!(lbls, _text_quads(lbl, tx, PY0 - lh - 0.022f0, lsc))
        end

        _draw_labels!(cw, lbls, 0.75f0, 0.75f0, 0.82f0)
    end

    GLFW.SetWindowTitle(cw.window,
        @sprintf("CL = %+.4f   CD = %+.4f   (■ blue = CL,  ■ orange = CD)", CL, CD))
    GLFW.SwapBuffers(cw.window)
    return true
end

function _destroy!(cw::CoeffWindow)
    GLFW.MakeContextCurrent(cw.window)
    vao_r = Ref(cw.vao); glDeleteVertexArrays(1, vao_r)
    vbo_r = Ref(cw.vbo); glDeleteBuffers(1, vbo_r)
    vao_t = Ref(cw.vao_text); glDeleteVertexArrays(1, vao_t)
    vbo_t = Ref(cw.vbo_text); glDeleteBuffers(1, vbo_t)
    tex_t = Ref(cw.text_texture); glDeleteTextures(1, tex_t)
    glDeleteProgram(cw.prog)
    glDeleteProgram(cw.prog_text)
    GLFW.DestroyWindow(cw.window)
    cw.alive = false
end

"""
    shutdown!(cw)

Destroy the coeff window and free all GL resources.
Call before VizEngine.shutdown! so GLFW.Terminate() fires only once.
"""
function shutdown!(cw::CoeffWindow)
    cw.alive && _destroy!(cw)
end

export CoeffWindow, init, render!, shutdown!

end # module
