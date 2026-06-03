// ============================================================================
// outline_vertex.glsl
// Sync Vision — Outline Vertex Shader
//
// Vertex shader for drawing contour outlines directly from contour data.
// Used when rendering pre-computed contour line segments rather than
// performing edge detection on a mask texture in the fragment shader.
//
// Input: line segment endpoints in normalized [0,1] coordinates
// Output: clip-space position and varying for line parameterization
// ============================================================================

#version 300 es

// Vertex attributes
layout(location = 0) in vec2 aPosition;    // Line endpoint in normalized coords [0, 1]
layout(location = 1) in vec2 aLineInfo;     // x: line segment index, y: parametric t along segment

// Uniforms
uniform vec2 uResolution;    // Viewport resolution in pixels

// Varying output to fragment shader
out vec2 vLineInfo;          // x: segment index, y: parametric position along segment
out vec2 vScreenPos;         // Screen-space position for anti-aliasing

void main() {
    // Convert normalized coordinates to NDC [-1, 1]
    vec2 ndc = aPosition * 2.0 - 1.0;

    // Flip Y axis (screen coords vs GL coords)
    ndc.y = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);

    // Pass line info to fragment shader
    vLineInfo = aLineInfo;

    // Compute screen-space position for anti-aliasing calculations
    vScreenPos = aPosition * uResolution;
}
