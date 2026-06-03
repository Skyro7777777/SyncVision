// ============================================================================
// path_vertex.glsl
// Sync Vision — Path Visualization Vertex Shader
//
// Vertex shader for rendering path visualization elements:
//   - Path lines (solid, dashed)
//   - Arrow heads indicating direction
//   - Waypoint markers
//
// Input layout:
//   - aPosition (vec2): vertex position in normalized coords [0,1]
//   - aPathInfo (vec4):
//       x = segment type: 0.0 = solid line, 1.0 = dashed line,
//                         2.0 = arrow head, 3.0 = waypoint dot
//       y = path status:  0.0 = CLEAR, 1.0 = PARTIAL, 2.0 = BLOCKED
//       z = parametric distance along path (0.0 to 1.0)
//       w = line width / point size scale factor
//
// Uses GL_LINES for path segments and GL_TRIANGLES for arrow heads.
// ============================================================================

#version 300 es

// Vertex attributes
layout(location = 0) in vec2 aPosition;    // Position in normalized coords [0, 1]
layout(location = 1) in vec4 aPathInfo;     // Segment metadata (see above)

// Uniforms
uniform vec2 uResolution;    // Viewport resolution for screen-space calculations

// Output to fragment shader
out float vSegmentType;      // 0=solid, 1=dashed, 2=arrow, 3=waypoint
out float vPathStatus;       // 0=CLEAR, 1=PARTIAL, 2=BLOCKED
out float vParametricDist;   // Distance along path [0,1]
out float vWidthScale;       // Line width / point size scale
out vec2  vScreenPos;        // Screen-space position for AA

void main() {
    // Convert normalized coordinates to NDC [-1, 1]
    vec2 ndc = aPosition * 2.0 - 1.0;

    // Flip Y axis
    ndc.y = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);

    // Pass path info to fragment shader
    vSegmentType   = aPathInfo.x;
    vPathStatus    = aPathInfo.y;
    vParametricDist = aPathInfo.z;
    vWidthScale    = aPathInfo.w;

    // Screen-space position for anti-aliasing
    vScreenPos = aPosition * uResolution;
}
