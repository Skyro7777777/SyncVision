// ============================================================================
// label_vertex.glsl
// Sync Vision — Label Vertex Shader
//
// Vertex shader for rendering text label quads. Each label is a screen-space
// quad with position and size specified per-instance. The vertex shader
// expands each quad from a unit square into the proper screen rectangle
// and computes texture coordinates for sampling from the label atlas.
//
// Input layout:
//   - aPosition (vec4): x, y = top-left corner in normalized coords [0,1]
//                        z = width, w = height in normalized coords
//   - aQuadVertex (vec2): unit quad corner (0,0), (1,0), (0,1), (1,1)
//
// Uses instanced rendering: one draw call renders all labels.
// ============================================================================

#version 300 es

// Vertex attributes
layout(location = 0) in vec4 aPosition;      // x, y, width, height of label quad
layout(location = 1) in vec2 aQuadVertex;     // Unit quad vertex (0 or 1 in each axis)

// Output to fragment shader
out vec2 vTexCoord;     // Texture coordinate for label atlas sampling
out vec2 vQuadPos;      // Position within the quad [0,1] for effects

void main() {
    // Expand unit quad to the label rectangle
    vec2 worldPos = aPosition.xy + aQuadVertex * aPosition.zw;

    // Convert normalized coordinates to NDC [-1, 1]
    vec2 ndc = worldPos * 2.0 - 1.0;

    // Flip Y axis (screen coords origin top-left vs GL origin bottom-left)
    ndc.y = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);

    // Texture coordinate matches quad vertex position
    // The label atlas stores each label as a sub-rect;
    // the fragment shader will handle atlas UV mapping
    vTexCoord = aQuadVertex;

    // Pass quad-relative position for per-fragment effects (e.g., shadow)
    vQuadPos = aQuadVertex;
}
