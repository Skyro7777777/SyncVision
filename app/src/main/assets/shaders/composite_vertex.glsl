// ============================================================================
// composite_vertex.glsl
// Sync Vision — Composite Pass-Through Vertex Shader
//
// Simple pass-through vertex shader for the main compositing pipeline.
// Takes a full-screen quad position and texture coordinates, passing
// them through to the fragment shader for camera + mask + label compositing.
// ============================================================================

#version 300 es

// Vertex attributes
layout(location = 0) in vec2 aPosition;   // Normalized device coordinates [-1, 1]
layout(location = 1) in vec2 aTexCoord;   // Texture coordinates [0, 1]

// Varying output to fragment shader
out vec2 vTexCoord;

void main() {
    // Pass through position unchanged (already in NDC)
    gl_Position = vec4(aPosition, 0.0, 1.0);

    // Pass texture coordinates to fragment shader
    vTexCoord = aTexCoord;
}
