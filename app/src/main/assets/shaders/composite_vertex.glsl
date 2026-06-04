// ============================================================================
// composite_vertex.glsl
// Sync Vision — Composite Pass-Through Vertex Shader
//
// Pass-through vertex shader for the main compositing pipeline.
// Takes a full-screen quad position and texture coordinates, passing
// them through to the fragment shader for camera + mask + label compositing.
//
// *** ROOT CAUSE FIX #2: Added uCameraTransform matrix ***
// The SurfaceTexture.getTransformMatrix() provides a 4x4 matrix that handles
// camera rotation and mirroring. Without applying this, the camera feed appears
// horizontal and mirrored. We now apply it to produce vCameraTexCoord for
// camera texture sampling, while vTexCoord remains untransformed for
// mask/label/depth textures.
// ============================================================================

#version 300 es

// Vertex attributes
layout(location = 0) in vec2 aPosition;   // Normalized device coordinates [-1, 1]
layout(location = 1) in vec2 aTexCoord;   // Texture coordinates [0, 1]

// *** FIX #2: Camera transform matrix from SurfaceTexture ***
// This matrix handles the rotation and mirroring of the camera feed.
// It is updated every frame from SurfaceTexture.getTransformMatrix().
uniform mat4 uCameraTransform;

// Varying output to fragment shader — for mask/label/depth textures
out vec2 vTexCoord;

// *** FIX #2: Transformed texture coordinates for camera sampling ***
out vec2 vCameraTexCoord;

void main() {
    // Pass through position unchanged (already in NDC)
    gl_Position = vec4(aPosition, 0.0, 1.0);

    // Pass untransformed texture coordinates for mask/label/depth textures
    vTexCoord = aTexCoord;

    // *** FIX #2: Apply SurfaceTexture transform for camera texture ***
    // The transform matrix handles rotation (portrait→landscape) and
    // mirroring (front camera). Without this, the camera appears
    // horizontal and/or mirrored.
    vCameraTexCoord = (uCameraTransform * vec4(aTexCoord, 0.0, 1.0)).xy;
}
