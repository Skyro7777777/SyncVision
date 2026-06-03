// ============================================================================
// label_fragment.glsl
// Sync Vision — Label Fragment Shader
//
// Fragment shader for rendering text labels from a pre-rendered label atlas.
// Produces terminal-green colored text with:
//   - Slightly transparent dark background behind text for readability
//   - Text shadow (1px offset) for legibility against any camera content
//   - ALL CAPS visual style (labels are pre-rendered in uppercase)
//   - Subtle border glow in terminal green
//
// The label atlas is a texture containing all current labels packed into
// a single bitmap, with each label's UV rect passed as a uniform or
// encoded in the vertex data.
// ============================================================================

#version 300 es
precision mediump float;

// Varying input from vertex shader
in vec2 vTexCoord;     // Texture coordinate for atlas sampling
in vec2 vQuadPos;      // Position within the quad [0,1] for effects

// Output
layout(location = 0) out vec4 fragColor;

// Uniforms
uniform sampler2D uLabelAtlas;      // Pre-rendered label texture atlas
uniform vec3  uGreenColor;          // Terminal green (#00FF41)
uniform float uTime;                // For subtle animation
uniform vec4  uAtlasRect;           // Atlas sub-rect: x,y = top-left UV, z,w = size UV

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const float BG_ALPHA          = 0.55;  // Background panel opacity
const float SHADOW_OFFSET     = 0.012; // Shadow displacement in UV space
const float SHADOW_ALPHA      = 0.7;   // Shadow darkness
const float BORDER_GLOW_WIDTH = 0.04;  // Border glow width as fraction of quad size
const float BORDER_GLOW_ALPHA = 0.25;  // Border glow opacity

void main() {
    // Map quad-local UV to atlas UV
    vec2 atlasUV = uAtlasRect.xy + vTexCoord * uAtlasRect.zw;

    // Sample the label texture
    float textAlpha = texture(uLabelAtlas, atlasUV).a;

    // --- Text shadow: sample offset for readability ---
    vec2 shadowOffset = vec2(SHADOW_OFFSET, -SHADOW_OFFSET);
    float shadowAlpha = texture(uLabelAtlas, atlasUV + shadowOffset).a;

    // --- Background: semi-transparent dark panel ---
    // Inset the background slightly so it doesn't touch the quad edges
    vec2 bgInset = vec2(0.03);
    vec2 bgMin = bgInset;
    vec2 bgMax = vec2(1.0) - bgInset;
    bool insideBg = vQuadPos.x >= bgMin.x && vQuadPos.x <= bgMax.x &&
                    vQuadPos.y >= bgMin.y && vQuadPos.y <= bgMax.y;

    // --- Border glow: subtle green glow at the edges ---
    float distToEdge = min(
        min(vQuadPos.x, 1.0 - vQuadPos.x),
        min(vQuadPos.y, 1.0 - vQuadPos.y)
    );
    float borderGlow = smoothstep(0.0, BORDER_GLOW_WIDTH, distToEdge);
    borderGlow = 1.0 - borderGlow;  // Invert: bright at edges

    // --- Compositing ---

    // Start with transparent
    vec4 result = vec4(0.0);

    if (insideBg) {
        // Dark background panel
        result = vec4(0.0, 0.02, 0.0, BG_ALPHA);
    }

    // Add border glow
    vec3 glowColor = uGreenColor * 0.3;
    result.rgb = mix(result.rgb, glowColor, borderGlow * BORDER_GLOW_ALPHA);
    result.a = max(result.a, borderGlow * BORDER_GLOW_ALPHA);

    // Add text shadow (dark, offset)
    if (shadowAlpha > 0.1) {
        vec3 shadowColor = vec3(0.0, 0.05, 0.0);
        float shadowContrib = shadowAlpha * SHADOW_ALPHA;
        result.rgb = mix(result.rgb, shadowColor, shadowContrib * 0.5);
        result.a = max(result.a, shadowContrib * 0.4);
    }

    // Add text (terminal green, ALL CAPS — pre-rendered uppercase)
    if (textAlpha > 0.05) {
        vec3 textColor = uGreenColor * 1.1;  // Slightly brighter than outline
        result.rgb = mix(result.rgb, textColor, textAlpha * 0.95);
        result.a = max(result.a, textAlpha);
    }

    // Subtle fade-in at the top of the label (appears to "print" in)
    float topFade = smoothstep(0.0, 0.08, vQuadPos.y);
    result.a *= topFade;

    fragColor = result;
}
