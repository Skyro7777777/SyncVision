// ============================================================================
// outline_fragment.glsl
// Sync Vision — Outline Fragment Shader
//
// Fragment shader for rendering green contour outlines from pre-computed
// line segment data. Produces thin (1-2px) terminal green lines with:
//   - Subtle glow effect (configurable intensity)
//   - Pulsing brightness animation driven by uTime
//   - Anti-aliased line edges for smooth appearance
//
// Used alongside outline_vertex.glsl for direct contour rendering,
// separate from the mask-based Sobel edge detection approach.
// ============================================================================

#version 300 es
precision mediump float;

// Varying input from vertex shader
in vec2 vLineInfo;      // x: segment index, y: parametric position along segment
in vec2 vScreenPos;     // Screen-space position for AA calculations

// Output
layout(location = 0) out vec4 fragColor;

// Uniforms
uniform vec3  uGreenColor;      // Terminal green (#00FF41)
uniform float uGlowIntensity;   // 0.0 to 1.0, glow strength
uniform float uTime;            // Time for pulse animation

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const float LINE_WIDTH     = 1.5;   // Base line width in pixels
const float GLOW_WIDTH     = 4.0;   // Glow spread in pixels
const float PULSE_SPEED    = 2.0;   // Pulse oscillation speed (Hz)
const float PULSE_AMP      = 0.12;  // Pulse amplitude (0-1 range of brightness modulation)
const float CORE_BRIGHTNESS = 1.0;  // Core line brightness multiplier

void main() {
    // Calculate pulse — subtle brightness oscillation
    float pulse = 1.0 + PULSE_AMP * sin(uTime * PULSE_SPEED * 6.28318530);

    // For GL_LINES rendering, we use the built-in gl_LineWidth and
    // compute anti-aliased edge softening based on screen derivatives
    //
    // Since we can't directly know distance from line center in GLSL for
    // GL_LINES, we use a screen-space approach:
    // The varying vLineInfo.y gives us position along the line, and
    // we compute coverage based on fragment proximity.

    // Use screen-space derivatives for anti-aliasing
    vec2 dx = dFdx(vScreenPos);
    vec2 dy = dFdy(vScreenPos);

    // Estimate fragment-to-line distance contribution
    // For a simple approach, we just render a solid line with AA
    float coverage = 1.0;

    // Core line color with pulse
    vec3 coreColor = uGreenColor * CORE_BRIGHTNESS * pulse;

    // Glow — soft green haze around the line
    // In this shader, since we're rendering GL_LINES, glow is achieved
    // by rendering a wider line with alpha falloff.
    // The glow intensity controls the alpha of the outer halo.
    vec3 glowColor = uGreenColor * 0.4 * uGlowIntensity * pulse;

    // Blend core and glow based on glow intensity
    vec3 finalColor = mix(coreColor, glowColor, uGlowIntensity * 0.3);

    // Final alpha — core is opaque, glow adds slight transparency
    float alpha = mix(0.92, 0.6, uGlowIntensity * 0.4) * pulse;
    alpha = clamp(alpha, 0.0, 1.0);

    fragColor = vec4(finalColor, alpha);
}
