// ============================================================================
// composite_fragment.glsl
// Sync Vision — Core Composite Fragment Shader
//
// The primary compositing shader that combines:
//   1. Camera feed (base layer)
//   2. Green outline overlay (via Sobel edge detection on segmentation mask)
//   3. Text label overlay (from pre-rendered label texture)
//   4. CRT scanline effect (subtle terminal aesthetic)
//   5. Night mode brightness boost
//   6. Glow effect around green outlines
//
// The green outline is the signature visual of Sync Vision — thin (1-2px),
// bright terminal green (#00FF41) lines tracing object boundaries detected
// by the segmentation model, creating an E.D.I.T.H-inspired AR overlay.
// ============================================================================

#version 300 es
precision mediump float;

// Varying input from vertex shader
in vec2 vTexCoord;

// Output
layout(location = 0) out vec4 fragColor;

// Texture uniforms
uniform sampler2D uCameraTexture;    // Camera feed (RGBA)
uniform sampler2D uMaskTexture;      // Segmentation mask (R8, class IDs as float)
uniform sampler2D uLabelTexture;     // Text labels on transparent background (RGBA)
uniform sampler2D uDepthTexture;     // Depth map (optional, R8)

// Parameter uniforms
uniform vec2  uTexelSize;           // 1.0/width, 1.0/height
uniform vec3  uGreenColor;          // #00FF41 = vec3(0.0, 1.0, 0.255)
uniform float uTime;                // Elapsed time for animations (seconds)
uniform float uScanlineIntensity;   // 0.0 to 1.0, CRT scanline strength
uniform float uNightMode;           // 0.0 or 1.0, night enhancement toggle

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const float SOBEL_THRESHOLD  = 0.15;  // Edge detection sensitivity
const float GLOW_RADIUS      = 2.0;   // Glow blur radius in texels (1-2px)
const float GLOW_INTENSITY   = 0.35;  // Glow brightness multiplier
const float OUTLINE_ALPHA    = 0.92;  // Outline opacity
const float SCANLINE_FREQ    = 800.0; // Scanline spatial frequency
const float SCANLINE_DARK    = 0.06;  // Max scanline darkening factor
const float PULSE_SPEED      = 2.5;   // Outline pulse animation speed
const float PULSE_AMP        = 0.08;  // Outline pulse amplitude

// ---------------------------------------------------------------------------
// Sample mask at given offset (returns single-channel float)
// ---------------------------------------------------------------------------
float sampleMask(vec2 uv) {
    return texture(uMaskTexture, uv).r;
}

// ---------------------------------------------------------------------------
// Sobel edge detection on segmentation mask
// Returns edge magnitude [0.0, ~4.0]
// ---------------------------------------------------------------------------
float sobelEdge(vec2 uv) {
    // Sample 3x3 neighborhood
    float tl = sampleMask(uv + vec2(-uTexelSize.x,  uTexelSize.y));
    float t  = sampleMask(uv + vec2( 0.0,           uTexelSize.y));
    float tr = sampleMask(uv + vec2( uTexelSize.x,  uTexelSize.y));
    float l  = sampleMask(uv + vec2(-uTexelSize.x,  0.0));
    float r  = sampleMask(uv + vec2( uTexelSize.x,  0.0));
    float bl = sampleMask(uv + vec2(-uTexelSize.x, -uTexelSize.y));
    float b  = sampleMask(uv + vec2( 0.0,          -uTexelSize.y));
    float br = sampleMask(uv + vec2( uTexelSize.x, -uTexelSize.y));

    // Sobel kernels
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;

    return sqrt(gx * gx + gy * gy);
}

// ---------------------------------------------------------------------------
// Compute glow by sampling edges in a small kernel around the fragment
// Uses a 3x3 box blur of edge values for a subtle 1-2px glow
// ---------------------------------------------------------------------------
float computeGlow(vec2 uv) {
    float glowSum = 0.0;
    float totalWeight = 0.0;

    // 5x5 Gaussian-like sampling for soft glow
    for (float dy = -GLOW_RADIUS; dy <= GLOW_RADIUS; dy += 1.0) {
        for (float dx = -GLOW_RADIUS; dx <= GLOW_RADIUS; dx += 1.0) {
            float dist = length(vec2(dx, dy));
            if (dist > GLOW_RADIUS) continue;

            float weight = 1.0 - (dist / (GLOW_RADIUS + 0.001));
            weight = weight * weight; // Quadratic falloff

            vec2 offsetUV = uv + vec2(dx, dy) * uTexelSize;
            float edge = sobelEdge(offsetUV);
            float mask = sampleMask(offsetUV);

            // Only glow where there are edges on valid mask regions
            float edgeContrib = step(SOBEL_THRESHOLD, edge) * step(0.01, mask);
            glowSum += edgeContrib * weight;
            totalWeight += weight;
        }
    }

    return (totalWeight > 0.0) ? (glowSum / totalWeight) : 0.0;
}

// ---------------------------------------------------------------------------
// CRT scanline effect — subtle darkened horizontal lines
// ---------------------------------------------------------------------------
float scanlineEffect(vec2 uv) {
    // Screen-space Y coordinate for scanline pattern
    float screenY = uv.y * SCANLINE_FREQ;
    float scanline = sin(screenY * 3.14159265) * 0.5 + 0.5;

    // Make it subtle — only darken slightly
    float darkening = mix(1.0, 1.0 - SCANLINE_DARK, scanline * uScanlineIntensity);

    return darkening;
}

// ---------------------------------------------------------------------------
// Night mode — brightness boost and slight warm tint
// ---------------------------------------------------------------------------
vec3 applyNightMode(vec3 color) {
    // Boost brightness
    float luminance = dot(color, vec3(0.299, 0.587, 0.114));
    float boost = 1.0 + 0.4 * uNightMode * (1.0 - luminance);
    color *= boost;

    // Slight warm tint for night mode to reduce blue light
    color.r += 0.03 * uNightMode;
    color.g += 0.01 * uNightMode;
    color.b -= 0.02 * uNightMode;

    return clamp(color, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
void main() {
    // 1. Sample camera feed
    vec3 cameraColor = texture(uCameraTexture, vTexCoord).rgb;

    // 2. Sample segmentation mask
    float maskValue = sampleMask(vTexCoord);

    // 3. Sobel edge detection on mask
    float edgeMagnitude = sobelEdge(vTexCoord);

    // 4. Determine if this pixel is on an outline
    bool isEdge = edgeMagnitude > SOBEL_THRESHOLD && maskValue > 0.01;

    // 5. Compute glow around edges (only if near an edge to save perf)
    float glow = 0.0;
    if (!isEdge) {
        // Only compute glow for non-edge pixels that might be near an edge
        // Quick check: sample neighbors for edges
        float nearEdge = 0.0;
        nearEdge += step(SOBEL_THRESHOLD, sobelEdge(vTexCoord + vec2(-uTexelSize.x, 0.0)));
        nearEdge += step(SOBEL_THRESHOLD, sobelEdge(vTexCoord + vec2( uTexelSize.x, 0.0)));
        nearEdge += step(SOBEL_THRESHOLD, sobelEdge(vTexCoord + vec2(0.0, -uTexelSize.y)));
        nearEdge += step(SOBEL_THRESHOLD, sobelEdge(vTexCoord + vec2(0.0,  uTexelSize.y)));

        if (nearEdge > 0.5) {
            glow = computeGlow(vTexCoord);
        }
    }

    // 6. Pulse animation on the outline brightness
    float pulse = 1.0 + PULSE_AMP * sin(uTime * PULSE_SPEED);

    // 7. Composite: camera + green outline
    vec3 finalColor = cameraColor;

    if (isEdge) {
        // Direct outline — bright terminal green with pulse
        float edgeIntensity = min(edgeMagnitude / 1.5, 1.0) * pulse;
        vec3 outlineColor = uGreenColor * edgeIntensity;
        finalColor = mix(cameraColor, outlineColor, OUTLINE_ALPHA);
    } else if (glow > 0.01) {
        // Glow around outline — softer, more transparent green
        vec3 glowColor = uGreenColor * GLOW_INTENSITY * glow;
        finalColor = mix(finalColor, finalColor + glowColor, glow * 0.7);
    }

    // 8. Composite text labels on top
    vec4 labelSample = texture(uLabelTexture, vTexCoord);
    if (labelSample.a > 0.01) {
        // Label text in terminal green, background slightly transparent
        vec3 labelColor = labelSample.rgb * uGreenColor;
        // Slight alpha premultiply for cleaner blending
        finalColor = mix(finalColor, labelColor, labelSample.a * 0.9);
    }

    // 9. Apply scanline effect
    float scanline = scanlineEffect(vTexCoord);
    finalColor *= scanline;

    // 10. Apply night mode
    finalColor = applyNightMode(finalColor);

    // Output with full opacity
    fragColor = vec4(clamp(finalColor, 0.0, 1.0), 1.0);
}
