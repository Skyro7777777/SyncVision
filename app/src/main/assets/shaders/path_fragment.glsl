// ============================================================================
// path_fragment.glsl
// Sync Vision — Path Visualization Fragment Shader
//
// Fragment shader for rendering path visualization with:
//   - Green solid lines for CLEAR path
//   - Green dashed lines for PARTIAL OBSTRUCTION
//   - Red-tinted lines for BLOCKED path
//   - Arrow head rendering with directional indicators
//   - Animated dash pattern (dashes flow along the path direction)
//   - Waypoint dots at path decision points
//
// Color scheme:
//   CLEAR   → Terminal green (#00FF41) solid line
//   PARTIAL → Terminal green dashed line (animated)
//   BLOCKED → Red-tinted (#FF3040) solid or dashed line
//
// The animated dash pattern creates a "marching ants" effect that
// indicates direction of travel along the path.
// ============================================================================

#version 300 es
precision mediump float;

// Varying input from vertex shader
in float vSegmentType;       // 0=solid, 1=dashed, 2=arrow, 3=waypoint
in float vPathStatus;        // 0=CLEAR, 1=PARTIAL, 2=BLOCKED
in float vParametricDist;    // Distance along path [0,1]
in float vWidthScale;        // Line width / point size scale
in vec2  vScreenPos;         // Screen-space position for AA

// Output
layout(location = 0) out vec4 fragColor;

// Uniforms
uniform vec3  uGreenColor;      // Terminal green (#00FF41)
uniform float uTime;            // Time for dash animation

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const vec3  RED_COLOR        = vec3(1.0, 0.188, 0.251);  // #FF3040
const vec3  PARTIAL_COLOR    = vec3(0.0, 0.8, 0.2);      // Dimmer green for partial
const float DASH_LENGTH      = 0.06;  // Dash length in parametric space
const float GAP_LENGTH       = 0.04;  // Gap length in parametric space
const float DASH_SPEED       = 0.4;   // Dash animation speed
const float ARROW_ALPHA      = 0.85;  // Arrow head opacity
const float LINE_ALPHA       = 0.80;  // Path line opacity
const float WAYPOINT_RADIUS  = 4.0;   // Waypoint dot radius in pixels
const float WAYPOINT_ALPHA   = 0.75;  // Waypoint opacity

// ---------------------------------------------------------------------------
// Animated dash pattern — returns 1.0 for dash, 0.0 for gap
// Uses parametric distance along path + time for flowing animation
// ---------------------------------------------------------------------------
float dashPattern(float t) {
    float totalLength = DASH_LENGTH + GAP_LENGTH;
    // Animate: shift the pattern along the path over time
    float phase = t - uTime * DASH_SPEED;
    // Wrap phase to [0, totalLength]
    phase = mod(phase, totalLength);
    return step(phase, DASH_LENGTH);
}

// ---------------------------------------------------------------------------
// Select color based on path status
// ---------------------------------------------------------------------------
vec3 selectColor() {
    if (vPathStatus < 0.5) {
        // CLEAR — bright terminal green
        return uGreenColor;
    } else if (vPathStatus < 1.5) {
        // PARTIAL — dimmer green
        return PARTIAL_COLOR;
    } else {
        // BLOCKED — red
        return RED_COLOR;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
void main() {
    vec3 pathColor = selectColor();

    // --- Solid path lines (CLEAR path) ---
    if (vSegmentType < 0.5) {
        fragColor = vec4(pathColor, LINE_ALPHA);
        return;
    }

    // --- Dashed path lines (PARTIAL OBSTRUCTION) ---
    if (vSegmentType < 1.5) {
        float dash = dashPattern(vParametricDist);
        if (dash < 0.5) {
            // Gap — discard or transparent
            discard;
        }
        fragColor = vec4(pathColor, LINE_ALPHA);
        return;
    }

    // --- Arrow heads ---
    if (vSegmentType < 2.5) {
        // Arrow heads are rendered as triangles; just fill with the path color
        // Add a slight brightness boost for visibility
        vec3 arrowColor = pathColor * 1.15;
        fragColor = vec4(arrowColor, ARROW_ALPHA);
        return;
    }

    // --- Waypoint dots ---
    if (vSegmentType < 3.5) {
        // Waypoints rendered as point sprites; use a circular mask
        // gl_PointCoord is available for GL_POINTS rendering
        // For now, just output a solid dot (the point sprite handling
        // can be done in a separate pass or via gl_PointSize)
        vec3 dotColor = pathColor * 1.1;
        fragColor = vec4(dotColor, WAYPOINT_ALPHA);
        return;
    }

    // Fallback — should not reach here
    fragColor = vec4(pathColor, 0.5);
}
