/**
 * LabelShader.java
 *
 * Wrapper for the label rendering shader program. This shader renders
 * text label quads with a pre-rendered label atlas texture. Each label
 * appears as a semi-transparent dark panel with terminal green text
 * in ALL CAPS, complete with text shadow and border glow.
 *
 * Supports instanced rendering for efficient multi-label drawing.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.rendering.shader
 * Target SDK: 29+
 */

package com.syncvision.app.rendering.shader;

import android.content.Context;
import android.opengl.GLES30;
import android.util.Log;

import androidx.annotation.NonNull;

/**
 * Typed wrapper around the label shader program.
 * <p>
 * The label shader renders text labels as screen-space quads. Each label
 * quad has position and size specified per-instance, and text content
 * is sampled from a pre-rendered label atlas texture.
 * <p>
 * Shader uniforms:
 *   - uLabelAtlas (sampler2D): Pre-rendered label texture atlas
 *   - uGreenColor (vec3):      Terminal green (#00FF41)
 *   - uTime       (float):     Time for fade-in animation
 *   - uAtlasRect  (vec4):      Atlas sub-rect: x,y = top-left UV, z,w = size UV
 * <p>
 * Shader attributes:
 *   - aPosition   (vec4, location 0): x, y, width, height of label quad
 *   - aQuadVertex (vec2, location 1): Unit quad corner (0,0), (1,0), (0,1), (1,1)
 */
public class LabelShader {

    private static final String TAG = "SV-LabelShader";

    /** Asset paths for the label shader pair. */
    private static final String VERTEX_PATH   = "shaders/label_vertex.glsl";
    private static final String FRAGMENT_PATH = "shaders/label_fragment.glsl";

    // -----------------------------------------------------------------
    // Attribute locations (fixed in shader via layout(location=N))
    // -----------------------------------------------------------------

    /** Label position/size attribute (layout(location=0)). */
    public static final int ATTR_POSITION = 0;

    /** Quad vertex attribute (layout(location=1)). */
    public static final int ATTR_QUAD_VERTEX = 1;

    // -----------------------------------------------------------------
    // Uniform location cache
    // -----------------------------------------------------------------

    private int uLabelAtlas = -1;
    private int uGreenColor = -1;
    private int uTime       = -1;
    private int uAtlasRect  = -1;

    // -----------------------------------------------------------------
    // Internal state
    // -----------------------------------------------------------------

    /** The underlying ShaderProgram utility. */
    private final ShaderProgram shaderProgram;

    /** Whether the program has been successfully created. */
    private boolean initialized = false;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new LabelShader wrapper.
     *
     * @param context Application context for loading shader assets.
     */
    public LabelShader(@NonNull Context context) {
        this.shaderProgram = new ShaderProgram(context);
    }

    // ================================================================
    // Initialization
    // ================================================================

    /**
     * Compiles and links the label shader program.
     * Must be called on the GL thread.
     *
     * @return true if initialization succeeded.
     */
    public boolean init() {
        int handle = shaderProgram.create(VERTEX_PATH, FRAGMENT_PATH);
        if (handle == 0) {
            Log.e(TAG, "Failed to create label shader program");
            return false;
        }

        // Cache uniform locations
        uLabelAtlas = shaderProgram.getUniformLocation("uLabelAtlas");
        uGreenColor = shaderProgram.getUniformLocation("uGreenColor");
        uTime       = shaderProgram.getUniformLocation("uTime");
        uAtlasRect  = shaderProgram.getUniformLocation("uAtlasRect");

        initialized = true;
        Log.i(TAG, "LabelShader initialized (program=" + handle + ")");
        return true;
    }

    // ================================================================
    // Activation
    // ================================================================

    /**
     * Activates this shader program for rendering.
     */
    public void use() {
        if (initialized) {
            shaderProgram.use();
        }
    }

    // ================================================================
    // Uniform Setters
    // ================================================================

    /**
     * Sets the label atlas texture unit.
     *
     * @param unit The texture unit index (0-based).
     */
    public void setLabelAtlas(int unit) {
        if (uLabelAtlas != -1) {
            GLES30.glUniform1i(uLabelAtlas, unit);
        }
    }

    /**
     * Sets the terminal green text color.
     * Default: #00FF41 = vec3(0.0, 1.0, 0.255).
     *
     * @param r Red component [0, 1].
     * @param g Green component [0, 1].
     * @param b Blue component [0, 1].
     */
    public void setGreenColor(float r, float g, float b) {
        if (uGreenColor != -1) {
            GLES30.glUniform3f(uGreenColor, r, g, b);
        }
    }

    /**
     * Sets the green text color from a float array.
     *
     * @param rgb 3-float array with [r, g, b].
     */
    public void setGreenColor(@NonNull float[] rgb) {
        if (uGreenColor != -1 && rgb.length >= 3) {
            GLES30.glUniform3fv(uGreenColor, 1, rgb, 0);
        }
    }

    /**
     * Sets the elapsed time for the fade-in animation.
     *
     * @param timeSeconds Time in seconds.
     */
    public void setTime(float timeSeconds) {
        if (uTime != -1) {
            GLES30.glUniform1f(uTime, timeSeconds);
        }
    }

    /**
     * Sets the atlas sub-rect for the current label being rendered.
     * This maps the label quad's UV coordinates to the correct
     * region within the label atlas texture.
     *
     * @param u  Top-left U coordinate in the atlas.
     * @param v  Top-left V coordinate in the atlas.
     * @param du Width of the sub-rect in UV space.
     * @param dv Height of the sub-rect in UV space.
     */
    public void setAtlasRect(float u, float v, float du, float dv) {
        if (uAtlasRect != -1) {
            GLES30.glUniform4f(uAtlasRect, u, v, du, dv);
        }
    }

    /**
     * Sets the atlas sub-rect from a float array.
     *
     * @param rect 4-float array: [u, v, du, dv].
     */
    public void setAtlasRect(@NonNull float[] rect) {
        if (uAtlasRect != -1 && rect.length >= 4) {
            GLES30.glUniform4fv(uAtlasRect, 1, rect, 0);
        }
    }

    // ================================================================
    // Attribute Setup Helpers
    // ================================================================

    /**
     * Enables the vertex attribute arrays for this shader.
     */
    public void enableAttributes() {
        GLES30.glEnableVertexAttribArray(ATTR_POSITION);
        GLES30.glEnableVertexAttribArray(ATTR_QUAD_VERTEX);
    }

    /**
     * Disables the vertex attribute arrays for this shader.
     */
    public void disableAttributes() {
        GLES30.glDisableVertexAttribArray(ATTR_POSITION);
        GLES30.glDisableVertexAttribArray(ATTR_QUAD_VERTEX);
    }

    // ================================================================
    // Cleanup
    // ================================================================

    /**
     * Releases the shader program and all associated GL resources.
     * Must be called on the GL thread.
     */
    public void cleanup() {
        if (shaderProgram != null) {
            shaderProgram.cleanup();
        }
        initialized = false;
        uLabelAtlas = -1;
        uGreenColor = -1;
        uTime       = -1;
        uAtlasRect  = -1;
    }

    /**
     * Returns whether the shader program is initialized and ready to use.
     */
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Returns the underlying program handle for advanced usage.
     */
    public int getProgramHandle() {
        return shaderProgram.getHandle();
    }
}
