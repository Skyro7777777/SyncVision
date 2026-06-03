/**
 * OutlineShader.java
 *
 * Wrapper for the outline rendering shader program. This shader is used
 * for direct contour line rendering from pre-computed vertex data
 * (as opposed to the Sobel-based approach in the composite shader).
 *
 * Renders thin (1-2px) terminal green lines with configurable glow
 * intensity and a pulsing brightness animation.
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

import java.nio.FloatBuffer;

/**
 * Typed wrapper around the outline shader program.
 * <p>
 * The outline shader renders GL_LINES from pre-computed contour data,
 * producing green outlines with glow and pulse effects. This is an
 * alternative to the Sobel edge detection in the composite shader
 * for cases where exact contour geometry is available.
 * <p>
 * Shader uniforms:
 *   - uGreenColor    (vec3):  Terminal green (#00FF41)
 *   - uGlowIntensity (float): Glow strength [0.0, 1.0]
 *   - uTime          (float): Elapsed time for pulse animation
 *   - uResolution    (vec2):  Viewport resolution in pixels
 * <p>
 * Shader attributes:
 *   - aPosition (vec2, location 0): Line endpoint in normalized coords [0, 1]
 *   - aLineInfo (vec2, location 1): Segment index + parametric t
 */
public class OutlineShader {

    private static final String TAG = "SV-OutlineShader";

    /** Asset paths for the outline shader pair. */
    private static final String VERTEX_PATH   = "shaders/outline_vertex.glsl";
    private static final String FRAGMENT_PATH = "shaders/outline_fragment.glsl";

    // -----------------------------------------------------------------
    // Attribute locations (fixed in shader via layout(location=N))
    // -----------------------------------------------------------------

    /** Position attribute location (layout(location=0)). */
    public static final int ATTR_POSITION = 0;

    /** Line info attribute location (layout(location=1)). */
    public static final int ATTR_LINE_INFO = 1;

    // -----------------------------------------------------------------
    // Uniform location cache
    // -----------------------------------------------------------------

    private int uGreenColor    = -1;
    private int uGlowIntensity = -1;
    private int uTime          = -1;
    private int uResolution    = -1;

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
     * Creates a new OutlineShader wrapper.
     *
     * @param context Application context for loading shader assets.
     */
    public OutlineShader(@NonNull Context context) {
        this.shaderProgram = new ShaderProgram(context);
    }

    // ================================================================
    // Initialization
    // ================================================================

    /**
     * Compiles and links the outline shader program.
     * Must be called on the GL thread.
     *
     * @return true if initialization succeeded.
     */
    public boolean init() {
        int handle = shaderProgram.create(VERTEX_PATH, FRAGMENT_PATH);
        if (handle == 0) {
            Log.e(TAG, "Failed to create outline shader program");
            return false;
        }

        // Cache uniform locations
        uGreenColor    = shaderProgram.getUniformLocation("uGreenColor");
        uGlowIntensity = shaderProgram.getUniformLocation("uGlowIntensity");
        uTime          = shaderProgram.getUniformLocation("uTime");
        uResolution    = shaderProgram.getUniformLocation("uResolution");

        initialized = true;
        Log.i(TAG, "OutlineShader initialized (program=" + handle + ")");
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
     * Sets the terminal green outline color.
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
     * Sets the green outline color from a float array.
     *
     * @param rgb 3-float array with [r, g, b].
     */
    public void setGreenColor(@NonNull float[] rgb) {
        if (uGreenColor != -1 && rgb.length >= 3) {
            GLES30.glUniform3fv(uGreenColor, 1, rgb, 0);
        }
    }

    /**
     * Sets the glow intensity for the outline.
     * Controls the soft green halo around contour lines.
     *
     * @param intensity Glow strength [0.0 = no glow, 1.0 = maximum glow].
     */
    public void setGlowIntensity(float intensity) {
        if (uGlowIntensity != -1) {
            GLES30.glUniform1f(uGlowIntensity, intensity);
        }
    }

    /**
     * Sets the elapsed time for the pulse animation.
     *
     * @param timeSeconds Time in seconds.
     */
    public void setTime(float timeSeconds) {
        if (uTime != -1) {
            GLES30.glUniform1f(uTime, timeSeconds);
        }
    }

    /**
     * Sets the viewport resolution for screen-space calculations.
     *
     * @param width  Viewport width in pixels.
     * @param height Viewport height in pixels.
     */
    public void setResolution(float width, float height) {
        if (uResolution != -1) {
            GLES30.glUniform2f(uResolution, width, height);
        }
    }

    // ================================================================
    // Drawing
    // ================================================================

    /**
     * Draws contour lines from the given vertex buffer.
     * The buffer must contain interleaved [position(2), lineInfo(2)] data
     * — 4 floats per vertex, to be drawn as GL_LINES.
     *
     * @param vertexBuffer  FloatBuffer with interleaved vertex data.
     * @param vertexCount   Number of vertices to draw (must be even for GL_LINES).
     */
    public void drawContours(@NonNull FloatBuffer vertexBuffer, int vertexCount) {
        if (!initialized || vertexCount < 2) {
            return;
        }

        use();

        // Bind vertex data
        vertexBuffer.position(0);

        // Stride: 4 floats * 4 bytes = 16 bytes per vertex
        final int stride = 4 * 4;

        // aPosition: 2 floats at offset 0
        GLES30.glVertexAttribPointer(ATTR_POSITION, 2, GLES30.GL_FLOAT,
                false, stride, vertexBuffer);
        GLES30.glEnableVertexAttribArray(ATTR_POSITION);

        // aLineInfo: 2 floats at offset 8 bytes
        vertexBuffer.position(2);
        GLES30.glVertexAttribPointer(ATTR_LINE_INFO, 2, GLES30.GL_FLOAT,
                false, stride, vertexBuffer);
        GLES30.glEnableVertexAttribArray(ATTR_LINE_INFO);

        // Draw as lines
        GLES30.glDrawArrays(GLES30.GL_LINES, 0, vertexCount);

        // Disable attribute arrays
        GLES30.glDisableVertexAttribArray(ATTR_POSITION);
        GLES30.glDisableVertexAttribArray(ATTR_LINE_INFO);
    }

    // ================================================================
    // Attribute Setup Helpers
    // ================================================================

    /**
     * Enables the vertex attribute arrays for this shader.
     */
    public void enableAttributes() {
        GLES30.glEnableVertexAttribArray(ATTR_POSITION);
        GLES30.glEnableVertexAttribArray(ATTR_LINE_INFO);
    }

    /**
     * Disables the vertex attribute arrays for this shader.
     */
    public void disableAttributes() {
        GLES30.glDisableVertexAttribArray(ATTR_POSITION);
        GLES30.glDisableVertexAttribArray(ATTR_LINE_INFO);
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
        uGreenColor    = -1;
        uGlowIntensity = -1;
        uTime          = -1;
        uResolution    = -1;
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
