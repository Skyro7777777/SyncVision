/**
 * CompositeShader.java
 *
 * Wrapper for the composite shader program — the primary rendering
 * shader that combines the camera feed, segmentation mask outlines,
 * text labels, depth data, and visual effects (scanlines, night mode).
 *
 * Wraps all uniform and attribute locations for the composite_vertex.glsl
 * and composite_fragment.glsl pair. Provides typed setter methods for
 * each uniform to avoid manual location lookups at draw time.
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
 * Typed wrapper around the composite shader program.
 * <p>
 * The composite shader is the most important shader in the pipeline.
 * It performs Sobel edge detection on the segmentation mask to produce
 * the signature green outlines, composites text labels, and applies
 * CRT scanline and night mode effects.
 * <p>
 * Shader uniforms:
 *   - uCameraTexture (sampler2D): Camera feed (GL_TEXTURE_EXTERNAL_OES)
 *   - uMaskTexture   (sampler2D): Segmentation mask (R8)
 *   - uLabelTexture  (sampler2D): Label overlay (RGBA)
 *   - uDepthTexture  (sampler2D): Depth map (R16F)
 *   - uTexelSize     (vec2):      1.0/width, 1.0/height
 *   - uGreenColor    (vec3):      Terminal green (#00FF41 = 0.0, 1.0, 0.255)
 *   - uTime          (float):     Elapsed time for animations
 *   - uScanlineIntensity (float): CRT scanline strength [0, 1]
 *   - uNightMode     (float):     Night mode toggle (0.0 or 1.0)
 * <p>
 * Shader attributes:
 *   - aPosition (vec2, location 0): NDC coordinates
 *   - aTexCoord (vec2, location 1): Texture coordinates
 */
public class CompositeShader {

    private static final String TAG = "SV-CompositeShader";

    /** Asset paths for the composite shader pair. */
    private static final String VERTEX_PATH   = "shaders/composite_vertex.glsl";
    private static final String FRAGMENT_PATH = "shaders/composite_fragment.glsl";

    // -----------------------------------------------------------------
    // Attribute locations (fixed in shader via layout(location=N))
    // -----------------------------------------------------------------

    /** Position attribute location (layout(location=0) in shader). */
    public static final int ATTR_POSITION = 0;

    /** Texture coordinate attribute location (layout(location=1) in shader). */
    public static final int ATTR_TEX_COORD = 1;

    // -----------------------------------------------------------------
    // Uniform location cache
    // -----------------------------------------------------------------

    private int uCameraTexture     = -1;
    private int uMaskTexture       = -1;
    private int uLabelTexture      = -1;
    private int uDepthTexture      = -1;
    private int uTexelSize         = -1;
    private int uGreenColor        = -1;
    private int uTime              = -1;
    private int uScanlineIntensity = -1;
    private int uNightMode         = -1;

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
     * Creates a new CompositeShader wrapper.
     *
     * @param context Application context for loading shader assets.
     */
    public CompositeShader(@NonNull Context context) {
        this.shaderProgram = new ShaderProgram(context);
    }

    // ================================================================
    // Initialization
    // ================================================================

    /**
     * Compiles and links the composite shader program.
     * Must be called on the GL thread (typically in onSurfaceCreated).
     *
     * @return true if initialization succeeded.
     */
    public boolean init() {
        int handle = shaderProgram.create(VERTEX_PATH, FRAGMENT_PATH);
        if (handle == 0) {
            Log.e(TAG, "Failed to create composite shader program");
            return false;
        }

        // Cache all uniform locations
        uCameraTexture     = shaderProgram.getUniformLocation("uCameraTexture");
        uMaskTexture       = shaderProgram.getUniformLocation("uMaskTexture");
        uLabelTexture      = shaderProgram.getUniformLocation("uLabelTexture");
        uDepthTexture      = shaderProgram.getUniformLocation("uDepthTexture");
        uTexelSize         = shaderProgram.getUniformLocation("uTexelSize");
        uGreenColor        = shaderProgram.getUniformLocation("uGreenColor");
        uTime              = shaderProgram.getUniformLocation("uTime");
        uScanlineIntensity = shaderProgram.getUniformLocation("uScanlineIntensity");
        uNightMode         = shaderProgram.getUniformLocation("uNightMode");

        initialized = true;
        Log.i(TAG, "CompositeShader initialized (program=" + handle + ")");
        return true;
    }

    // ================================================================
    // Activation
    // ================================================================

    /**
     * Activates this shader program for rendering.
     * Equivalent to glUseProgram().
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
     * Sets the camera texture unit (e.g., 0 for GL_TEXTURE0).
     *
     * @param unit The texture unit index (0-based).
     */
    public void setCameraTexture(int unit) {
        if (uCameraTexture != -1) {
            GLES30.glUniform1i(uCameraTexture, unit);
        }
    }

    /**
     * Sets the segmentation mask texture unit.
     *
     * @param unit The texture unit index (0-based).
     */
    public void setMaskTexture(int unit) {
        if (uMaskTexture != -1) {
            GLES30.glUniform1i(uMaskTexture, unit);
        }
    }

    /**
     * Sets the label overlay texture unit.
     *
     * @param unit The texture unit index (0-based).
     */
    public void setLabelTexture(int unit) {
        if (uLabelTexture != -1) {
            GLES30.glUniform1i(uLabelTexture, unit);
        }
    }

    /**
     * Sets the depth map texture unit.
     *
     * @param unit The texture unit index (0-based).
     */
    public void setDepthTexture(int unit) {
        if (uDepthTexture != -1) {
            GLES30.glUniform1i(uDepthTexture, unit);
        }
    }

    /**
     * Sets the texel size uniform (1.0/width, 1.0/height).
     * Required for the Sobel edge detection kernel sampling.
     *
     * @param texelSizeX 1.0 / texture width.
     * @param texelSizeY 1.0 / texture height.
     */
    public void setTexelSize(float texelSizeX, float texelSizeY) {
        if (uTexelSize != -1) {
            GLES30.glUniform2f(uTexelSize, texelSizeX, texelSizeY);
        }
    }

    /**
     * Sets the green outline color uniform.
     * Default: terminal green #00FF41 = vec3(0.0, 1.0, 0.255).
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
     * Sets the elapsed time uniform for animation effects.
     *
     * @param timeSeconds Time in seconds since the surface was created.
     */
    public void setTime(float timeSeconds) {
        if (uTime != -1) {
            GLES30.glUniform1f(uTime, timeSeconds);
        }
    }

    /**
     * Sets the CRT scanline intensity.
     *
     * @param intensity Scanline strength [0.0 = off, 1.0 = maximum].
     */
    public void setScanlineIntensity(float intensity) {
        if (uScanlineIntensity != -1) {
            GLES30.glUniform1f(uScanlineIntensity, intensity);
        }
    }

    /**
     * Sets the night mode toggle.
     *
     * @param enabled 0.0 for off, 1.0 for on.
     */
    public void setNightMode(float enabled) {
        if (uNightMode != -1) {
            GLES30.glUniform1f(uNightMode, enabled);
        }
    }

    // ================================================================
    // Attribute Setup Helpers
    // ================================================================

    /**
     * Enables the vertex attribute arrays for this shader.
     * Call before drawing the fullscreen quad.
     */
    public void enableAttributes() {
        GLES30.glEnableVertexAttribArray(ATTR_POSITION);
        GLES30.glEnableVertexAttribArray(ATTR_TEX_COORD);
    }

    /**
     * Disables the vertex attribute arrays for this shader.
     * Call after drawing to restore GL state.
     */
    public void disableAttributes() {
        GLES30.glDisableVertexAttribArray(ATTR_POSITION);
        GLES30.glDisableVertexAttribArray(ATTR_TEX_COORD);
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
        uCameraTexture     = -1;
        uMaskTexture       = -1;
        uLabelTexture      = -1;
        uDepthTexture      = -1;
        uTexelSize         = -1;
        uGreenColor        = -1;
        uTime              = -1;
        uScanlineIntensity = -1;
        uNightMode         = -1;
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
