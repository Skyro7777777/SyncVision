/**
 * ShaderProgram.java
 *
 * Utility class for compiling and linking OpenGL ES 3.0 GLSL shaders.
 * Loads shader source from the assets/shaders/ directory, compiles
 * vertex and fragment shaders, links them into a program, and provides
 * convenience methods for setting uniform values.
 *
 * All shader files are expected to be OpenGL ES 3.0 compatible
 * (#version 300 es).
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.rendering.shader
 * Target SDK: 29+
 */

package com.syncvision.app.rendering.shader;

import android.content.Context;
import android.content.res.AssetManager;
import android.opengl.GLES30;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

/**
 * Compiles, links, and manages an OpenGL ES 3.0 shader program.
 * <p>
 * Usage:
 * <pre>
 *   ShaderProgram program = new ShaderProgram(context);
 *   int programId = program.create("shaders/composite_vertex.glsl",
 *                                  "shaders/composite_fragment.glsl");
 *   program.use();
 *   program.setUniform("uTime", timeSeconds);
 * </pre>
 */
public class ShaderProgram {

    private static final String TAG = "SV-ShaderProgram";

    /** Application context for loading shader assets. */
    private final Context context;

    /** The linked GL program handle. 0 if not yet created. */
    private int programHandle = 0;

    /** Handle to the compiled vertex shader. */
    private int vertexShaderHandle = 0;

    /** Handle to the compiled fragment shader. */
    private int fragmentShaderHandle = 0;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new ShaderProgram with the given context for asset access.
     *
     * @param context Application or activity context (used for asset loading).
     */
    public ShaderProgram(@NonNull Context context) {
        this.context = context.getApplicationContext();
    }

    // ================================================================
    // Program Creation
    // ================================================================

    /**
     * Creates a complete shader program from the given vertex and fragment
     * shader asset paths.
     *
     * @param vertexAssetPath   Path to the vertex shader in assets
     *                          (e.g., "shaders/composite_vertex.glsl").
     * @param fragmentAssetPath Path to the fragment shader in assets
     *                          (e.g., "shaders/composite_fragment.glsl").
     * @return The linked program handle, or 0 on failure.
     */
    public int create(@NonNull String vertexAssetPath,
                      @NonNull String fragmentAssetPath) {
        // Load shader source code from assets
        String vertexSource = loadShaderSource(vertexAssetPath);
        String fragmentSource = loadShaderSource(fragmentAssetPath);

        if (vertexSource == null || fragmentSource == null) {
            Log.e(TAG, "Failed to load shader sources: vertex="
                    + vertexAssetPath + ", fragment=" + fragmentAssetPath);
            return 0;
        }

        // Compile shaders
        vertexShaderHandle = compileShader(GLES30.GL_VERTEX_SHADER, vertexSource);
        if (vertexShaderHandle == 0) {
            Log.e(TAG, "Vertex shader compilation failed: " + vertexAssetPath);
            return 0;
        }

        fragmentShaderHandle = compileShader(GLES30.GL_FRAGMENT_SHADER, fragmentSource);
        if (fragmentShaderHandle == 0) {
            Log.e(TAG, "Fragment shader compilation failed: " + fragmentAssetPath);
            GLES30.glDeleteShader(vertexShaderHandle);
            vertexShaderHandle = 0;
            return 0;
        }

        // Link program
        programHandle = linkProgram(vertexShaderHandle, fragmentShaderHandle);
        if (programHandle == 0) {
            Log.e(TAG, "Program linking failed for: " + vertexAssetPath);
            cleanup();
            return 0;
        }

        Log.i(TAG, "Shader program created successfully: " + vertexAssetPath
                + " (handle=" + programHandle + ")");
        return programHandle;
    }

    // ================================================================
    // Program Usage
    // ================================================================

    /**
     * Sets this program as the active shader program.
     * Equivalent to glUseProgram(programHandle).
     */
    public void use() {
        if (programHandle != 0) {
            GLES30.glUseProgram(programHandle);
        } else {
            Log.w(TAG, "Attempted to use invalid shader program");
        }
    }

    /**
     * Returns the OpenGL program handle.
     *
     * @return The program handle, or 0 if not created.
     */
    public int getHandle() {
        return programHandle;
    }

    /**
     * Returns whether the program has been successfully created.
     */
    public boolean isValid() {
        return programHandle != 0;
    }

    // ================================================================
    // Attribute & Uniform Locations
    // ================================================================

    /**
     * Gets the location of an attribute variable in the shader program.
     *
     * @param name The attribute name as declared in the vertex shader.
     * @return The attribute location, or -1 if not found.
     */
    public int getAttributeLocation(@NonNull String name) {
        if (programHandle == 0) {
            Log.w(TAG, "getAttributeLocation called on invalid program");
            return -1;
        }
        int loc = GLES30.glGetAttribLocation(programHandle, name);
        if (loc == -1) {
            // Not necessarily an error — the attribute may be optimized away
            Log.d(TAG, "Attribute '" + name + "' not found (may be optimized out)");
        }
        return loc;
    }

    /**
     * Gets the location of a uniform variable in the shader program.
     *
     * @param name The uniform name as declared in the shader.
     * @return The uniform location, or -1 if not found.
     */
    public int getUniformLocation(@NonNull String name) {
        if (programHandle == 0) {
            Log.w(TAG, "getUniformLocation called on invalid program");
            return -1;
        }
        int loc = GLES30.glGetUniformLocation(programHandle, name);
        if (loc == -1) {
            // Not necessarily an error — the uniform may be optimized away
            Log.d(TAG, "Uniform '" + name + "' not found (may be optimized out)");
        }
        return loc;
    }

    // ================================================================
    // Uniform Setters
    // ================================================================

    /**
     * Sets a float uniform value.
     *
     * @param name  The uniform name.
     * @param value The float value.
     */
    public void setUniform(@NonNull String name, float value) {
        int loc = getUniformLocation(name);
        if (loc != -1) {
            GLES30.glUniform1f(loc, value);
        }
    }

    /**
     * Sets an integer uniform value.
     *
     * @param name  The uniform name.
     * @param value The integer value.
     */
    public void setUniform(@NonNull String name, int value) {
        int loc = getUniformLocation(name);
        if (loc != -1) {
            GLES30.glUniform1i(loc, value);
        }
    }

    /**
     * Sets a vec2 uniform value.
     *
     * @param name The uniform name.
     * @param v0   First component.
     * @param v1   Second component.
     */
    public void setUniform(@NonNull String name, float v0, float v1) {
        int loc = getUniformLocation(name);
        if (loc != -1) {
            GLES30.glUniform2f(loc, v0, v1);
        }
    }

    /**
     * Sets a vec3 uniform value.
     *
     * @param name The uniform name.
     * @param v0   First component.
     * @param v1   Second component.
     * @param v2   Third component.
     */
    public void setUniform(@NonNull String name, float v0, float v1, float v2) {
        int loc = getUniformLocation(name);
        if (loc != -1) {
            GLES30.glUniform3f(loc, v0, v1, v2);
        }
    }

    /**
     * Sets a vec4 uniform value.
     *
     * @param name The uniform name.
     * @param v0   First component.
     * @param v1   Second component.
     * @param v2   Third component.
     * @param v3   Fourth component.
     */
    public void setUniform(@NonNull String name, float v0, float v1, float v2, float v3) {
        int loc = getUniformLocation(name);
        if (loc != -1) {
            GLES30.glUniform4f(loc, v0, v1, v2, v3);
        }
    }

    /**
     * Sets a vec2 uniform from a float array.
     *
     * @param name   The uniform name.
     * @param values Array of at least 2 floats.
     * @param offset Starting offset in the array.
     */
    public void setUniform2fv(@NonNull String name, @NonNull float[] values, int offset) {
        int loc = getUniformLocation(name);
        if (loc != -1) {
            GLES30.glUniform2fv(loc, 1, values, offset);
        }
    }

    /**
     * Sets a vec3 uniform from a float array.
     *
     * @param name   The uniform name.
     * @param values Array of at least 3 floats.
     * @param offset Starting offset in the array.
     */
    public void setUniform3fv(@NonNull String name, @NonNull float[] values, int offset) {
        int loc = getUniformLocation(name);
        if (loc != -1) {
            GLES30.glUniform3fv(loc, 1, values, offset);
        }
    }

    /**
     * Sets a vec4 uniform from a float array.
     *
     * @param name   The uniform name.
     * @param values Array of at least 4 floats.
     * @param offset Starting offset in the array.
     */
    public void setUniform4fv(@NonNull String name, @NonNull float[] values, int offset) {
        int loc = getUniformLocation(name);
        if (loc != -1) {
            GLES30.glUniform4fv(loc, 1, values, offset);
        }
    }

    /**
     * Sets a mat4 uniform from a float array (column-major).
     *
     * @param name   The uniform name.
     * @param matrix 16-float column-major matrix.
     * @param offset Starting offset in the array.
     */
    public void setUniformMatrix4fv(@NonNull String name, @NonNull float[] matrix, int offset) {
        int loc = getUniformLocation(name);
        if (loc != -1) {
            GLES30.glUniformMatrix4fv(loc, 1, false, matrix, offset);
        }
    }

    // ================================================================
    // Cleanup
    // ================================================================

    /**
     * Deletes the shader program and associated shader objects.
     * Call this when the program is no longer needed (e.g., on surface destroy).
     */
    public void cleanup() {
        if (programHandle != 0) {
            GLES30.glDeleteProgram(programHandle);
            programHandle = 0;
        }
        if (vertexShaderHandle != 0) {
            GLES30.glDeleteShader(vertexShaderHandle);
            vertexShaderHandle = 0;
        }
        if (fragmentShaderHandle != 0) {
            GLES30.glDeleteShader(fragmentShaderHandle);
            fragmentShaderHandle = 0;
        }
    }

    // ================================================================
    // Internal — Shader Compilation
    // ================================================================

    /**
     * Compiles a single shader from source code.
     *
     * @param shaderType   GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.
     * @param shaderSource The GLSL source code string.
     * @return The compiled shader handle, or 0 on failure.
     */
    private int compileShader(int shaderType, @NonNull String shaderSource) {
        int shader = GLES30.glCreateShader(shaderType);
        if (shader == 0) {
            Log.e(TAG, "glCreateShader failed for type " + shaderType);
            return 0;
        }

        GLES30.glShaderSource(shader, shaderSource);
        GLES30.glCompileShader(shader);

        // Check compilation status
        int[] compileStatus = new int[1];
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, compileStatus, 0);

        if (compileStatus[0] == GLES30.GL_FALSE) {
            String log = GLES30.glGetShaderInfoLog(shader);
            Log.e(TAG, "Shader compilation failed:\n" + log);
            Log.e(TAG, "Source:\n" + shaderSource);
            GLES30.glDeleteShader(shader);
            return 0;
        }

        return shader;
    }

    /**
     * Links a vertex and fragment shader into a program.
     *
     * @param vertexShaderHandle   Compiled vertex shader handle.
     * @param fragmentShaderHandle Compiled fragment shader handle.
     * @return The linked program handle, or 0 on failure.
     */
    private int linkProgram(int vertexShaderHandle, int fragmentShaderHandle) {
        int program = GLES30.glCreateProgram();
        if (program == 0) {
            Log.e(TAG, "glCreateProgram failed");
            return 0;
        }

        GLES30.glAttachShader(program, vertexShaderHandle);
        GLES30.glAttachShader(program, fragmentShaderHandle);
        GLES30.glLinkProgram(program);

        // Check link status
        int[] linkStatus = new int[1];
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, linkStatus, 0);

        if (linkStatus[0] == GLES30.GL_FALSE) {
            String log = GLES30.glGetProgramInfoLog(program);
            Log.e(TAG, "Program linking failed:\n" + log);
            GLES30.glDeleteProgram(program);
            return 0;
        }

        // Shaders can be detached after linking (they're still valid until deleted)
        GLES30.glDetachShader(program, vertexShaderHandle);
        GLES30.glDetachShader(program, fragmentShaderHandle);

        return program;
    }

    /**
     * Loads shader source code from the assets directory.
     *
     * @param assetPath Path relative to the assets folder
     *                  (e.g., "shaders/composite_vertex.glsl").
     * @return The shader source as a single String, or null on failure.
     */
    @Nullable
    private String loadShaderSource(@NonNull String assetPath) {
        AssetManager assets = context.getAssets();
        StringBuilder source = new StringBuilder();

        try (InputStream is = assets.open(assetPath);
             BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {

            String line;
            while ((line = reader.readLine()) != null) {
                source.append(line).append('\n');
            }

            return source.toString();

        } catch (IOException e) {
            Log.e(TAG, "Failed to load shader from assets: " + assetPath, e);
            return null;
        }
    }
}
