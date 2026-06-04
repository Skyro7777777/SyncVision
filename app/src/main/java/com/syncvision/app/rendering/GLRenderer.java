/**
 * GLRenderer.java
 *
 * The MAIN OpenGL ES 3.0 renderer for the Sync Vision app.
 * Implements GLSurfaceView.Renderer and manages the entire rendering
 * pipeline: camera feed, segmentation mask outlines, depth maps,
 * text labels, and path visualization.
 *
 * Rendering pipeline per frame (onDrawFrame):
 *   1. Update camera texture from SurfaceTexture
 *   2. Upload mask/depth/label textures if new data is available
 *   3. Run composite shader (Sobel edge detection on mask → green outline)
 *   4. Draw label overlay
 *   5. Draw path overlay (if path data exists)
 *
 * Textures:
 *   - cameraTexture: GL_TEXTURE_EXTERNAL_OES (from camera SurfaceTexture)
 *   - maskTexture:   GL_TEXTURE_2D (R8, from segmentation pipeline)
 *   - depthTexture:  GL_TEXTURE_2D (R16F, from depth pipeline)
 *   - labelTexture:  GL_TEXTURE_2D (RGBA, from Canvas-rendered Bitmap)
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.rendering
 * Target SDK: 29+
 */

package com.syncvision.app.rendering;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.opengl.GLES11Ext;
import android.opengl.GLES30;
import android.opengl.GLUtils;
import android.opengl.Matrix;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.syncvision.app.rendering.shader.CompositeShader;
import com.syncvision.app.rendering.shader.LabelShader;
import com.syncvision.app.rendering.shader.OutlineShader;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

/**
 * OpenGL ES 3.0 renderer for the Sync Vision AR overlay.
 * <p>
 * This renderer manages the complete visual pipeline from camera feed
 * to final composited output with green outlines, labels, and effects.
 * It creates and manages all GL resources and shader programs.
 */
public class GLRenderer implements android.opengl.GLSurfaceView.Renderer {

    private static final String TAG = "SV-GLRenderer";

    // -----------------------------------------------------------------
    // Terminal green color constants (#00FF41)
    // -----------------------------------------------------------------

    /** Green color as float array [R, G, B]. */
    private static final float[] GREEN_COLOR = {0.0f, 1.0f, 0.255f};

    // -----------------------------------------------------------------
    // Texture unit assignments
    // -----------------------------------------------------------------

    /** Camera texture unit index. */
    private static final int TEX_UNIT_CAMERA = 0;

    /** Segmentation mask texture unit index. */
    private static final int TEX_UNIT_MASK = 1;

    /** Label overlay texture unit index. */
    private static final int TEX_UNIT_LABEL = 2;

    /** Depth map texture unit index. */
    private static final int TEX_UNIT_DEPTH = 3;

    // -----------------------------------------------------------------
    // Fullscreen quad geometry
    // -----------------------------------------------------------------

    /**
     * Vertex data for the fullscreen quad: [x, y, u, v] interleaved.
     * Positions in NDC [-1, 1], tex coords in [0, 1].
     * The V coordinate is flipped to match GL convention.
     */
    private static final float[] FULLSCREEN_QUAD = {
            // position(x, y)  texCoord(u, v)
            -1f, -1f,          0f, 0f,    // bottom-left
             1f, -1f,          1f, 0f,    // bottom-right
            -1f,  1f,          0f, 1f,    // top-left
             1f,  1f,          1f, 1f,    // top-right
    };

    /** Draw order for the fullscreen quad as two triangles. */
    private static final short[] QUAD_INDICES = {0, 1, 2, 2, 1, 3};

    /** Floats per vertex: 2 position + 2 texcoord. */
    private static final int FLOATS_PER_VERTEX = 4;

    /** Bytes per float. */
    private static final int BYTES_PER_FLOAT = 4;

    /** Bytes per short. */
    private static final int BYTES_PER_SHORT = 2;

    // -----------------------------------------------------------------
    // GL handles
    // -----------------------------------------------------------------

    /** Camera texture handle (GL_TEXTURE_EXTERNAL_OES). */
    private int cameraTextureId = 0;

    /** Segmentation mask texture handle (GL_TEXTURE_2D, R8). */
    private int maskTextureId = 0;

    /** Depth map texture handle (GL_TEXTURE_2D, R16F). */
    private int depthTextureId = 0;

    /** Label overlay texture handle (GL_TEXTURE_2D, RGBA). */
    private int labelTextureId = 0;

    /** Vertex buffer object for the fullscreen quad. */
    private int quadVboId = 0;

    /** Index buffer object for the fullscreen quad. */
    private int quadIboId = 0;

    /** Vertex array object for the fullscreen quad. */
    private int quadVaoId = 0;

    // -----------------------------------------------------------------
    // SurfaceTexture for camera feed
    // -----------------------------------------------------------------

    /** The SurfaceTexture that receives camera preview frames. */
    private SurfaceTexture cameraSurfaceTexture;

    /** *** FIX #2: Camera transform matrix from SurfaceTexture. ***
     * Updated every frame from SurfaceTexture.getTransformMatrix().
     * This matrix handles rotation and mirroring of the camera feed.
     * Without applying it, the camera appears horizontal and/or mirrored. */
    private final float[] cameraTransformMatrix = new float[16];

    /** Callback to notify when the SurfaceTexture is ready. */
    @Nullable
    private OnSurfaceTextureReadyListener surfaceTextureListener;

    // -----------------------------------------------------------------
    // Shader programs
    // -----------------------------------------------------------------

    private CompositeShader compositeShader;
    private OutlineShader   outlineShader;
    private LabelShader     labelShader;

    // -----------------------------------------------------------------
    // Rendering state
    // -----------------------------------------------------------------

    /** Viewport width in pixels. */
    private int viewportWidth = 0;

    /** Viewport height in pixels. */
    private int viewportHeight = 0;

    /** Start time for animation effects. */
    private long startTimeMs = 0;

    /** Whether the mask texture needs updating. */
    private volatile boolean maskDirty = false;

    /** Whether the depth texture needs updating. */
    private volatile boolean depthDirty = false;

    /** Whether the label texture needs updating. */
    private volatile boolean labelDirty = false;

    /** Current mask data (int[height][width] of class IDs). */
    @Nullable
    private volatile int[][] pendingMaskData;

    /** Width of the pending mask. */
    private volatile int pendingMaskWidth;

    /** Height of the pending mask. */
    private volatile int pendingMaskHeight;

    /** Current depth data (float[height][width]). */
    @Nullable
    private volatile float[][] pendingDepthData;

    /** Width of the pending depth map. */
    private volatile int pendingDepthWidth;

    /** Height of the pending depth map. */
    private volatile int pendingDepthHeight;

    /** Current label bitmap pixel data (RGBA). */
    @Nullable
    private volatile ByteBuffer pendingLabelData;

    /** Width of the pending label texture. */
    private volatile int pendingLabelWidth;

    /** Height of the pending label texture. */
    private volatile int pendingLabelHeight;

    // -----------------------------------------------------------------
    // Effect parameters
    // -----------------------------------------------------------------

    /** Night mode enabled flag. */
    private volatile boolean nightMode = false;

    /** Scanline intensity [0.0, 1.0]. */
    private volatile float scanlineIntensity = 0.3f;

    /** Glow intensity for outline shader [0.0, 1.0]. */
    private volatile float glowIntensity = 0.5f;

    // -----------------------------------------------------------------
    // Path data (for path visualization)
    // -----------------------------------------------------------------

    /** Path vertex buffer for path line rendering. */
    @Nullable
    private FloatBuffer pathVertexBuffer;

    /** Number of path vertices to draw. */
    private volatile int pathVertexCount = 0;

    // -----------------------------------------------------------------
    // Context
    // -----------------------------------------------------------------

    private final Context context;

    // ================================================================
    // Listener interface
    // ================================================================

    /**
     * Callback interface for when the camera SurfaceTexture is created.
     * The camera subsystem uses this SurfaceTexture as the preview
     * surface provider.
     */
    public interface OnSurfaceTextureReadyListener {
        /**
         * Called when the SurfaceTexture is created and ready to receive
         * camera frames. The listener should set this SurfaceTexture as
         * the surface provider for the CameraX Preview use case.
         *
         * @param surfaceTexture The created SurfaceTexture.
         */
        void onSurfaceTextureReady(@NonNull SurfaceTexture surfaceTexture);
    }

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new GLRenderer.
     *
     * @param context Application context for loading shader assets.
     */
    public GLRenderer(@NonNull Context context) {
        this.context = context.getApplicationContext();
        this.startTimeMs = System.currentTimeMillis();
    }

    // ================================================================
    // SurfaceTexture Access
    // ================================================================

    /**
     * Sets a listener to be notified when the camera SurfaceTexture is ready.
     * The camera subsystem should use the provided SurfaceTexture as its
     * preview surface provider.
     *
     * @param listener The listener, or null to remove.
     */
    public void setOnSurfaceTextureReadyListener(@Nullable OnSurfaceTextureReadyListener listener) {
        this.surfaceTextureListener = listener;
    }

    /**
     * Returns the camera SurfaceTexture, or null if not yet created.
     * Prefer using the listener instead of polling this method.
     */
    @Nullable
    public SurfaceTexture getCameraSurfaceTexture() {
        return cameraSurfaceTexture;
    }

    // ================================================================
    // Data Update Methods (called from other threads)
    // ================================================================

    /**
     * Queues new segmentation mask data for upload to the GPU.
     * The data will be uploaded on the next onDrawFrame call.
     *
     * @param mask   2D int array of class IDs [height][width].
     * @param width  Mask width.
     * @param height Mask height.
     */
    public void updateMaskData(@NonNull int[][] mask, int width, int height) {
        this.pendingMaskData = mask;
        this.pendingMaskWidth = width;
        this.pendingMaskHeight = height;
        this.maskDirty = true;
    }

    /**
     * Queues new depth map data for upload to the GPU.
     *
     * @param depthMap 2D float array of depth values [height][width].
     * @param width    Depth map width.
     * @param height   Depth map height.
     */
    public void updateDepthData(@NonNull float[][] depthMap, int width, int height) {
        this.pendingDepthData = depthMap;
        this.pendingDepthWidth = width;
        this.pendingDepthHeight = height;
        this.depthDirty = true;
    }

    /**
     * Queues new label texture data for upload to the GPU.
     * The data should be in RGBA format, already pre-multiplied.
     *
     * @param pixelData ByteBuffer of RGBA pixel data.
     * @param width     Label texture width.
     * @param height    Label texture height.
     */
    public void updateLabelData(@NonNull ByteBuffer pixelData, int width, int height) {
        this.pendingLabelData = pixelData;
        this.pendingLabelWidth = width;
        this.pendingLabelHeight = height;
        this.labelDirty = true;
    }

    /**
     * Sets night mode on or off.
     *
     * @param enabled true to enable night mode.
     */
    public void setNightMode(boolean enabled) {
        this.nightMode = enabled;
    }

    /**
     * Sets the CRT scanline effect intensity.
     *
     * @param intensity Intensity [0.0 = off, 1.0 = maximum].
     */
    public void setScanlineIntensity(float intensity) {
        this.scanlineIntensity = Math.max(0f, Math.min(1f, intensity));
    }

    /**
     * Sets the glow intensity for contour outlines.
     *
     * @param intensity Glow strength [0.0, 1.0].
     */
    public void setGlowIntensity(float intensity) {
        this.glowIntensity = Math.max(0f, Math.min(1f, intensity));
    }

    /**
     * Sets the path vertex data for path visualization rendering.
     *
     * @param vertexBuffer FloatBuffer containing path vertex data.
     * @param vertexCount  Number of vertices in the buffer.
     */
    public void setPathData(@Nullable FloatBuffer vertexBuffer, int vertexCount) {
        this.pathVertexBuffer = vertexBuffer;
        this.pathVertexCount = vertexCount;
    }

    // ================================================================
    // GLSurfaceView.Renderer implementation
    // ================================================================

    @Override
    public void onSurfaceCreated(javax.microedition.khronos.opengles.GL10 gl,
                                  javax.microedition.khronos.egl.EGLConfig config) {
        Log.i(TAG, "onSurfaceCreated");

        // Reset start time for animations
        startTimeMs = System.currentTimeMillis();

        // Set clear color to black
        GLES30.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        // Create all shader programs
        compositeShader = new CompositeShader(context);
        if (!compositeShader.init()) {
            Log.e(TAG, "Failed to initialize composite shader");
        }

        outlineShader = new OutlineShader(context);
        if (!outlineShader.init()) {
            Log.e(TAG, "Failed to initialize outline shader");
        }

        labelShader = new LabelShader(context);
        if (!labelShader.init()) {
            Log.e(TAG, "Failed to initialize label shader");
        }

        // Create textures
        createCameraTexture();
        createMaskTexture();
        createDepthTexture();
        createLabelTexture();

        // Create fullscreen quad geometry
        createQuadBuffers();

        Log.i(TAG, "Surface created and all resources initialized");
    }

    @Override
    public void onSurfaceChanged(javax.microedition.khronos.opengles.GL10 gl,
                                  int width, int height) {
        Log.i(TAG, "onSurfaceChanged: " + width + "x" + height);
        viewportWidth = width;
        viewportHeight = height;
        GLES30.glViewport(0, 0, width, height);
    }

    @Override
    public void onDrawFrame(javax.microedition.khronos.opengles.GL10 gl) {
        // Clear the framebuffer
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);

        // 1. Update camera texture from SurfaceTexture
        if (cameraSurfaceTexture != null) {
            cameraSurfaceTexture.updateTexImage();
            // *** FIX #2: Get the transform matrix from SurfaceTexture every frame ***
            // This matrix handles camera rotation (portrait→landscape) and mirroring.
            // Without applying it, the camera feed appears horizontal and/or mirrored.
            cameraSurfaceTexture.getTransformMatrix(cameraTransformMatrix);
        }

        // 2. Upload dirty textures
        if (maskDirty) {
            uploadMaskTexture();
            maskDirty = false;
        }
        if (depthDirty) {
            uploadDepthTexture();
            depthDirty = false;
        }
        if (labelDirty) {
            uploadLabelTexture();
            labelDirty = false;
        }

        // 3. Calculate animation time
        float timeSeconds = (System.currentTimeMillis() - startTimeMs) / 1000f;

        // 4. Draw composite pass (camera + mask + labels + effects)
        drawCompositePass(timeSeconds);

        // 5. Draw path overlay (if path data exists)
        if (pathVertexCount > 0 && pathVertexBuffer != null) {
            // Path rendering would use a separate path shader
            // For now, we use the outline shader as a stand-in
            drawPathPass(timeSeconds);
        }
    }

    // ================================================================
    // Composite Pass — Main rendering
    // ================================================================

    /**
     * Draws the main composite pass: camera + green outlines + labels + effects.
     *
     * @param timeSeconds Current time for animation effects.
     */
    private void drawCompositePass(float timeSeconds) {
        if (!compositeShader.isInitialized()) {
            return;
        }

        // Activate the composite shader
        compositeShader.use();

        // Set texture uniforms — bind each texture to its unit
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + TEX_UNIT_CAMERA);
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId);
        compositeShader.setCameraTexture(TEX_UNIT_CAMERA);

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + TEX_UNIT_MASK);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, maskTextureId);
        compositeShader.setMaskTexture(TEX_UNIT_MASK);

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + TEX_UNIT_LABEL);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, labelTextureId);
        compositeShader.setLabelTexture(TEX_UNIT_LABEL);

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + TEX_UNIT_DEPTH);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTextureId);
        compositeShader.setDepthTexture(TEX_UNIT_DEPTH);

        // Reset active texture to default
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0);

        // Set parameter uniforms
        if (viewportWidth > 0 && viewportHeight > 0) {
            compositeShader.setTexelSize(
                    1.0f / viewportWidth, 1.0f / viewportHeight);
        }
        compositeShader.setGreenColor(GREEN_COLOR);
        compositeShader.setTime(timeSeconds);
        compositeShader.setScanlineIntensity(scanlineIntensity);
        compositeShader.setNightMode(nightMode ? 1.0f : 0.0f);

        // *** FIX #2: Pass the SurfaceTexture transform matrix to the shader ***
        // This handles camera rotation and mirroring so the preview displays correctly.
        compositeShader.setCameraTransform(cameraTransformMatrix);

        // Bind VAO and draw
        GLES30.glBindVertexArray(quadVaoId);
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, 6,
                GLES30.GL_UNSIGNED_SHORT, 0);
        GLES30.glBindVertexArray(0);
    }

    // ================================================================
    // Path Pass — Path visualization
    // ================================================================

    /**
     * Draws the path visualization overlay using the outline shader.
     *
     * @param timeSeconds Current time for animation effects.
     */
    private void drawPathPass(float timeSeconds) {
        if (!outlineShader.isInitialized() || pathVertexBuffer == null) {
            return;
        }

        // Enable blending for transparent path lines
        GLES30.glEnable(GLES30.GL_BLEND);
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA);

        outlineShader.use();
        outlineShader.setGreenColor(GREEN_COLOR);
        outlineShader.setGlowIntensity(glowIntensity);
        outlineShader.setTime(timeSeconds);
        outlineShader.setResolution(viewportWidth, viewportHeight);

        outlineShader.drawContours(pathVertexBuffer, pathVertexCount);

        GLES30.glDisable(GLES30.GL_BLEND);
    }

    // ================================================================
    // Texture Creation
    // ================================================================

    /**
     * Creates the camera texture as GL_TEXTURE_EXTERNAL_OES.
     * This texture receives camera frames via a SurfaceTexture.
     */
    private void createCameraTexture() {
        int[] textures = new int[1];
        GLES30.glGenTextures(1, textures, 0);
        cameraTextureId = textures[0];

        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId);
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);

        // Create the SurfaceTexture from the camera texture
        cameraSurfaceTexture = new SurfaceTexture(cameraTextureId);
        cameraSurfaceTexture.setDefaultBufferSize(
                com.syncvision.app.camera.CameraConfig.TARGET_WIDTH,
                com.syncvision.app.camera.CameraConfig.TARGET_HEIGHT);

        // Notify listener that the SurfaceTexture is ready
        if (surfaceTextureListener != null) {
            surfaceTextureListener.onSurfaceTextureReady(cameraSurfaceTexture);
        }

        Log.i(TAG, "Camera texture created (id=" + cameraTextureId + ")");
    }

    /**
     * Creates the segmentation mask texture as GL_TEXTURE_2D (R8 format).
     */
    private void createMaskTexture() {
        int[] textures = new int[1];
        GLES30.glGenTextures(1, textures, 0);
        maskTextureId = textures[0];

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, maskTextureId);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);

        // Allocate initial storage (will be resized on first upload)
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_R8,
                1, 1, 0, GLES30.GL_RED, GLES30.GL_UNSIGNED_BYTE, null);

        Log.i(TAG, "Mask texture created (id=" + maskTextureId + ")");
    }

    /**
     * Creates the depth map texture as GL_TEXTURE_2D (R16F format).
     */
    private void createDepthTexture() {
        int[] textures = new int[1];
        GLES30.glGenTextures(1, textures, 0);
        depthTextureId = textures[0];

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTextureId);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);

        // Allocate initial storage (R16F)
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_R16F,
                1, 1, 0, GLES30.GL_RED, GLES30.GL_HALF_FLOAT, null);

        Log.i(TAG, "Depth texture created (id=" + depthTextureId + ")");
    }

    /**
     * Creates the label overlay texture as GL_TEXTURE_2D (RGBA format).
     */
    private void createLabelTexture() {
        int[] textures = new int[1];
        GLES30.glGenTextures(1, textures, 0);
        labelTextureId = textures[0];

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, labelTextureId);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D,
                GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);

        // Allocate initial storage (1x1 transparent pixel)
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA,
                1, 1, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null);

        Log.i(TAG, "Label texture created (id=" + labelTextureId + ")");
    }

    // ================================================================
    // Quad Geometry
    // ================================================================

    /**
     * Creates the VBO, IBO, and VAO for the fullscreen quad.
     */
    private void createQuadBuffers() {
        // Create vertex buffer
        FloatBuffer vertexData = ByteBuffer.allocateDirect(
                FULLSCREEN_QUAD.length * BYTES_PER_FLOAT)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        vertexData.put(FULLSCREEN_QUAD).position(0);

        // Create index buffer
        ShortBuffer indexData = ByteBuffer.allocateDirect(
                QUAD_INDICES.length * BYTES_PER_SHORT)
                .order(ByteOrder.nativeOrder())
                .asShortBuffer();
        indexData.put(QUAD_INDICES).position(0);

        // Generate buffers
        int[] buffers = new int[2];
        GLES30.glGenBuffers(2, buffers, 0);
        quadVboId = buffers[0];
        quadIboId = buffers[1];

        // Upload vertex data
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, quadVboId);
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER,
                FULLSCREEN_QUAD.length * BYTES_PER_FLOAT,
                vertexData, GLES30.GL_STATIC_DRAW);

        // Upload index data
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, quadIboId);
        GLES30.glBufferData(GLES30.GL_ELEMENT_ARRAY_BUFFER,
                QUAD_INDICES.length * BYTES_PER_SHORT,
                indexData, GLES30.GL_STATIC_DRAW);

        // Create and configure VAO
        int[] vaos = new int[1];
        GLES30.glGenVertexArrays(1, vaos, 0);
        quadVaoId = vaos[0];

        GLES30.glBindVertexArray(quadVaoId);

        // Bind VBO and IBO
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, quadVboId);
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, quadIboId);

        // Setup attribute pointers for the composite shader layout
        // Stride: 4 floats * 4 bytes = 16 bytes
        final int stride = FLOATS_PER_VERTEX * BYTES_PER_FLOAT;

        // aPosition (location 0): 2 floats at offset 0
        GLES30.glVertexAttribPointer(
                CompositeShader.ATTR_POSITION, 2, GLES30.GL_FLOAT,
                false, stride, 0);
        GLES30.glEnableVertexAttribArray(CompositeShader.ATTR_POSITION);

        // aTexCoord (location 1): 2 floats at offset 8
        GLES30.glVertexAttribPointer(
                CompositeShader.ATTR_TEX_COORD, 2, GLES30.GL_FLOAT,
                false, stride, 2 * BYTES_PER_FLOAT);
        GLES30.glEnableVertexAttribArray(CompositeShader.ATTR_TEX_COORD);

        // Unbind VAO
        GLES30.glBindVertexArray(0);

        // Unbind buffers
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, 0);

        Log.i(TAG, "Fullscreen quad geometry created (VBO=" + quadVboId
                + ", IBO=" + quadIboId + ", VAO=" + quadVaoId + ")");
    }

    // ================================================================
    // Texture Upload Methods
    // ================================================================

    /**
     * Uploads the pending segmentation mask data to the mask texture.
     * Called from the GL thread during onDrawFrame.
     */
    private void uploadMaskTexture() {
        int[][] mask = pendingMaskData;
        if (mask == null) return;

        int w = pendingMaskWidth;
        int h = pendingMaskHeight;

        if (w <= 0 || h <= 0 || mask.length != h) {
            Log.w(TAG, "Invalid mask dimensions: " + w + "x" + h);
            return;
        }

        // Flatten the 2D mask into a single ByteBuffer (R8 format)
        // Each pixel is stored as a single byte representing the class ID
        ByteBuffer buffer = ByteBuffer.allocateDirect(w * h)
                .order(ByteOrder.nativeOrder());

        for (int y = 0; y < h; y++) {
            if (mask[y] != null && mask[y].length >= w) {
                for (int x = 0; x < w; x++) {
                    // Normalize class ID to [0, 1] range by dividing by max class (20)
                    // This way, different classes produce different brightnesses,
                    // and the Sobel edge detection picks up class boundaries
                    float normalized = mask[y][x] / 20.0f;
                    buffer.put((byte) (normalized * 255));
                }
            } else {
                // Fill with zeros for missing rows
                for (int x = 0; x < w; x++) {
                    buffer.put((byte) 0);
                }
            }
        }
        buffer.position(0);

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, maskTextureId);
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_R8,
                w, h, 0, GLES30.GL_RED, GLES30.GL_UNSIGNED_BYTE, buffer);
    }

    /**
     * Uploads the pending depth map data to the depth texture.
     * Called from the GL thread during onDrawFrame.
     */
    private void uploadDepthTexture() {
        float[][] depthMap = pendingDepthData;
        if (depthMap == null) return;

        int w = pendingDepthWidth;
        int h = pendingDepthHeight;

        if (w <= 0 || h <= 0 || depthMap.length != h) {
            Log.w(TAG, "Invalid depth dimensions: " + w + "x" + h);
            return;
        }

        // Flatten depth map into a ByteBuffer of half-floats (R16F)
        // OpenGL ES 3.0 supports GL_HALF_FLOAT
        ByteBuffer buffer = ByteBuffer.allocateDirect(w * h * 2)
                .order(ByteOrder.nativeOrder());

        for (int y = 0; y < h; y++) {
            if (depthMap[y] != null && depthMap[y].length >= w) {
                for (int x = 0; x < w; x++) {
                    // Convert float to half-float
                    float val = depthMap[y][x];
                    buffer.putShort(floatToHalfFloat(val));
                }
            } else {
                for (int x = 0; x < w; x++) {
                    buffer.putShort((short) 0);
                }
            }
        }
        buffer.position(0);

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, depthTextureId);
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_R16F,
                w, h, 0, GLES30.GL_RED, GLES30.GL_HALF_FLOAT, buffer);
    }

    /**
     * Uploads the pending label texture data.
     * Called from the GL thread during onDrawFrame.
     */
    private void uploadLabelTexture() {
        ByteBuffer data = pendingLabelData;
        if (data == null) return;

        int w = pendingLabelWidth;
        int h = pendingLabelHeight;

        if (w <= 0 || h <= 0) {
            Log.w(TAG, "Invalid label dimensions: " + w + "x" + h);
            return;
        }

        data.position(0);

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, labelTextureId);
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA,
                w, h, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, data);
    }

    // ================================================================
    // Cleanup
    // ================================================================

    /**
     * Releases all GL resources. Must be called on the GL thread.
     */
    public void cleanup() {
        Log.i(TAG, "Cleaning up GL resources");

        // Delete textures
        if (cameraTextureId != 0) {
            GLES30.glDeleteTextures(1, new int[]{cameraTextureId}, 0);
            cameraTextureId = 0;
        }
        if (maskTextureId != 0) {
            GLES30.glDeleteTextures(1, new int[]{maskTextureId}, 0);
            maskTextureId = 0;
        }
        if (depthTextureId != 0) {
            GLES30.glDeleteTextures(1, new int[]{depthTextureId}, 0);
            depthTextureId = 0;
        }
        if (labelTextureId != 0) {
            GLES30.glDeleteTextures(1, new int[]{labelTextureId}, 0);
            labelTextureId = 0;
        }

        // Delete buffers
        if (quadVboId != 0) {
            GLES30.glDeleteBuffers(1, new int[]{quadVboId}, 0);
            quadVboId = 0;
        }
        if (quadIboId != 0) {
            GLES30.glDeleteBuffers(1, new int[]{quadIboId}, 0);
            quadIboId = 0;
        }
        if (quadVaoId != 0) {
            GLES30.glDeleteVertexArrays(1, new int[]{quadVaoId}, 0);
            quadVaoId = 0;
        }

        // Release SurfaceTexture
        if (cameraSurfaceTexture != null) {
            cameraSurfaceTexture.release();
            cameraSurfaceTexture = null;
        }

        // Cleanup shader programs
        if (compositeShader != null) {
            compositeShader.cleanup();
        }
        if (outlineShader != null) {
            outlineShader.cleanup();
        }
        if (labelShader != null) {
            labelShader.cleanup();
        }
    }

    // ================================================================
    // Utility — Float to Half-Float Conversion
    // ================================================================

    /**
     * Converts a 32-bit float to a 16-bit half-float (IEEE 754 half-precision).
     * This is needed for uploading depth data to R16F textures on OpenGL ES 3.0.
     * <p>
     * Handles zero, denormals, infinity, and NaN.
     *
     * @param value The 32-bit float value.
     * @return The 16-bit half-float as a short.
     */
    private static short floatToHalfFloat(float value) {
        int bits = Float.floatToIntBits(value);
        int sign = (bits >>> 16) & 0x8000;     // Sign bit
        int exponent = ((bits >>> 23) & 0xFF);   // 8-bit exponent
        int mantissa = bits & 0x7FFFFF;           // 23-bit mantissa

        if (exponent == 0) {
            // Zero or denormal — convert to half zero
            return (short) sign;
        }

        if (exponent == 0xFF) {
            // Inf or NaN
            if (mantissa != 0) {
                return (short) (sign | 0x7FFF); // NaN
            }
            return (short) (sign | 0x7C00); // Inf
        }

        // Re-bias exponent: 127 → 15
        int newExp = exponent - 127 + 15;

        if (newExp >= 0x1F) {
            // Overflow — return Inf
            return (short) (sign | 0x7C00);
        }

        if (newExp <= 0) {
            // Underflow — return zero (simplified; could handle denormals)
            return (short) sign;
        }

        // Truncate mantissa from 23 bits to 10 bits
        int newMantissa = mantissa >> 13;

        return (short) (sign | (newExp << 10) | newMantissa);
    }
}
