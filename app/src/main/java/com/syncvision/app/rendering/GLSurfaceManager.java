/**
 * GLSurfaceManager.java
 *
 * Manages the GLSurfaceView lifecycle for the Sync Vision app.
 * Creates and configures the GLSurfaceView with an OpenGL ES 3.0 context,
 * connects it to the GLRenderer, and handles surface creation, change,
 * and destruction events.
 *
 * Provides the SurfaceTexture from the GLRenderer for use by the
 * CameraX Preview use case, bridging the camera subsystem with the
 * rendering subsystem.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.rendering
 * Target SDK: 29+
 */

package com.syncvision.app.rendering;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.opengl.GLSurfaceView;
import android.util.Log;
import android.view.SurfaceHolder;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.syncvision.app.camera.CameraConfig;

/**
 * Manages the GLSurfaceView lifecycle and configuration for the Sync
 * Vision AR overlay rendering pipeline.
 * <p>
 * Responsibilities:
 *   - Creates and configures the GLSurfaceView with OpenGL ES 3.0
 *   - Connects the GLSurfaceView to the GLRenderer
 *   - Provides the camera SurfaceTexture for CameraX Preview
 *   - Handles surface lifecycle events (create, change, destroy)
 *   - Manages render mode (continuous vs on-demand)
 * <p>
 * Usage:
 * <pre>
 *   GLSurfaceManager surfaceManager = new GLSurfaceManager(context);
 *   GLSurfaceView surfaceView = surfaceManager.getSurfaceView();
 *   // Add surfaceView to your layout
 *   surfaceManager.setCameraPreviewProvider(provider -> {
 *       // Set this provider as CameraX Preview.SurfaceProvider
 *   });
 * </pre>
 */
public class GLSurfaceManager {

    private static final String TAG = "SV-GLSurfaceManager";

    // -----------------------------------------------------------------
    // Core components
    // -----------------------------------------------------------------

    /** The GLSurfaceView that hosts the OpenGL rendering surface. */
    private final GLSurfaceView glSurfaceView;

    /** The GLRenderer that performs the actual OpenGL drawing. */
    private final GLRenderer glRenderer;

    // -----------------------------------------------------------------
    // State
    // -----------------------------------------------------------------

    /** Whether the surface has been created and is ready for rendering. */
    private boolean surfaceAvailable = false;

    /** Whether the renderer has been properly initialized. */
    private boolean rendererInitialized = false;

    // -----------------------------------------------------------------
    // Listener
    // -----------------------------------------------------------------

    /**
     * Callback interface for camera preview SurfaceTexture availability.
     * The camera subsystem implements this to receive the SurfaceTexture
     * and set it as the CameraX Preview surface provider.
     */
    public interface OnCameraPreviewProviderListener {
        /**
         * Called when the SurfaceTexture for camera preview is ready.
         * The listener should use this SurfaceTexture to configure
         * the CameraX Preview use case.
         *
         * @param surfaceTexture The camera preview SurfaceTexture.
         */
        void onCameraPreviewProviderReady(@NonNull SurfaceTexture surfaceTexture);
    }

    @Nullable
    private OnCameraPreviewProviderListener cameraPreviewListener;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new GLSurfaceManager, initializing the GLSurfaceView
     * and GLRenderer with an OpenGL ES 3.0 context.
     *
     * @param context Application or activity context.
     */
    public GLSurfaceManager(@NonNull Context context) {
        // Create the GLSurfaceView
        glSurfaceView = new GLSurfaceView(context);

        // Create the renderer
        glRenderer = new GLRenderer(context);

        // Configure OpenGL ES 3.0 context
        glSurfaceView.setEGLContextClientVersion(3);

        // Configure EGL config: 8-bit RGBA, 16-bit depth, no stencil
        glSurfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0);

        // Set the renderer
        glSurfaceView.setRenderer(glRenderer);

        // Set render mode to continuous for camera preview (30 FPS)
        glSurfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);

        // Listen for the SurfaceTexture from the renderer
        glRenderer.setOnSurfaceTextureReadyListener(this::onSurfaceTextureReady);

        // Add a SurfaceHolder callback for lifecycle tracking
        glSurfaceView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(@NonNull SurfaceHolder holder) {
                surfaceAvailable = true;
                rendererInitialized = true;
                Log.i(TAG, "Surface created");
            }

            @Override
            public void surfaceChanged(@NonNull SurfaceHolder holder,
                                       int format, int width, int height) {
                Log.i(TAG, "Surface changed: " + width + "x" + height
                        + " format=" + format);
            }

            @Override
            public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
                surfaceAvailable = false;
                Log.i(TAG, "Surface destroyed");
            }
        });

        Log.i(TAG, "GLSurfaceManager initialized with OpenGL ES 3.0");
    }

    // ================================================================
    // Public API
    // ================================================================

    /**
     * Returns the GLSurfaceView for adding to a layout.
     * This view handles the OpenGL rendering surface.
     *
     * @return The configured GLSurfaceView.
     */
    @NonNull
    public GLSurfaceView getSurfaceView() {
        return glSurfaceView;
    }

    /**
     * Returns the GLRenderer instance for direct interaction
     * (e.g., setting rendering parameters).
     *
     * @return The GLRenderer.
     */
    @NonNull
    public GLRenderer getRenderer() {
        return glRenderer;
    }

    /**
     * Sets a listener to be notified when the camera preview SurfaceTexture
     * is available. The listener should use the provided SurfaceTexture
     * to configure the CameraX Preview use case.
     *
     * @param listener The listener, or null to remove.
     */
    public void setOnCameraPreviewProviderListener(
            @Nullable OnCameraPreviewProviderListener listener) {
        this.cameraPreviewListener = listener;
    }

    /**
     * Returns whether the GL surface is available for rendering.
     */
    public boolean isSurfaceAvailable() {
        return surfaceAvailable;
    }

    /**
     * Pauses rendering. Call this in Activity.onPause().
     * This stops the rendering thread from updating the surface.
     */
    public void onPause() {
        if (glSurfaceView != null) {
            glSurfaceView.onPause();
            Log.i(TAG, "GLSurfaceView paused");
        }
    }

    /**
     * Resumes rendering. Call this in Activity.onResume().
     * This restarts the rendering thread.
     */
    public void onResume() {
        if (glSurfaceView != null) {
            glSurfaceView.onResume();
            Log.i(TAG, "GLSurfaceView resumed");
        }
    }

    /**
     * Queues a task to be executed on the GL rendering thread.
     * Use this for any GL operations that must happen on the GL thread
     * (e.g., texture updates that are not already handled by the renderer).
     *
     * @param runnable The task to run on the GL thread.
     */
    public void queueEvent(@NonNull Runnable runnable) {
        if (glSurfaceView != null) {
            glSurfaceView.queueEvent(runnable);
        }
    }

    /**
     * Requests a single render frame. Useful when render mode is set
     * to RENDERMODE_WHEN_DIRTY and you need to trigger a redraw.
     */
    public void requestRender() {
        if (glSurfaceView != null) {
            glSurfaceView.requestRender();
        }
    }

    /**
     * Sets the render mode.
     *
     * @param mode One of GLSurfaceView.RENDERMODE_CONTINUOUSLY
     *             or GLSurfaceView.RENDERMODE_WHEN_DIRTY.
     */
    public void setRenderMode(int mode) {
        if (glSurfaceView != null) {
            glSurfaceView.setRenderMode(mode);
        }
    }

    /**
     * Releases all resources associated with the surface manager.
     * Call this in Activity.onDestroy(). Must be called after
     * the surface is destroyed (pause should be called first).
     */
    public void destroy() {
        // Queue cleanup on the GL thread
        queueEvent(() -> {
            if (glRenderer != null) {
                glRenderer.cleanup();
            }
        });
        Log.i(TAG, "GLSurfaceManager destroyed");
    }

    // ================================================================
    // Camera Preview SurfaceTexture
    // ================================================================

    /**
     * Returns the SurfaceTexture for camera preview, if available.
     * The camera subsystem uses this as the preview surface.
     *
     * @return The camera SurfaceTexture, or null if not yet created.
     */
    @Nullable
    public SurfaceTexture getCameraSurfaceTexture() {
        return glRenderer != null ? glRenderer.getCameraSurfaceTexture() : null;
    }

    /**
     * Creates a Preview.SurfaceProvider that can be passed directly
     * to the CameraX Preview use case. This provides the camera
     * subsystem with the surface it needs to render preview frames.
     *
     * @return A SurfaceProvider for CameraX Preview, or null if
     *         the SurfaceTexture is not yet available.
     */
    @Nullable
    public androidx.camera.core.Preview.SurfaceProvider getCameraSurfaceProvider() {
        SurfaceTexture st = getCameraSurfaceTexture();
        if (st == null) {
            Log.w(TAG, "SurfaceTexture not yet available for camera preview");
            return null;
        }

        // Create a SurfaceProvider that feeds camera frames to our SurfaceTexture
        return request -> {
            SurfaceTexture surfaceTexture = getCameraSurfaceTexture();
            if (surfaceTexture != null) {
                android.view.Surface surface = new android.view.Surface(surfaceTexture);
                request.provideSurface(surface, Runnable::run, () -> {
                    // Surface is no longer needed
                    if (!surface.isReleased()) {
                        surface.release();
                    }
                });
            }
        };
    }

    // ================================================================
    // Internal
    // ================================================================

    /**
     * Called when the GLRenderer creates the camera SurfaceTexture.
     * Notifies the camera preview listener.
     */
    private void onSurfaceTextureReady(@NonNull SurfaceTexture surfaceTexture) {
        Log.i(TAG, "Camera SurfaceTexture ready");
        if (cameraPreviewListener != null) {
            cameraPreviewListener.onCameraPreviewProviderReady(surfaceTexture);
        }
    }
}
