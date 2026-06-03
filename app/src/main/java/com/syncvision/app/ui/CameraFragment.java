/**
 * CameraFragment.java
 *
 * Fragment containing the GLSurfaceView for camera rendering.
 * Manages the GLSurfaceView lifecycle, connects it to the
 * CameraManager for preview surface, and handles surface
 * creation/change/destruction callbacks.
 *
 * This fragment serves as the bottom layer of the camera view
 * stack, providing the OpenGL rendering surface that composites
 * the camera feed with ML overlay effects.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.ui
 * Target SDK: 29+
 */

package com.syncvision.app.ui;

import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.SurfaceHolder;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

/**
 * Fragment that hosts the GLSurfaceView for camera rendering.
 * <p>
 * The GLSurfaceView is created externally by GLSurfaceManager and
 * injected into this fragment via {@link #setGLSurfaceView(GLSurfaceView)}.
 * The fragment manages the view lifecycle (add/remove from hierarchy)
 * and forwards surface callbacks to the rendering system.
 * <p>
 * Usage:
 * <pre>
 *   CameraFragment fragment = new CameraFragment();
 *   fragment.setGLSurfaceView(glSurfaceManager.getSurfaceView());
 *   getSupportFragmentManager().beginTransaction()
 *       .add(android.R.id.content, fragment)
 *       .commit();
 * </pre>
 */
public class CameraFragment extends Fragment {

    private static final String TAG = "SV-CameraFragment";

    /** The GLSurfaceView for OpenGL camera rendering. */
    @Nullable
    private GLSurfaceView glSurfaceView;

    /** Whether the surface has been created. */
    private boolean surfaceCreated = false;

    /** Current surface width in pixels. */
    private int surfaceWidth = 0;

    /** Current surface height in pixels. */
    private int surfaceHeight = 0;

    // ================================================================
    // GLSurfaceView Injection
    // ================================================================

    /**
     * Sets the GLSurfaceView to be hosted by this fragment.
     * Must be called before the fragment is added to the activity.
     *
     * @param surfaceView The GLSurfaceView from GLSurfaceManager.
     */
    public void setGLSurfaceView(@NonNull GLSurfaceView surfaceView) {
        this.glSurfaceView = surfaceView;

        // Attach surface holder callback for lifecycle tracking
        surfaceView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(@NonNull SurfaceHolder holder) {
                surfaceCreated = true;
                Log.i(TAG, "Camera surface created");
            }

            @Override
            public void surfaceChanged(@NonNull SurfaceHolder holder,
                                       int format, int width, int height) {
                surfaceWidth = width;
                surfaceHeight = height;
                Log.d(TAG, "Camera surface changed: " + width + "x" + height);
            }

            @Override
            public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
                surfaceCreated = false;
                Log.i(TAG, "Camera surface destroyed");
            }
        });
    }

    // ================================================================
    // Fragment Lifecycle
    // ================================================================

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                             @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        if (glSurfaceView != null) {
            // If the GLSurfaceView already has a parent, remove it first
            if (glSurfaceView.getParent() != null) {
                ((ViewGroup) glSurfaceView.getParent()).removeView(glSurfaceView);
            }
            return glSurfaceView;
        }

        // Fallback: create a simple container if no GLSurfaceView was set
        Log.w(TAG, "No GLSurfaceView set — creating empty container");
        return new View(requireContext());
    }

    @Override
    public void onResume() {
        super.onResume();
        if (glSurfaceView != null) {
            glSurfaceView.onResume();
            Log.d(TAG, "GLSurfaceView resumed");
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        if (glSurfaceView != null) {
            glSurfaceView.onPause();
            Log.d(TAG, "GLSurfaceView paused");
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // Note: We do NOT destroy the GLSurfaceView here because
        // it is owned by GLSurfaceManager, which handles cleanup.
        // Only detach it from the fragment's view hierarchy.
        if (glSurfaceView != null && glSurfaceView.getParent() != null) {
            ((ViewGroup) glSurfaceView.getParent()).removeView(glSurfaceView);
        }
        Log.d(TAG, "CameraFragment view destroyed");
    }

    // ================================================================
    // Public Accessors
    // ================================================================

    /**
     * Returns the GLSurfaceView hosted by this fragment.
     *
     * @return The GLSurfaceView, or null if not set.
     */
    @Nullable
    public GLSurfaceView getGLSurfaceView() {
        return glSurfaceView;
    }

    /**
     * Returns whether the GL surface has been created and is available.
     *
     * @return true if the surface is ready for rendering.
     */
    public boolean isSurfaceCreated() {
        return surfaceCreated;
    }

    /**
     * Returns the current surface width in pixels.
     */
    public int getSurfaceWidth() {
        return surfaceWidth;
    }

    /**
     * Returns the current surface height in pixels.
     */
    public int getSurfaceHeight() {
        return surfaceHeight;
    }
}
