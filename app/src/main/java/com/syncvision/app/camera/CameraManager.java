/**
 * CameraManager.java
 *
 * Manages the CameraX lifecycle for the Sync Vision app.
 * Binds Preview and ImageAnalysis use cases, handles camera
 * permissions, camera switching (front/back), and flash toggle.
 * Delivers ImageProxy frames to the FrameDispatcher for ML processing.
 *
 * Sync Vision â€” Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.camera
 * Target SDK: 29+
 */

package com.syncvision.app.camera;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.util.Log;
import android.view.Surface;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Manages the CameraX camera lifecycle, use-case binding, and frame delivery.
 * <p>
 * Usage:
 * <pre>
 *   CameraManager mgr = new CameraManager(context, lifecycleOwner);
 *   mgr.setFrameDispatcher(dispatcher);
 *   if (mgr.hasPermission()) {
 *       mgr.startCamera(surfaceProvider);
 *   } else {
 *       mgr.requestPermission(activity);
 *   }
 * </pre>
 */
public class CameraManager {

    private static final String TAG = "SV-CameraManager";

    /** Permission request code for camera access. */
    public static final int CAMERA_PERMISSION_REQUEST_CODE = 1001;

    // -----------------------------------------------------------------
    // Dependencies
    // -----------------------------------------------------------------

    private final Context context;
    private final LifecycleOwner lifecycleOwner;

    /** Receives frames from ImageAnalysis for ML processing. */
    @Nullable
    private FrameDispatcher frameDispatcher;

    // -----------------------------------------------------------------
    // CameraX components
    // -----------------------------------------------------------------

    @Nullable
    private ProcessCameraProvider cameraProvider;

    @Nullable
    private Preview preview;

    @Nullable
    private ImageAnalysis imageAnalysis;

    @Nullable
    private Camera camera;

    /** Background executor for ImageAnalysis callbacks. */
    private final ExecutorService analysisExecutor;

    // -----------------------------------------------------------------
    // State
    // -----------------------------------------------------------------

    /** Current camera lens facing direction. */
    private int currentLensFacing = CameraSelector.LENS_FACING_BACK;

    /** Whether the flash is currently enabled. */
    private boolean flashEnabled = false;

    /** Whether the camera is currently bound and running. */
    private boolean isRunning = false;

    // -----------------------------------------------------------------
    // Listener
    // -----------------------------------------------------------------

    /**
     * Callback interface for camera lifecycle events.
     */
    public interface OnCameraEventListener {
        /** Called when the camera has started successfully. */
        void onCameraStarted();

        /** Called when a camera error occurs. */
        void onCameraError(String message);

        /** Called when the camera is switched (front â†” back). */
        void onCameraSwitched(boolean isFrontCamera);

        /** Called when flash state changes. */
        void onFlashToggled(boolean isEnabled);
    }

    @Nullable
    private OnCameraEventListener eventListener;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new CameraManager.
     *
     * @param context       Application or activity context.
     * @param lifecycleOwner Lifecycle owner (typically the Activity).
     */
    public CameraManager(@NonNull Context context, @NonNull LifecycleOwner lifecycleOwner) {
        this.context = context.getApplicationContext();
        this.lifecycleOwner = lifecycleOwner;
        this.analysisExecutor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "SV-CameraAnalysis");
            t.setPriority(Thread.NORM_PRIORITY + 1);
            return t;
        });
    }

    // ================================================================
    // Public API
    // ================================================================

    /**
     * Sets the FrameDispatcher to receive camera frames.
     *
     * @param dispatcher The frame dispatcher instance, or null to disconnect.
     */
    public void setFrameDispatcher(@Nullable FrameDispatcher dispatcher) {
        this.frameDispatcher = dispatcher;
    }

    /**
     * Sets an event listener for camera lifecycle events.
     *
     * @param listener The listener, or null to remove.
     */
    public void setEventListener(@Nullable OnCameraEventListener listener) {
        this.eventListener = listener;
    }

    /**
     * Checks if the camera permission has been granted.
     *
     * @return true if permission is granted.
     */
    public boolean hasPermission() {
        return ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * Requests camera permission from the user.
     * Must be called from an Activity.
     *
     * @param activity The requesting Activity.
     */
    public void requestPermission(@NonNull androidx.activity.ComponentActivity activity) {
        ActivityCompat.requestPermissions(
                activity,
                new String[]{Manifest.permission.CAMERA},
                CAMERA_PERMISSION_REQUEST_CODE
        );
    }

    /**
     * Handles the permission request result.
     *
     * @param requestCode  The request code passed to requestPermission.
     * @param grantResults The grant results.
     * @return true if camera permission was granted.
     */
    public boolean onRequestPermissionsResult(int requestCode, @NonNull int[] grantResults) {
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.i(TAG, "Camera permission granted");
                return true;
            } else {
                Log.w(TAG, "Camera permission denied");
                if (eventListener != null) {
                    eventListener.onCameraError("Camera permission denied");
                }
            }
        }
        return false;
    }

    /**
     * Starts the camera with the given surface provider for preview rendering.
     * Must be called on the main thread.
     *
     * @param surfaceProvider The surface provider for the Preview use case
     *                        (typically from a PreviewView or GLSurfaceView).
     */
    public void startCamera(@NonNull Preview.SurfaceProvider surfaceProvider) {
        if (!hasPermission()) {
            Log.e(TAG, "Cannot start camera: permission not granted");
            if (eventListener != null) {
                eventListener.onCameraError("Camera permission not granted");
            }
            return;
        }

        ListenableFuture<ProcessCameraProvider> providerFuture =
                ProcessCameraProvider.getInstance(context);

        providerFuture.addListener(() -> {
            try {
                cameraProvider = providerFuture.get();
                bindUseCases(surfaceProvider);
                isRunning = true;
                Log.i(TAG, "Camera started successfully");
                if (eventListener != null) {
                    eventListener.onCameraStarted();
                }
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Failed to start camera", e);
                if (eventListener != null) {
                    eventListener.onCameraError("Failed to start camera: " + e.getMessage());
                }
            }
        }, ContextCompat.getMainExecutor(context));
    }

    /**
     * Stops the camera and releases resources.
     */
    public void stopCamera() {
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
            isRunning = false;
            Log.i(TAG, "Camera stopped");
        }
    }

    /**
     * Switches between front and back cameras.
     * Rebinds the use cases with the new camera selector.
     *
     * @param surfaceProvider The current surface provider for re-binding.
     */
    public void switchCamera(@NonNull Preview.SurfaceProvider surfaceProvider) {
        if (cameraProvider == null) {
            Log.w(TAG, "Cannot switch camera: provider not initialized");
            return;
        }

        currentLensFacing = (currentLensFacing == CameraSelector.LENS_FACING_BACK)
                ? CameraSelector.LENS_FACING_FRONT
                : CameraSelector.LENS_FACING_BACK;

        // Flash is not available on front camera
        if (currentLensFacing == CameraSelector.LENS_FACING_FRONT) {
            flashEnabled = false;
        }

        bindUseCases(surfaceProvider);

        boolean isFront = (currentLensFacing == CameraSelector.LENS_FACING_FRONT);
        Log.i(TAG, "Switched to " + (isFront ? "front" : "back") + " camera");
        if (eventListener != null) {
            eventListener.onCameraSwitched(isFront);
        }
    }

    /**
     * Toggles the flash (torch) on/off.
     * Only works on the back camera.
     *
     * @return The new flash state (true = on).
     */
    public boolean toggleFlash() {
        if (camera == null) {
            Log.w(TAG, "Cannot toggle flash: camera not bound");
            return false;
        }

        // Front camera does not support flash
        if (currentLensFacing == CameraSelector.LENS_FACING_FRONT) {
            Log.w(TAG, "Flash not available on front camera");
            return false;
        }

        flashEnabled = !flashEnabled;
        camera.getCameraControl().enableTorch(flashEnabled);

        Log.i(TAG, "Flash " + (flashEnabled ? "ON" : "OFF"));
        if (eventListener != null) {
            eventListener.onFlashToggled(flashEnabled);
        }
        return flashEnabled;
    }

    /**
     * Returns whether the flash is currently enabled.
     */
    public boolean isFlashEnabled() {
        return flashEnabled;
    }

    /**
     * Returns whether the camera is currently running.
     */
    public boolean isRunning() {
        return isRunning;
    }

    /**
     * Returns the current lens facing direction.
     *
     * @return CameraSelector.LENS_FACING_BACK or LENS_FACING_FRONT.
     */
    public int getCurrentLensFacing() {
        return currentLensFacing;
    }

    /**
     * Returns the camera preview resolution width.
     * Returns 0 if the preview is not yet bound.
     */
    public int getPreviewWidth() {
        if (preview != null && preview.getResolutionInfo() != null) {
            return preview.getResolutionInfo().getResolution().getWidth();
        }
        return CameraConfig.TARGET_WIDTH;
    }

    /**
     * Returns the camera preview resolution height.
     * Returns 0 if the preview is not yet bound.
     */
    public int getPreviewHeight() {
        if (preview != null && preview.getResolutionInfo() != null) {
            return preview.getResolutionInfo().getResolution().getHeight();
        }
        return CameraConfig.TARGET_HEIGHT;
    }

    /**
     * Returns the camera sensor rotation (degrees).
     * Useful for correctly orienting ML input and overlay rendering.
     */
    public int getSensorRotation() {
        if (imageAnalysis != null) {
            int rotation = imageAnalysis.getTargetRotation();
            // Validate that the rotation is a legitimate Surface constant
            if (rotation == Surface.ROTATION_0 || rotation == Surface.ROTATION_90
                    || rotation == Surface.ROTATION_180 || rotation == Surface.ROTATION_270) {
                return rotation;
            }
        }
        return Surface.ROTATION_0;
    }

    /**
     * Shuts down the analysis executor. Call this in Activity.onDestroy().
     */
    public void shutdown() {
        stopCamera();
        analysisExecutor.shutdownNow();
        Log.i(TAG, "CameraManager shut down");
    }

    // ================================================================
    // Internal â€” Use Case Binding
    // ================================================================

    /**
     * Binds (or re-binds) Preview and ImageAnalysis use cases to the camera.
     * Must be called on the main thread.
     *
     * @param surfaceProvider The surface provider for preview rendering.
     */
    private void bindUseCases(@NonNull Preview.SurfaceProvider surfaceProvider) {
        if (cameraProvider == null) {
            Log.e(TAG, "Cannot bind use cases: cameraProvider is null");
            return;
        }

        // Unbind any existing use cases before re-binding
        cameraProvider.unbindAll();

        // ----- Camera Selector -----
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(currentLensFacing)
                .build();

        // ----- Preview Use Case -----
        preview = new Preview.Builder()
                .setTargetResolution(
                        new android.util.Size(CameraConfig.TARGET_WIDTH, CameraConfig.TARGET_HEIGHT))
                .build();
        preview.setSurfaceProvider(surfaceProvider);

        // ----- ImageAnalysis Use Case -----
        imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(
                        new android.util.Size(CameraConfig.TARGET_WIDTH, CameraConfig.TARGET_HEIGHT))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build();

        // Set the analyzer that delivers frames to FrameDispatcher
        imageAnalysis.setAnalyzer(analysisExecutor, this::onImageAvailable);

        // ----- Bind to Lifecycle -----
        try {
            camera = cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalysis
            );

            // Apply initial flash state
            if (flashEnabled && currentLensFacing == CameraSelector.LENS_FACING_BACK) {
                camera.getCameraControl().enableTorch(true);
            }
        } catch (IllegalArgumentException e) {
            Log.e(TAG, "Failed to bind use cases. Is the camera selector valid?", e);
            if (eventListener != null) {
                eventListener.onCameraError("Camera binding failed: " + e.getMessage());
            }
        }
    }

    // ================================================================
    // Internal â€” Frame Delivery
    // ================================================================

    /**
     * ImageAnalysis.Analyzer callback. Called on the analysis executor thread
     * for each new camera frame.
     *
     * @param imageProxy The camera frame with image data.
     */
    private void onImageAvailable(@NonNull ImageProxy imageProxy) {
        try {
            if (frameDispatcher != null) {
                frameDispatcher.dispatchFrame(imageProxy);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error dispatching frame", e);
        } finally {
            // IMPORTANT: Always close the ImageProxy to free the buffer
            // and allow the next frame to be delivered.
            imageProxy.close();
        }
    }
}
