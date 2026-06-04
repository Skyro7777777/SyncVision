/**
 * MainActivity.java
 *
 * Main activity for the Sync Vision camera app. Provides a fullscreen,
 * immersive camera view with real-time ML overlay rendering.
 *
 * Sets up and connects:
 *   - CameraManager (camera feed via CameraX)
 *   - GLSurfaceManager (OpenGL ES 3.0 rendering)
 *   - FrameDispatcher (ML pipeline dispatch hub)
 *   - OverlayRenderer (high-level rendering API)
 *   - OverlayView (Canvas-based HUD elements)
 *   - SyncDiagramView (relationship graph)
 *   - InfoPanelView (object detail panel)
 *   - SceneUnderstanding (scene analysis engine)
 *
 * Touch interactions:
 *   - Single tap:  trigger on-demand analysis (OCR, plant ID, landmark)
 *   - Long press:  toggle settings panel
 *   - Swipe L/R:   switch camera (front/back)
 *   - Pinch:       zoom (future)
 *
 * UI overlay elements:
 *   - Top-left:    Weather status ("WEATHER: PARTLY CLOUDY 72%")
 *   - Top-right:   Face count ("3 FACES DETECTED")
 *   - Bottom-left: FPS counter + threat indicator
 *   - Bottom-right: Sync diagram (small floating view)
 *   - Center-bottom: Mode toggle buttons (SCAN / PATH / IDENTIFY)
 *   - Status bar:  Green status indicators
 *
 * Sync Vision â€” Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.ui
 * Target SDK: 29+
 */

package com.syncvision.app.ui;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.PointF;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.syncvision.app.R;
import com.syncvision.app.camera.CameraManager;
import com.syncvision.app.camera.FrameDispatcher;
import com.syncvision.app.ml.DetectionPipeline;
import com.syncvision.app.ml.DepthPipeline;
import com.syncvision.app.ml.FacePipeline;
import com.syncvision.app.ml.InferenceResult;
import com.syncvision.app.ml.ModelManager;
import com.syncvision.app.ml.SegmentationPipeline;
import com.syncvision.app.ml.WeatherPipeline;
import com.syncvision.app.ml.OcrPipeline;
import com.syncvision.app.ml.PlantPipeline;
import com.syncvision.app.ml.BarcodePipeline;
import com.syncvision.app.ml.LandmarkPipeline;
import com.syncvision.app.nativelib.NativeProcessor;
import com.syncvision.app.rendering.GLSurfaceManager;
import com.syncvision.app.rendering.OverlayRenderer;
import com.syncvision.app.scene.SceneUnderstanding;

import java.util.Locale;

/**
 * Main activity â€” fullscreen camera view with ML-powered overlay.
 * <p>
 * This is the primary (and only visible) activity in the Sync Vision app.
 * It orchestrates the camera, ML pipelines, OpenGL rendering, and HUD
 * overlay into a unified E.D.I.T.H-style augmented view.
 * <p>
 * The activity maintains an immersive, fullscreen experience with
 * transparent system bars and keeps the screen on during use.
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "SV-MainActivity";

    /** Permission request code for camera access. */
    private static final int CAMERA_PERMISSION_REQUEST = 1001;

    /** Minimum swipe distance in pixels for camera switch gesture. */
    private static final float SWIPE_THRESHOLD = 120f;

    /** Long press duration in milliseconds to open settings. */
    private static final long LONG_PRESS_TIMEOUT_MS = 800L;

    /** Vibration duration for hazard alerts in milliseconds. */
    private static final long HAZARD_VIBRATION_MS = 200L;

    // ================================================================
    // Core Components
    // ================================================================

    /** Camera subsystem manager. */
    private CameraManager cameraManager;

    /** OpenGL surface and renderer manager. */
    private GLSurfaceManager glSurfaceManager;

    /** ML pipeline dispatch hub. */
    private FrameDispatcher frameDispatcher;

    /** High-level rendering API. */
    private OverlayRenderer overlayRenderer;

    /** Native processor for C++ operations. */
    private NativeProcessor nativeProcessor;

    /** Scene understanding engine (fuses all ML results). */
    private SceneUnderstanding sceneUnderstanding;

    // ================================================================
    // ML Pipelines
    // ================================================================

    private SegmentationPipeline segmentationPipeline;
    private DetectionPipeline detectionPipeline;
    private DepthPipeline depthPipeline;
    private FacePipeline facePipeline;
    private WeatherPipeline weatherPipeline;
    private OcrPipeline ocrPipeline;
    private PlantPipeline plantPipeline;
    private BarcodePipeline barcodePipeline;
    private LandmarkPipeline landmarkPipeline;

    // ================================================================
    // UI Views
    // ================================================================

    /** Root layout for the activity. */
    private FrameLayout rootLayout;

    /** Canvas-based HUD overlay view. */
    private OverlayView overlayView;

    /** Sync diagram floating view. */
    private SyncDiagramView syncDiagramView;

    /** Object info panel (slides in from right). */
    private InfoPanelView infoPanelView;

    /** Camera fragment hosting the GLSurfaceView. */
    private CameraFragment cameraFragment;

    // ================================================================
    // State
    // ================================================================

    /** Current operating mode. */
    private OperatingMode currentMode = OperatingMode.SCAN;

    /** Whether the camera has been started. */
    private boolean cameraStarted = false;

    /** Last scene result from the frame dispatcher. */
    @Nullable
    private volatile InferenceResult.SceneResult lastSceneResult;

    /** Handler for posting UI updates from background threads. */
    private final Handler uiHandler = new Handler(Looper.getMainLooper());

    /** Vibrator service for haptic feedback on hazard alerts. */
    @Nullable
    private Vibrator vibrator;

    /** Previous threat level for change detection. */
    private int previousThreatLevel = 0;

    // ================================================================
    // Gesture Detection
    // ================================================================

    private GestureDetector gestureDetector;

    // ================================================================
    // Operating Mode Enum
    // ================================================================

    /**
     * Operating modes for the camera overlay.
     * Each mode changes which ML pipelines are prioritized and
     * what information is displayed on the HUD.
     */
    public enum OperatingMode {
        /** Default mode: object detection, segmentation, labels. */
        SCAN,
        /** Path/wayfinding mode: depth analysis, obstacle detection. */
        PATH,
        /** Identification mode: OCR, plant ID, landmark recognition. */
        IDENTIFY
    }

    // ================================================================
    // Activity Lifecycle
    // ================================================================

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Enable fullscreen immersive mode
        setupImmersiveMode();

        // Keep screen on while the activity is visible
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // Initialize core components
        initializeComponents();

        // Set up the UI layout (programmatic, no XML layout needed)
        setupLayout();

        // Set up gesture detection
        setupGestures();

        // Check and request camera permission
        if (checkCameraPermission()) {
            startCameraSystem();
        } else {
            requestCameraPermission();
        }

        Log.i(TAG, "MainActivity created");
    }

    @Override
    protected void onResume() {
        super.onResume();

        // Resume GL rendering
        if (glSurfaceManager != null) {
            glSurfaceManager.onResume();
        }

        // Resume frame dispatcher
        if (frameDispatcher != null) {
            frameDispatcher.resume();
        }

        // Re-enable immersive mode
        setupImmersiveMode();

        Log.d(TAG, "MainActivity resumed");
    }

    @Override
    protected void onPause() {
        super.onPause();

        // Pause GL rendering
        if (glSurfaceManager != null) {
            glSurfaceManager.onPause();
        }

        // Pause frame dispatcher
        if (frameDispatcher != null) {
            frameDispatcher.pause();
        }

        Log.d(TAG, "MainActivity paused");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        // Shutdown ML pipelines
        releasePipelines();

        // Shutdown frame dispatcher
        if (frameDispatcher != null) {
            frameDispatcher.shutdown();
        }

        // Shutdown camera
        if (cameraManager != null) {
            cameraManager.shutdown();
        }

        // Destroy GL surface
        if (glSurfaceManager != null) {
            glSurfaceManager.destroy();
        }

        // Cleanup overlay renderer
        if (overlayRenderer != null) {
            overlayRenderer.cleanup();
        }

        // Release scene understanding
        if (sceneUnderstanding != null) {
            sceneUnderstanding.release();
        }

        Log.i(TAG, "MainActivity destroyed");
    }

    // ================================================================
    // Permission Handling
    // ================================================================

    /**
     * Checks if the camera permission has been granted.
     *
     * @return true if permission is granted.
     */
    private boolean checkCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * Requests camera permission from the user.
     */
    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(
                this,
                new String[]{Manifest.permission.CAMERA},
                CAMERA_PERMISSION_REQUEST
        );
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.i(TAG, "Camera permission granted");
                startCameraSystem();
            } else {
                Log.w(TAG, "Camera permission denied");
                Toast.makeText(this, "Camera permission is required for Sync Vision",
                        Toast.LENGTH_LONG).show();
            }
        }
    }

    // ================================================================
    // Component Initialization
    // ================================================================

    /**
     * Initializes all core components: camera, GL surface, frame dispatcher,
     * overlay renderer, native processor, and scene understanding.
     * Each component is wrapped in try-catch to prevent a single failure
     * from crashing the entire app.
     */
    private void initializeComponents() {
        // *** ROOT CAUSE FIX #1: Initialize ModelManager BEFORE any pipelines ***
        // Without this, all pipelines throw IllegalStateException("ModelManager not initialized")
        // which is silently caught by try-catch → ALL ML pipelines are null → zero ML processing.
        try {
            ModelManager.getInstance().initialize(this);
            Log.i(TAG, "ModelManager initialized with Context");
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize ModelManager — all ML pipelines will fail!", e);
        }

        try {
            // Native processor (JNI bridge to C++)
            nativeProcessor = new NativeProcessor();
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize NativeProcessor", e);
        }

        try {
            // Camera manager
            cameraManager = new CameraManager(this, this);
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize CameraManager", e);
        }

        try {
            // GL surface manager (OpenGL ES 3.0 rendering)
            glSurfaceManager = new GLSurfaceManager(this);
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize GLSurfaceManager", e);
        }

        try {
            // Frame dispatcher (ML pipeline hub)
            frameDispatcher = new FrameDispatcher();
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize FrameDispatcher", e);
        }

        try {
            // Overlay renderer (high-level rendering API)
            overlayRenderer = new OverlayRenderer(
                    glSurfaceManager.getRenderer(),
                    nativeProcessor
            );
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize OverlayRenderer", e);
        }

        try {
            // Scene understanding engine
            sceneUnderstanding = new SceneUnderstanding();
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize SceneUnderstanding", e);
        }

        // Initialize ML pipelines
        initializePipelines();

        // Vibrator for haptic alerts
        try {
            vibrator = (Vibrator) getSystemService(VIBRATOR_SERVICE);
        } catch (Exception e) {
            Log.e(TAG, "Failed to get Vibrator service", e);
        }

        Log.i(TAG, "Core components initialized");
    }

    /**
     * Initializes all ML pipelines and registers them with the frame dispatcher.
     * Each pipeline is wrapped in try-catch so a single model failure
     * does not prevent the rest from loading.
     */
    private void initializePipelines() {
        // Segmentation pipeline (runs every frame)
        try {
            segmentationPipeline = new SegmentationPipeline();
            segmentationPipeline.initialize();
            frameDispatcher.setSegmentationPipeline(segmentationPipeline);
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize SegmentationPipeline", e);
        }

        // Detection pipeline (runs every frame)
        try {
            detectionPipeline = new DetectionPipeline();
            detectionPipeline.initialize();
            frameDispatcher.setDetectionPipeline(detectionPipeline);
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize DetectionPipeline", e);
        }

        // Depth pipeline (runs every 3rd frame)
        try {
            depthPipeline = new DepthPipeline();
            depthPipeline.initialize();
            frameDispatcher.setDepthPipeline(depthPipeline);
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize DepthPipeline", e);
        }

        // Face pipeline (runs every 2nd frame) â€” requires Context for MediaPipe
        try {
            facePipeline = new FacePipeline();
            facePipeline.initialize(this);
            frameDispatcher.setFacePipeline(facePipeline);
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize FacePipeline", e);
        }

        // Weather pipeline (runs periodically)
        try {
            weatherPipeline = new WeatherPipeline();
            weatherPipeline.initialize();
            frameDispatcher.setWeatherPipeline(weatherPipeline);
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize WeatherPipeline", e);
        }

        // On-demand pipelines
        try {
            ocrPipeline = new OcrPipeline();
            ocrPipeline.initialize();
            frameDispatcher.setOcrPipeline(ocrPipeline);
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize OcrPipeline", e);
        }

        try {
            plantPipeline = new PlantPipeline();
            plantPipeline.initialize();
            frameDispatcher.setPlantPipeline(plantPipeline);
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize PlantPipeline", e);
        }

        try {
            barcodePipeline = new BarcodePipeline();
            barcodePipeline.initialize();
            frameDispatcher.setBarcodePipeline(barcodePipeline);
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize BarcodePipeline", e);
        }

        try {
            landmarkPipeline = new LandmarkPipeline();
            landmarkPipeline.initialize();
            frameDispatcher.setLandmarkPipeline(landmarkPipeline);
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize LandmarkPipeline", e);
        }

        // Set the frame processed listener
        if (frameDispatcher != null) {
            frameDispatcher.setOnFrameProcessedListener(this::onFrameProcessed);
        }

        Log.i(TAG, "ML pipelines initialized");
    }

    /**
     * Releases all ML pipeline resources.
     */
    private void releasePipelines() {
        if (segmentationPipeline != null) segmentationPipeline.release();
        if (detectionPipeline != null) detectionPipeline.release();
        if (depthPipeline != null) depthPipeline.release();
        if (plantPipeline != null) plantPipeline.release();
        if (landmarkPipeline != null) landmarkPipeline.release();
        // WeatherPipeline, FacePipeline, OcrPipeline, BarcodePipeline
        // do not have TFLite interpreters to release
        Log.i(TAG, "ML pipelines released");
    }

    // ================================================================
    // Camera System Startup
    // ================================================================

    /**
     * Starts the complete camera system: connects the camera to the
     * GL surface, begins frame dispatching.
     */
    private void startCameraSystem() {
        if (cameraStarted) {
            Log.w(TAG, "Camera system already started");
            return;
        }

        if (cameraManager == null || glSurfaceManager == null) {
            Log.e(TAG, "Cannot start camera: CameraManager or GLSurfaceManager is null");
            return;
        }

        // Connect the GL surface to the camera preview
        // *** ROOT CAUSE FIX #3: onSurfaceTextureReady fires on the GL thread, ***
        // but startCamera() MUST be called on the main thread. Use runOnUiThread().
        glSurfaceManager.setOnCameraPreviewProviderListener(surfaceTexture -> {
            // When the SurfaceTexture is ready, start the camera on the MAIN thread
            runOnUiThread(() -> {
                androidx.camera.core.Preview.SurfaceProvider provider =
                        glSurfaceManager.getCameraSurfaceProvider();
                if (provider != null && cameraManager != null) {
                    cameraManager.startCamera(provider);
                }
            });
        });

        // Set the frame dispatcher to receive camera frames
        cameraManager.setFrameDispatcher(frameDispatcher);

        // Set camera event listener
        cameraManager.setEventListener(new CameraManager.OnCameraEventListener() {
            @Override
            public void onCameraStarted() {
                Log.i(TAG, "Camera started event received");
                cameraStarted = true;
            }

            @Override
            public void onCameraError(String message) {
                Log.e(TAG, "Camera error: " + message);
                uiHandler.post(() -> Toast.makeText(MainActivity.this,
                        "Camera error: " + message, Toast.LENGTH_SHORT).show());
            }

            @Override
            public void onCameraSwitched(boolean isFrontCamera) {
                Log.i(TAG, "Camera switched: " + (isFrontCamera ? "front" : "back"));
            }

            @Override
            public void onFlashToggled(boolean isEnabled) {
                Log.d(TAG, "Flash: " + (isEnabled ? "ON" : "OFF"));
            }
        });

        // Try to start camera immediately if surface is ready
        androidx.camera.core.Preview.SurfaceProvider provider =
                glSurfaceManager.getCameraSurfaceProvider();
        if (provider != null) {
            cameraManager.startCamera(provider);
            cameraStarted = true;
        }

        Log.i(TAG, "Camera system startup initiated");
    }

    // ================================================================
    // UI Layout Setup
    // ================================================================

    /**
     * Sets up the programmatic UI layout.
     * The layout is a FrameLayout stack:
     *   1. CameraFragment (GLSurfaceView) â€” bottom layer
     *   2. OverlayView (Canvas HUD) â€” middle layer
     *   3. SyncDiagramView â€” floating overlay
     *   4. InfoPanelView â€” slide-in detail panel
     */
    private void setupLayout() {
        // Root FrameLayout (fullscreen)
        rootLayout = new FrameLayout(this);
        rootLayout.setLayoutParams(new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));

        // Layer 1: Camera fragment with GLSurfaceView
        cameraFragment = new CameraFragment();
        cameraFragment.setGLSurfaceView(glSurfaceManager.getSurfaceView());

        getSupportFragmentManager().beginTransaction()
                .add(android.R.id.content, cameraFragment)
                .commit();

        // Layer 2: Canvas-based HUD overlay
        overlayView = new OverlayView(this);
        overlayView.setLayoutParams(new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));
        rootLayout.addView(overlayView);

        // Layer 3: Sync diagram (bottom-right corner, ~200x200dp)
        int diagramSizePx = dpToPx(200);
        FrameLayout.LayoutParams diagramParams = new FrameLayout.LayoutParams(
                diagramSizePx, diagramSizePx);
        diagramParams.gravity = android.view.Gravity.BOTTOM
                | android.view.Gravity.END;
        diagramParams.rightMargin = dpToPx(8);
        diagramParams.bottomMargin = dpToPx(64);

        syncDiagramView = new SyncDiagramView(this);
        syncDiagramView.setLayoutParams(diagramParams);
        rootLayout.addView(syncDiagramView);

        // Layer 4: Info panel (slides in from right)
        int panelWidth = (int) (getResources().getDisplayMetrics().widthPixels * 0.7);
        FrameLayout.LayoutParams infoParams = new FrameLayout.LayoutParams(
                panelWidth,
                FrameLayout.LayoutParams.MATCH_PARENT);
        infoParams.gravity = android.view.Gravity.END;

        infoPanelView = new InfoPanelView(this);
        infoPanelView.setLayoutParams(infoParams);
        infoPanelView.setVisibility(View.GONE);
        infoPanelView.setOnCloseListener(() -> infoPanelView.hide());
        rootLayout.addView(infoPanelView);

        // Set the root layout as content view (on top of fragment)
        // Use addContentView so the fragment stays underneath
        addContentView(rootLayout, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));

        Log.d(TAG, "UI layout setup complete");
    }

    // ================================================================
    // Gesture Handling
    // ================================================================

    /**
     * Sets up touch gesture detection for the fullscreen camera view.
     */
    private void setupGestures() {
        gestureDetector = new GestureDetector(this, new GestureDetector.SimpleOnGestureListener() {

            @Override
            public boolean onSingleTapConfirmed(@NonNull MotionEvent e) {
                // Single tap: trigger on-demand analysis at tap location
                handleTapAnalysis(e.getX(), e.getY());
                return true;
            }

            @Override
            public void onLongPress(@NonNull MotionEvent e) {
                // Long press: toggle settings panel
                openSettings();
            }

            @Override
            public boolean onFling(@NonNull MotionEvent e1,
                                   @NonNull MotionEvent e2,
                                   float velocityX,
                                   float velocityY) {
                float diffX = e2.getX() - e1.getX();
                float diffY = e2.getY() - e1.getY();

                // Horizontal swipe detection (left/right)
                if (Math.abs(diffX) > Math.abs(diffY)
                        && Math.abs(diffX) > SWIPE_THRESHOLD) {
                    if (diffX > 0) {
                        // Swipe right â†’ switch to back camera
                        switchToBackCamera();
                    } else {
                        // Swipe left â†’ switch to front camera
                        switchToFrontCamera();
                    }
                    return true;
                }
                return false;
            }

            @Override
            public boolean onDoubleTap(@NonNull MotionEvent e) {
                // Double tap: toggle flash
                if (cameraManager != null) {
                    boolean flashOn = cameraManager.toggleFlash();
                    if (overlayView != null) {
                        overlayView.setFlashEnabled(flashOn);
                    }
                }
                return true;
            }
        });
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        // Pass touch events to gesture detector first
        if (gestureDetector != null && gestureDetector.onTouchEvent(event)) {
            return true;
        }
        return super.onTouchEvent(event);
    }

    // ================================================================
    // Touch Actions
    // ================================================================

    /**
     * Handles a single tap on the camera view to trigger on-demand
     * analysis (OCR, plant identification, or landmark recognition)
     * at the tapped location.
     *
     * @param x Tap X coordinate in pixels.
     * @param y Tap Y coordinate in pixels.
     */
    private void handleTapAnalysis(float x, float y) {
        Log.d(TAG, String.format(Locale.US, "Tap analysis at (%.0f, %.0f) â€” mode: %s",
                x, y, currentMode.name()));

        // Convert pixel coordinates to normalized [0, 1]
        int viewWidth = overlayView != null ? overlayView.getWidth() : 1;
        int viewHeight = overlayView != null ? overlayView.getHeight() : 1;
        float nx = x / viewWidth;
        float ny = y / viewHeight;

        switch (currentMode) {
            case IDENTIFY:
                // In IDENTIFY mode: trigger OCR, plant ID, and landmark
                if (frameDispatcher != null) {
                    frameDispatcher.requestOcr();
                    frameDispatcher.requestPlantIdentification();
                    frameDispatcher.requestLandmarkRecognition();
                }
                break;

            case PATH:
                // In PATH mode: check for obstacle at tap location
                // (path analysis runs automatically based on depth)
                if (frameDispatcher != null) {
                    frameDispatcher.requestBarcodeScan();
                }
                break;

            case SCAN:
            default:
                // In SCAN mode: show info panel for object at tap location
                showObjectInfoAt(nx, ny);
                // Also trigger on-demand analysis
                if (frameDispatcher != null) {
                    frameDispatcher.requestOcr();
                    frameDispatcher.requestBarcodeScan();
                }
                break;
        }
    }

    /**
     * Shows the info panel for a detected object at the given
     * normalized coordinates, if one exists.
     *
     * @param nx Normalized X [0, 1].
     * @param ny Normalized Y [0, 1].
     */
    private void showObjectInfoAt(float nx, float ny) {
        if (sceneUnderstanding == null) return;

        InferenceResult.DetectedObject obj =
                sceneUnderstanding.getObjectAtPosition(nx, ny);
        if (obj != null && infoPanelView != null) {
            infoPanelView.show(obj);
        }
    }

    /**
     * Switches to the front (selfie) camera.
     */
    private void switchToFrontCamera() {
        if (cameraManager != null) {
            androidx.camera.core.Preview.SurfaceProvider provider =
                    glSurfaceManager.getCameraSurfaceProvider();
            if (provider != null) {
                cameraManager.switchCamera(provider);
            }
        }
    }

    /**
     * Switches to the back camera.
     */
    private void switchToBackCamera() {
        if (cameraManager != null) {
            androidx.camera.core.Preview.SurfaceProvider provider =
                    glSurfaceManager.getCameraSurfaceProvider();
            if (provider != null) {
                cameraManager.switchCamera(provider);
            }
        }
    }

    /**
     * Opens the Settings activity.
     */
    private void openSettings() {
        Intent intent = new Intent(this, SettingsActivity.class);
        startActivity(intent);
    }

    // ================================================================
    // Frame Processing Callback
    // ================================================================

    /**
     * Called by the FrameDispatcher when a new frame has been processed
     * by the ML pipelines. Updates the scene understanding, overlay
     * rendering, and HUD display.
     *
     * This method is called on a background thread, so UI updates
     * must be posted to the main thread via uiHandler.
     *
     * @param result The combined ML scene result.
     */
    private void onFrameProcessed(@NonNull InferenceResult.SceneResult result) {
        lastSceneResult = result;

        // Process scene through the understanding engine
        InferenceResult.SceneResult fusedResult =
                sceneUnderstanding.processScene(result);

        // Update the OpenGL overlay renderer
        if (overlayRenderer != null) {
            overlayRenderer.updateScene(fusedResult);
        }

        // Update the sync diagram
        if (syncDiagramView != null) {
            syncDiagramView.updateFromScene(fusedResult);
        }

        // Post UI updates to the main thread
        uiHandler.post(() -> updateHUD(fusedResult));
    }

    /**
     * Updates the HUD overlay view with the latest scene results.
     * Must be called on the UI thread.
     *
     * @param result The fused scene result.
     */
    private void updateHUD(@NonNull InferenceResult.SceneResult result) {
        if (overlayView == null) return;

        // Update FPS
        overlayView.setFps(result.fps);

        // Update weather status
        String weatherStatus = sceneUnderstanding.getWeatherStatus();
        overlayView.setWeatherStatus(weatherStatus);

        // Update face count
        int faceCount = (result.face != null) ? result.face.faceCount : 0;
        overlayView.setFaceCount(faceCount);

        // Update threat level
        int threatLevel = sceneUnderstanding.getThreatLevel();
        overlayView.setThreatLevel(threatLevel);

        // Trigger haptic alert if threat level increased
        if (threatLevel > previousThreatLevel && threatLevel >= 2) {
            triggerHazardAlert(threatLevel);
        }
        previousThreatLevel = threatLevel;

        // Update mode indicator
        overlayView.setOperatingMode(currentMode);

        // Update detection count for scene summary
        String sceneSummary = sceneUnderstanding.getSceneSummary();
        overlayView.setSceneSummary(sceneSummary);
    }

    // ================================================================
    // Hazard Alert
    // ================================================================

    /**
     * Triggers a haptic (vibration) alert for detected hazards.
     *
     * @param threatLevel The current threat level (0-3).
     */
    private void triggerHazardAlert(int threatLevel) {
        if (vibrator == null || !vibrator.hasVibrator()) return;

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                // Use VibrationEffect on API 26+
                int intensity = Math.min(255, threatLevel * 80);
                VibrationEffect effect = VibrationEffect.createOneShot(
                        HAZARD_VIBRATION_MS * threatLevel,
                        intensity);
                vibrator.vibrate(effect);
            } else {
                // Legacy vibration
                vibrator.vibrate(HAZARD_VIBRATION_MS * threatLevel);
            }
        } catch (Exception e) {
            Log.w(TAG, "Failed to trigger vibration alert", e);
        }
    }

    // ================================================================
    // Immersive Mode
    // ================================================================

    /**
     * Enables fullscreen immersive mode with transparent system bars
     * and sticky immersive mode for the E.D.I.T.H-style experience.
     */
    private void setupImmersiveMode() {
        getWindow().setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            getWindow().setDecorFitsSystemWindows(false);
        }

        // Sticky immersive mode
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                        | View.SYSTEM_UI_FLAG_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            setupImmersiveMode();
        }
    }

    // ================================================================
    // Mode Switching
    // ================================================================

    /**
     * Sets the current operating mode. This affects which ML pipelines
     * are prioritized and what information is shown on the HUD.
     *
     * @param mode The new operating mode.
     */
    public void setOperatingMode(@NonNull OperatingMode mode) {
        this.currentMode = mode;
        if (overlayView != null) {
            overlayView.setOperatingMode(mode);
        }
        Log.i(TAG, "Operating mode set to: " + mode.name());
    }

    /**
     * Returns the current operating mode.
     */
    @NonNull
    public OperatingMode getOperatingMode() {
        return currentMode;
    }

    // ================================================================
    // Utility
    // ================================================================

    /**
     * Converts density-independent pixels (dp) to actual pixels.
     *
     * @param dp Value in dp.
     * @return Value in pixels.
     */
    private int dpToPx(int dp) {
        float density = getResources().getDisplayMetrics().density;
        return Math.round(dp * density);
    }

    /**
     * Returns the last scene result for external access.
     *
     * @return The last processed scene result, or null.
     */
    @Nullable
    public InferenceResult.SceneResult getLastSceneResult() {
        return lastSceneResult;
    }
}
