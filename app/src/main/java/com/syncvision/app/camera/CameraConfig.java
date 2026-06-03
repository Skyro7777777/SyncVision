/**
 * CameraConfig.java
 * 
 * Constants for camera configuration in the Sync Vision app.
 * Defines resolution targets, ML input sizes, frame rates,
 * and frame-skip intervals for each ML pipeline.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.camera
 * Target SDK: 29+
 */

package com.syncvision.app.camera;

/**
 * Central configuration for the camera subsystem.
 * All values are tuned for mid-range devices (Snapdragon 660 class)
 * to maintain 30 FPS while running multiple ML pipelines.
 */
public final class CameraConfig {

    // Private constructor — this is a constants-only class
    private CameraConfig() {
        throw new AssertionError("CameraConfig is a constants class; do not instantiate.");
    }

    // -----------------------------------------------------------------
    // Camera Resolution
    // -----------------------------------------------------------------

    /** Target preview/analysis width in pixels. */
    public static final int TARGET_WIDTH = 640;

    /** Target preview/analysis height in pixels. */
    public static final int TARGET_HEIGHT = 480;

    // -----------------------------------------------------------------
    // ML Model Input Sizes
    // -----------------------------------------------------------------

    /**
     * DeepLab v3+ MobileNet V2 input size (square).
     * The segmentation model expects 257x257x3 RGB input.
     */
    public static final int ML_INPUT_SIZE = 257;

    /**
     * COCO SSD MobileNet V2 input size (square).
     * The object detection model expects 300x300x3 RGB input.
     */
    public static final int DETECTION_INPUT_SIZE = 300;

    /**
     * MiDaS v2.1 Small input size (square).
     * The depth estimation model expects 256x256x3 RGB input.
     */
    public static final int DEPTH_INPUT_SIZE = 256;

    // -----------------------------------------------------------------
    // Frame Rate
    // -----------------------------------------------------------------

    /** Target frames per second for the camera feed. */
    public static final int TARGET_FPS = 30;

    // -----------------------------------------------------------------
    // Frame-Skip Intervals
    // -----------------------------------------------------------------

    /**
     * Run depth estimation every Nth frame.
     * Depth estimation is computationally expensive and changes
     * slowly between frames, so we skip frames to save resources.
     * A value of 3 means depth runs at ~10 FPS on a 30 FPS camera.
     */
    public static final int DEPTH_FRAME_SKIP = 3;

    /**
     * Run face detection every Nth frame.
     * Face detection is moderately expensive; running every 2nd
     * frame provides good responsiveness while saving CPU cycles.
     * A value of 2 means face runs at ~15 FPS on a 30 FPS camera.
     */
    public static final int FACE_FRAME_SKIP = 2;

    /**
     * Run weather classification every N seconds.
     * Weather conditions change slowly, so we throttle this
     * pipeline to avoid unnecessary computation.
     */
    public static final float WEATHER_INTERVAL_SECONDS = 2.5f;

    // -----------------------------------------------------------------
    // Detection Thresholds
    // -----------------------------------------------------------------

    /** Default confidence threshold for object detection (0.0 - 1.0). */
    public static final float DETECTION_CONFIDENCE_THRESHOLD = 0.5f;

    /** Default IoU threshold for Non-Maximum Suppression. */
    public static final float NMS_IOU_THRESHOLD = 0.45f;

    /** Maximum number of detected objects to return per frame. */
    public static final int MAX_DETECTIONS = 10;

    // -----------------------------------------------------------------
    // Threading
    // -----------------------------------------------------------------

    /** Number of threads for TFLite inference (optimized for big.LITTLE). */
    public static final int NUM_INFERENCE_THREADS = 4;

    /** Ring buffer capacity for storing latest ML results. */
    public static final int RESULT_RING_BUFFER_SIZE = 3;

    // -----------------------------------------------------------------
    // Camera IDs
    // -----------------------------------------------------------------

    /** Lens facing constant for the rear (back) camera. */
    public static final int LENS_FACING_BACK = 0;

    /** Lens facing constant for the front (selfie) camera. */
    public static final int LENS_FACING_FRONT = 1;
}
