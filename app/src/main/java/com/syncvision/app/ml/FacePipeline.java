/**
 * FacePipeline.java
 *
 * MediaPipe BlazeFace face detection pipeline for the Sync Vision app.
 * Uses MediaPipe Tasks Vision FaceDetector for lightweight, on-device
 * face detection. This pipeline counts and outlines faces but performs
 * NO facial recognition or identity extraction â€” privacy is paramount.
 *
 * Input: camera frame Bitmap
 * Output: FaceResult (face bounding rectangles + count)
 *
 * Runs every 2nd frame to balance responsiveness with performance.
 *
 * Sync Vision â€” Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.ml
 * Target SDK: 29+
 */

package com.syncvision.app.ml;

import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector;
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult;

import java.util.ArrayList;
import java.util.List;

/**
 * Face detection pipeline using MediaPipe BlazeFace.
 * <p>
 * <b>PRIVACY NOTICE:</b> This pipeline only detects the presence and
 * location of faces. It does NOT perform facial recognition, identity
 * extraction, or any form of biometric identification. Face data is
 * never stored, transmitted, or used for identification purposes.
 * <p>
 * The face count and bounding rectangles are used for:
 * <ul>
 *   <li>Awareness: alerting the user to the presence of people</li>
 *   <li>Social context: displaying face count in the HUD</li>
 *   <li>Anonymization guidance: indicating regions to blur if needed</li>
 * </ul>
 */
public class FacePipeline {

    private static final String TAG = "SV-FacePipeline";

    /** Minimum confidence threshold for face detection. */
    private static final float MIN_CONFIDENCE = 0.5f;

    /** Model filename in assets/models/. */
    private static final String MODEL_NAME = "blaze_face_short_range.tflite";

    // -----------------------------------------------------------------
    // MediaPipe components
    // -----------------------------------------------------------------

    @Nullable
    private FaceDetector faceDetector;

    /** Whether the pipeline is initialized and ready. */
    private boolean initialized = false;

    /** Last frame width (for coordinate normalization). */
    private int lastFrameWidth = 0;

    /** Last frame height (for coordinate normalization). */
    private int lastFrameHeight = 0;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new FacePipeline.
     * Call initialize() before processing frames.
     */
    public FacePipeline() {
    }

    // ================================================================
    // Lifecycle
    // ================================================================

    /**
     * Initializes the pipeline by setting up the MediaPipe FaceDetector.
     *
     * @param context Android context for asset access.
     * @return true if initialization succeeded.
     */
    public boolean initialize(@NonNull android.content.Context context) {
        if (initialized) {
            Log.w(TAG, "Pipeline already initialized");
            return true;
        }

        try {
            // Build the FaceDetector using MediaPipe Tasks Vision API
            FaceDetector.FaceDetectorOptions options =
                    FaceDetector.FaceDetectorOptions.builder()
                            .setRunningMode(RunningMode.IMAGE)
                            .setMinDetectionConfidence(MIN_CONFIDENCE)
                            .build();

            // Load model from assets
            String modelPath = MODEL_NAME;
            faceDetector = FaceDetector.createFromOptions(context, options);

            initialized = true;
            Log.i(TAG, "FacePipeline initialized (MediaPipe BlazeFace)");
            return true;

        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize FacePipeline", e);
            return false;
        }
    }

    /**
     * Releases pipeline resources.
     */
    public void release() {
        if (faceDetector != null) {
            faceDetector.close();
            faceDetector = null;
        }
        initialized = false;
        Log.i(TAG, "FacePipeline released");
    }

    /**
     * Returns whether the pipeline is initialized and ready.
     */
    public boolean isInitialized() {
        return initialized;
    }

    // ================================================================
    // Frame Processing
    // ================================================================

    /**
     * Processes a camera frame for face detection.
     *
     * @param bitmap The camera frame Bitmap (ARGB_8888).
     * @return FaceResult with detected face rectangles and count, or null on error.
     */
    @Nullable
    public InferenceResult.FaceResult processFrame(@NonNull Bitmap bitmap) {
        if (!initialized || faceDetector == null) {
            Log.w(TAG, "Pipeline not initialized");
            return null;
        }

        try {
            lastFrameWidth = bitmap.getWidth();
            lastFrameHeight = bitmap.getHeight();

            // Create MediaPipe Image from Bitmap
            com.google.mediapipe.framework.image.MPImage mpImage =
                    new com.google.mediapipe.framework.image.BitmapImageBuilder(bitmap).build();

            // Run face detection
            FaceDetectorResult result = faceDetector.detect(mpImage);

            // Convert results to normalized FaceResult
            return convertResult(result);

        } catch (Exception e) {
            Log.e(TAG, "Error processing frame", e);
            return null;
        }
    }

    // ================================================================
    // Result Conversion
    // ================================================================

    /**
     * Converts MediaPipe FaceDetectorResult to our FaceResult format.
     * Normalizes bounding box coordinates from pixel space to [0, 1].
     *
     * @param result MediaPipe face detection result.
     * @return Normalized FaceResult.
     */
    @NonNull
    private InferenceResult.FaceResult convertResult(@NonNull FaceDetectorResult result) {
        List<InferenceResult.RectF> faceRects = new ArrayList<>();

        if (result.detections() != null) {
            for (com.google.mediapipe.tasks.components.containers.Detection detection :
                    result.detections()) {
                if (detection.boundingBox() != null) {
                    com.google.mediapipe.tasks.components.containers.RectF bbox = detection.boundingBox();

                    // Normalize from pixel coordinates to [0, 1]
                    float left = bbox.left / lastFrameWidth;
                    float top = bbox.top / lastFrameHeight;
                    float right = bbox.right / lastFrameWidth;
                    float bottom = bbox.bottom / lastFrameHeight;

                    // Clamp to [0, 1]
                    left = Math.max(0f, Math.min(1f, left));
                    top = Math.max(0f, Math.min(1f, top));
                    right = Math.max(0f, Math.min(1f, right));
                    bottom = Math.max(0f, Math.min(1f, bottom));

                    faceRects.add(new InferenceResult.RectF(left, top, right, bottom));
                }
            }
        }

        return new InferenceResult.FaceResult(faceRects);
    }

    // ================================================================
    // Privacy
    // ================================================================

    /**
     * Confirms that this pipeline does NOT collect any biometric or
     * identity data. This method exists for auditing and compliance.
     *
     * @return Always returns false â€” no identity data is ever collected.
     */
    @SuppressWarnings("unused")
    public static boolean collectsIdentityData() {
        // PRIVACY: This pipeline NEVER collects or processes identity data.
        // It only detects the presence and location of faces.
        return false;
    }
}
