/**
 * DepthPipeline.java
 *
 * MiDaS v2.1 Small monocular depth estimation pipeline for the Sync Vision app.
 * Takes a camera frame Bitmap, resizes to 256x256 RGB input, runs TFLite
 * inference, and produces a 256x256 relative inverse depth map. Higher values
 * indicate objects closer to the camera.
 *
 * Model: midas_v21_small_256.tflite
 * Input: 1x256x256x3 float32 (RGB, normalized [0, 1])
 * Output: 1x256x256 float32 (relative inverse depth map)
 *
 * Runs every 3rd frame to conserve CPU/GPU resources.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.ml
 * Target SDK: 29+
 */

package com.syncvision.app.ml;

import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.syncvision.app.camera.CameraConfig;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

/**
 * Monocular depth estimation pipeline using MiDaS v2.1 Small.
 * <p>
 * Produces a relative inverse depth map where higher values indicate
 * closer objects. The depth map is used for:
 * <ul>
 *   <li>Approximate distance estimation to detected objects</li>
 *   <li>Ground plane extraction for path finding</li>
 *   <li>Hazard level assessment (nearby obstacles)</li>
 *   <li>Focus-dependent rendering (depth-of-field effects)</li>
 * </ul>
 * <p>
 * Note: MiDaS outputs relative inverse depth, not absolute metric depth.
 * Distance estimation requires a reference point with known distance.
 */
public class DepthPipeline {

    private static final String TAG = "SV-DepthPipeline";

    /** Model filename in assets/models/. */
    private static final String MODEL_NAME = "midas_v21_small_256.tflite";

    /** Input image size (square). */
    private static final int INPUT_SIZE = CameraConfig.DEPTH_INPUT_SIZE; // 256

    // -----------------------------------------------------------------
    // TFLite components
    // -----------------------------------------------------------------

    @Nullable
    private Interpreter interpreter;

    /** Pre-allocated input buffer: 1 x INPUT_SIZE x INPUT_SIZE x 3. */
    private ByteBuffer inputBuffer;

    /** Pre-allocated output buffer: 1 x INPUT_SIZE x INPUT_SIZE. */
    private float[][][] outputBuffer;

    /** Whether the pipeline is initialized and ready. */
    private boolean initialized = false;

    /** Minimum depth value observed (for normalization). */
    private float minDepth = Float.MAX_VALUE;

    /** Maximum depth value observed (for normalization). */
    private float maxDepth = Float.MIN_VALUE;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new DepthPipeline.
     * Call initialize() before processing frames.
     */
    public DepthPipeline() {
    }

    // ================================================================
    // Lifecycle
    // ================================================================

    /**
     * Initializes the pipeline by loading the TFLite model.
     *
     * @return true if initialization succeeded.
     */
    public boolean initialize() {
        if (initialized) {
            Log.w(TAG, "Pipeline already initialized");
            return true;
        }

        try {
            ModelManager modelManager = ModelManager.getInstance();
            interpreter = modelManager.getInterpreter(MODEL_NAME, CameraConfig.NUM_INFERENCE_THREADS);

            // Pre-allocate input buffer
            inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
            inputBuffer.order(ByteOrder.nativeOrder());

            // Pre-allocate output buffer
            outputBuffer = new float[1][INPUT_SIZE][INPUT_SIZE];

            initialized = true;
            Log.i(TAG, "DepthPipeline initialized (input: " + INPUT_SIZE + "x" + INPUT_SIZE + ")");
            return true;

        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize DepthPipeline", e);
            return false;
        }
    }

    /**
     * Releases pipeline resources.
     */
    public void release() {
        if (interpreter != null) {
            ModelManager.getInstance().releaseInterpreter(MODEL_NAME);
            interpreter = null;
        }
        inputBuffer = null;
        outputBuffer = null;
        initialized = false;
        Log.i(TAG, "DepthPipeline released");
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
     * Processes a camera frame through the depth estimation model.
     *
     * @param bitmap The camera frame Bitmap (ARGB_8888).
     * @return DepthResult with the estimated depth map, or null on error.
     */
    @Nullable
    public InferenceResult.DepthResult processFrame(@NonNull Bitmap bitmap) {
        if (!initialized || interpreter == null) {
            Log.w(TAG, "Pipeline not initialized");
            return null;
        }

        try {
            // Step 1: Preprocess input
            preprocessInput(bitmap);

            // Step 2: Run inference
            Object[] inputs = {inputBuffer};
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, outputBuffer);
            interpreter.runForMultipleInputsOutputs(inputs, outputs);

            // Step 3: Post-process output
            return postprocessOutput();

        } catch (Exception e) {
            Log.e(TAG, "Error processing frame", e);
            return null;
        }
    }

    // ================================================================
    // Pre-processing
    // ================================================================

    /**
     * Preprocesses the input bitmap for the depth estimation model.
     * Resizes to INPUT_SIZE x INPUT_SIZE and normalizes RGB.
     * <p>
     * MiDaS expects input normalized using ImageNet mean/std:
     * R: (pixel/255 - 0.485) / 0.229
     * G: (pixel/255 - 0.456) / 0.224
     * B: (pixel/255 - 0.406) / 0.225
     */
    private void preprocessInput(@NonNull Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        if (resized != bitmap) {
            resized.recycle();
        }

        // ImageNet normalization constants
        final float[] MEAN = {0.485f, 0.456f, 0.406f};
        final float[] STD  = {0.229f, 0.224f, 0.225f};

        inputBuffer.rewind();
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;

            // Normalize with ImageNet statistics
            inputBuffer.putFloat((r - MEAN[0]) / STD[0]);
            inputBuffer.putFloat((g - MEAN[1]) / STD[1]);
            inputBuffer.putFloat((b - MEAN[2]) / STD[2]);
        }
    }

    // ================================================================
    // Post-processing
    // ================================================================

    /**
     * Post-processes the model output to produce a depth map.
     * Normalizes the raw output to [0, 1] range for visualization
     * and distance estimation.
     *
     * @return DepthResult with the estimated depth map.
     */
    @NonNull
    private InferenceResult.DepthResult postprocessOutput() {
        float[][] depthMap = new float[INPUT_SIZE][INPUT_SIZE];

        // Find min/max for normalization
        float localMin = Float.MAX_VALUE;
        float localMax = Float.MIN_VALUE;

        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                float val = outputBuffer[0][y][x];
                if (val < localMin) localMin = val;
                if (val > localMax) localMax = val;
            }
        }

        // Update running min/max with smoothing
        if (minDepth == Float.MAX_VALUE) {
            minDepth = localMin;
            maxDepth = localMax;
        } else {
            // Exponential moving average for stable normalization
            minDepth = minDepth * 0.8f + localMin * 0.2f;
            maxDepth = maxDepth * 0.8f + localMax * 0.2f;
        }

        // Normalize to [0, 1] range
        float range = maxDepth - minDepth;
        if (range < 1e-6f) range = 1f;  // Avoid division by zero

        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                depthMap[y][x] = (outputBuffer[0][y][x] - minDepth) / range;
            }
        }

        return new InferenceResult.DepthResult(depthMap, INPUT_SIZE, INPUT_SIZE);
    }

    // ================================================================
    // Distance Estimation
    // ================================================================

    /**
     * Estimates approximate distance to a point in the depth map.
     * Uses a simple pinhole camera model with assumed focal length.
     * <p>
     * WARNING: This is a rough estimate. MiDaS produces relative depth,
     * not absolute metric depth. Accuracy depends on the scene and
     * the assumed reference parameters.
     *
     * @param depthValue    Normalized depth value at the point [0, 1].
     * @param assumedNearM  Assumed distance to the nearest object (meters).
     * @param assumedFarM   Assumed distance to the farthest object (meters).
     * @return Estimated distance in meters.
     */
    public static float estimateDistance(float depthValue, float assumedNearM, float assumedFarM) {
        // MiDaS outputs inverse depth: higher value = closer
        // Map [0, 1] depth to [far, near] distance
        float distance = assumedFarM - depthValue * (assumedFarM - assumedNearM);
        return Math.max(assumedNearM, Math.min(assumedFarM, distance));
    }

    /**
     * Extracts the average depth for a rectangular region.
     * Useful for estimating the distance to a detected object's bounding box.
     *
     * @param depthResult The depth result from this pipeline.
     * @param left   Normalized left coordinate [0, 1].
     * @param top    Normalized top coordinate [0, 1].
     * @param right  Normalized right coordinate [0, 1].
     * @param bottom Normalized bottom coordinate [0, 1].
     * @return Average depth value in the region, or 0 if out of bounds.
     */
    public static float getAverageDepth(@NonNull InferenceResult.DepthResult depthResult,
                                        float left, float top, float right, float bottom) {
        int x1 = Math.max(0, (int) (left * depthResult.width));
        int y1 = Math.max(0, (int) (top * depthResult.height));
        int x2 = Math.min(depthResult.width - 1, (int) (right * depthResult.width));
        int y2 = Math.min(depthResult.height - 1, (int) (bottom * depthResult.height));

        if (x1 >= x2 || y1 >= y2) return 0f;

        float sum = 0f;
        int count = 0;
        for (int y = y1; y <= y2; y++) {
            for (int x = x1; x <= x2; x++) {
                sum += depthResult.depthMap[y][x];
                count++;
            }
        }

        return count > 0 ? sum / count : 0f;
    }
}
