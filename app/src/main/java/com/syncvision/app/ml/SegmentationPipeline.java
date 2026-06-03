/**
 * SegmentationPipeline.java
 *
 * DeepLab v3+ MobileNet V2 semantic segmentation pipeline for the
 * Sync Vision app. Takes a camera frame Bitmap, resizes to 257x257
 * RGB input, runs TFLite inference, and produces a per-pixel class
 * label mask with 21 COCO Pascal VOC classes.
 *
 * Model: deeplab_v3_mnv2_257.tflite
 * Input: 1x257x257x3 float32 (RGB, normalized [0, 1])
 * Output: 1x257x257x21 float32 (per-pixel class probabilities)
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
 * Semantic segmentation pipeline using DeepLab v3+ MobileNet V2.
 * <p>
 * Processes every camera frame to produce a per-pixel class label mask
 * identifying objects like person, car, road, etc. The mask is then
 * used by the OpenGL renderer to draw green outlines around detected
 * object boundaries.
 * <p>
 * The 21 COCO Pascal VOC class labels are:
 * background, aeroplane, bicycle, bird, boat, bottle, bus, car,
 * cat, chair, cow, diningtable, dog, horse, motorbike, person,
 * pottedplant, sheep, sofa, train, tv
 */
public class SegmentationPipeline {

    private static final String TAG = "SV-SegmentationPipeline";

    /** Model filename in assets/models/. */
    private static final String MODEL_NAME = "deeplab_v3_mnv2_257.tflite";

    /** Number of segmentation classes (Pascal VOC). */
    private static final int NUM_CLASSES = 21;

    /** Input image size (square). */
    private static final int INPUT_SIZE = CameraConfig.ML_INPUT_SIZE; // 257

    /** COCO Pascal VOC class labels for DeepLab v3+. */
    private static final String[] CLASS_LABELS = {
            "background",   // 0
            "aeroplane",    // 1
            "bicycle",      // 2
            "bird",         // 3
            "boat",         // 4
            "bottle",       // 5
            "bus",          // 6
            "car",          // 7
            "cat",          // 8
            "chair",        // 9
            "cow",          // 10
            "diningtable",  // 11
            "dog",          // 12
            "horse",        // 13
            "motorbike",    // 14
            "person",       // 15
            "pottedplant",  // 16
            "sheep",        // 17
            "sofa",         // 18
            "train",        // 19
            "tv"            // 20
    };

    // -----------------------------------------------------------------
    // TFLite components
    // -----------------------------------------------------------------

    @Nullable
    private Interpreter interpreter;

    /** Pre-allocated input buffer: 1 x INPUT_SIZE x INPUT_SIZE x 3. */
    private ByteBuffer inputBuffer;

    /** Pre-allocated output array: 1 x INPUT_SIZE x INPUT_SIZE x NUM_CLASSES. */
    private float[][][][] outputBuffer;

    /** Whether the pipeline is initialized and ready. */
    private boolean initialized = false;

    /** Class label map for inclusion in results. */
    private final Map<Integer, String> classLabelMap;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new SegmentationPipeline.
     * Call initialize() before processing frames.
     */
    public SegmentationPipeline() {
        // Build class label map
        classLabelMap = new HashMap<>();
        for (int i = 0; i < CLASS_LABELS.length; i++) {
            classLabelMap.put(i, CLASS_LABELS[i]);
        }
    }

    // ================================================================
    // Lifecycle
    // ================================================================

    /**
     * Initializes the pipeline by loading the TFLite model.
     * Must be called before processFrame().
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

            // Pre-allocate input buffer (float32, native byte order)
            inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
            inputBuffer.order(ByteOrder.nativeOrder());

            // Pre-allocate output buffer
            outputBuffer = new float[1][INPUT_SIZE][INPUT_SIZE][NUM_CLASSES];

            initialized = true;
            Log.i(TAG, "SegmentationPipeline initialized (input: " + INPUT_SIZE + "x" + INPUT_SIZE + ")");
            return true;

        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize SegmentationPipeline", e);
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
        Log.i(TAG, "SegmentationPipeline released");
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
     * Processes a camera frame through the segmentation model.
     *
     * @param bitmap The camera frame Bitmap (ARGB_8888).
     * @return SegmentationResult with per-pixel class labels, or null on error.
     */
    @Nullable
    public InferenceResult.SegmentationResult processFrame(@NonNull Bitmap bitmap) {
        if (!initialized || interpreter == null) {
            Log.w(TAG, "Pipeline not initialized");
            return null;
        }

        try {
            // Step 1: Resize and normalize the input bitmap
            preprocessInput(bitmap);

            // Step 2: Run inference
            Object[] inputs = {inputBuffer};
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, outputBuffer);
            interpreter.runForMultipleInputsOutputs(inputs, outputs);

            // Step 3: Post-process output to get per-pixel class labels
            return postprocessOutput(bitmap.getWidth(), bitmap.getHeight());

        } catch (Exception e) {
            Log.e(TAG, "Error processing frame", e);
            return null;
        }
    }

    // ================================================================
    // Pre-processing
    // ================================================================

    /**
     * Preprocesses the input bitmap for the segmentation model.
     * Resizes to INPUT_SIZE x INPUT_SIZE and normalizes RGB to [0, 1].
     *
     * @param bitmap The original camera frame.
     */
    private void preprocessInput(@NonNull Bitmap bitmap) {
        // Resize to model input size
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);

        // Convert to int pixel array
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        // If the resized bitmap is different from the original, recycle it
        if (resized != bitmap) {
            resized.recycle();
        }

        // Fill input buffer with normalized float32 RGB values
        inputBuffer.rewind();
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            // Normalize to [0, 1] range
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;
            inputBuffer.putFloat(r);
            inputBuffer.putFloat(g);
            inputBuffer.putFloat(b);
        }
    }

    // ================================================================
    // Post-processing
    // ================================================================

    /**
     * Post-processes the model output to produce a per-pixel class label mask.
     * For each pixel, finds the class with the highest probability
     * (argmax over the NUM_CLASSES dimension).
     *
     * @param originalWidth  Width of the original camera frame.
     * @param originalHeight Height of the original camera frame.
     * @return SegmentationResult with class labels for each pixel.
     */
    @NonNull
    private InferenceResult.SegmentationResult postprocessOutput(int originalWidth,
                                                                  int originalHeight) {
        // Extract argmax class labels from output probabilities
        int[][] mask = new int[INPUT_SIZE][INPUT_SIZE];

        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                int bestClass = 0;
                float bestProb = outputBuffer[0][y][x][0];
                for (int c = 1; c < NUM_CLASSES; c++) {
                    if (outputBuffer[0][y][x][c] > bestProb) {
                        bestProb = outputBuffer[0][y][x][c];
                        bestClass = c;
                    }
                }
                mask[y][x] = bestClass;
            }
        }

        return new InferenceResult.SegmentationResult(
                mask, INPUT_SIZE, INPUT_SIZE, classLabelMap);
    }

    // ================================================================
    // Utility
    // ================================================================

    /**
     * Returns the human-readable label for a given class ID.
     *
     * @param classId The class ID (0-20).
     * @return The class label, or "unknown" if out of range.
     */
    @NonNull
    public static String getClassName(int classId) {
        if (classId >= 0 && classId < CLASS_LABELS.length) {
            return CLASS_LABELS[classId];
        }
        return "unknown";
    }

    /**
     * Returns the class ID for a given label name (case-insensitive).
     *
     * @param name The class name.
     * @return The class ID, or 0 (background) if not found.
     */
    public static int getClassId(@NonNull String name) {
        for (int i = 0; i < CLASS_LABELS.length; i++) {
            if (CLASS_LABELS[i].equalsIgnoreCase(name)) {
                return i;
            }
        }
        return 0;
    }

    /**
     * Returns all class labels as an array.
     */
    @NonNull
    public static String[] getClassLabels() {
        return CLASS_LABELS.clone();
    }
}
