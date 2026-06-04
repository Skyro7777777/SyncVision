/**
 * PlantPipeline.java
 *
 * iNaturalist MobileNet TFLite plant identification pipeline for the
 * Sync Vision app. Identifies plant species from a cropped region of
 * the camera frame. Operates on-demand only (tap-triggered) to avoid
 * unnecessary computation.
 *
 * Model: inaturalist_mnv2_int8.tflite
 * Input: 224x224x3 float32 (RGB, normalized [0, 1])
 * Output: 1xNUM_SPECIES float32 (species probability distribution)
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
 * On-demand plant identification pipeline using iNaturalist MobileNet.
 * <p>
 * This pipeline is NOT run continuously — it only activates when the
 * user explicitly taps on a plant or selects a region of interest.
 * The model identifies the most likely plant species from a cropped
 * region of the camera frame.
 * <p>
 * The iNaturalist dataset covers thousands of plant species commonly
 * encountered worldwide. Top-1 accuracy is ~60% on mobile-optimized models,
 * which is sufficient for general awareness and educational purposes.
 */
public class PlantPipeline {

    private static final String TAG = "SV-PlantPipeline";

    /** Model filename in assets/models/. Must match the actual .tflite file name. */
    private static final String MODEL_NAME = "inaturalist_mnv2_int8.tflite";

    /** Input image size (square) for the iNaturalist model. */
    private static final int INPUT_SIZE = 224;

    /** Default region crop padding fraction (adds margin around the tap point). */
    private static final float DEFAULT_CROP_PADDING = 0.15f;

    /** Minimum confidence to report a species identification. */
    private static final float MIN_CONFIDENCE = 0.15f;

    // -----------------------------------------------------------------
    // TFLite components
    // -----------------------------------------------------------------

    @Nullable
    private Interpreter interpreter;

    /** Pre-allocated input buffer. */
    private ByteBuffer inputBuffer;

    /** Pre-allocated output buffer. Size depends on model; we'll allocate dynamically. */
    @Nullable
    private float[][] outputBuffer;

    /** Number of output classes (species). Determined from model metadata. */
    private int numSpecies = 1000;  // Default; updated after model load

    /** Whether the pipeline is initialized and ready. */
    private boolean initialized = false;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new PlantPipeline.
     * Call initialize() before processing frames.
     */
    public PlantPipeline() {
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

            if (!modelManager.isModelAvailable(MODEL_NAME)) {
                Log.w(TAG, "Plant model not available: " + MODEL_NAME);
                return false;
            }

            interpreter = modelManager.getInterpreter(MODEL_NAME,
                    CameraConfig.NUM_INFERENCE_THREADS);

            // Get output tensor shape to determine number of species
            int[] outputShape = interpreter.getOutputTensor(0).shape();
            if (outputShape.length > 1) {
                numSpecies = outputShape[1];
            }

            // Pre-allocate input buffer
            inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
            inputBuffer.order(ByteOrder.nativeOrder());

            // Pre-allocate output buffer
            outputBuffer = new float[1][numSpecies];

            initialized = true;
            Log.i(TAG, "PlantPipeline initialized (input: " + INPUT_SIZE + "x" + INPUT_SIZE
                    + ", species: " + numSpecies + ")");
            return true;

        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize PlantPipeline", e);
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
        Log.i(TAG, "PlantPipeline released");
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
     * Processes a cropped region of the camera frame for plant identification.
     * The crop region is typically selected by the user tapping on a plant.
     *
     * @param bitmap The full camera frame Bitmap (ARGB_8888).
     * @param cropCenterX Normalized center X of the crop region [0, 1].
     * @param cropCenterY Normalized center Y of the crop region [0, 1].
     * @return PlantResult with species identification, or null on error.
     */
    @Nullable
    public InferenceResult.PlantResult processFrame(@NonNull Bitmap bitmap,
                                                     float cropCenterX,
                                                     float cropCenterY) {
        if (!initialized || interpreter == null) {
            Log.w(TAG, "Pipeline not initialized");
            return null;
        }

        try {
            // Extract the crop region around the tap point
            Bitmap cropRegion = extractCropRegion(bitmap, cropCenterX, cropCenterY);

            // Preprocess and run inference
            preprocessInput(cropRegion);

            Object[] inputs = {inputBuffer};
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, outputBuffer);
            interpreter.runForMultipleInputsOutputs(inputs, outputs);

            // Recycle crop region
            if (cropRegion != bitmap) {
                cropRegion.recycle();
            }

            // Post-process: find top species
            return postprocessOutput();

        } catch (Exception e) {
            Log.e(TAG, "Error processing frame", e);
            return null;
        }
    }

    /**
     * Processes the full camera frame for plant identification.
     * Convenience method that uses the center of the frame.
     *
     * @param bitmap The camera frame Bitmap (ARGB_8888).
     * @return PlantResult with species identification, or null on error.
     */
    @Nullable
    public InferenceResult.PlantResult processFrame(@NonNull Bitmap bitmap) {
        return processFrame(bitmap, 0.5f, 0.5f);
    }

    // ================================================================
    // Crop Region Extraction
    // ================================================================

    /**
     * Extracts a square crop region centered on the given point.
     * Adds padding around the point for better identification.
     *
     * @param bitmap    The full camera frame.
     * @param centerX   Normalized center X [0, 1].
     * @param centerY   Normalized center Y [0, 1].
     * @return A cropped Bitmap region.
     */
    @NonNull
    private Bitmap extractCropRegion(@NonNull Bitmap bitmap, float centerX, float centerY) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        // Calculate crop size (square, with padding)
        int cropSize = (int) (Math.min(width, height) * (0.3f + DEFAULT_CROP_PADDING));

        // Center the crop on the tap point
        int cx = (int) (centerX * width);
        int cy = (int) (centerY * height);

        int left = Math.max(0, cx - cropSize / 2);
        int top = Math.max(0, cy - cropSize / 2);
        int right = Math.min(width, left + cropSize);
        int bottom = Math.min(height, top + cropSize);

        // Adjust if we hit boundaries
        if (right - left < cropSize) {
            left = Math.max(0, right - cropSize);
        }
        if (bottom - top < cropSize) {
            top = Math.max(0, bottom - cropSize);
        }

        return Bitmap.createBitmap(bitmap, left, top,
                Math.min(cropSize, right - left),
                Math.min(cropSize, bottom - top));
    }

    // ================================================================
    // Pre-processing
    // ================================================================

    /**
     * Preprocesses the crop region for the iNaturalist model.
     * Resizes to INPUT_SIZE x INPUT_SIZE and normalizes RGB to [0, 1].
     */
    private void preprocessInput(@NonNull Bitmap cropRegion) {
        Bitmap resized = Bitmap.createScaledBitmap(cropRegion,
                INPUT_SIZE, INPUT_SIZE, true);

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        if (resized != cropRegion) {
            resized.recycle();
        }

        inputBuffer.rewind();
        for (int pixel : pixels) {
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
     * Post-processes the model output to find the top-1 species.
     *
     * @return PlantResult with the identified species, or null if confidence is too low.
     */
    @Nullable
    private InferenceResult.PlantResult postprocessOutput() {
        if (outputBuffer == null) return null;

        // Find the species with highest probability
        int bestIdx = 0;
        float bestProb = outputBuffer[0][0];
        int secondIdx = -1;
        float secondProb = 0f;

        for (int i = 1; i < numSpecies; i++) {
            if (outputBuffer[0][i] > bestProb) {
                secondProb = bestProb;
                secondIdx = bestIdx;
                bestProb = outputBuffer[0][i];
                bestIdx = i;
            } else if (outputBuffer[0][i] > secondProb) {
                secondProb = outputBuffer[0][i];
                secondIdx = i;
            }
        }

        // Filter by minimum confidence
        if (bestProb < MIN_CONFIDENCE) {
            Log.d(TAG, "Plant identification confidence too low: " + bestProb);
            return new InferenceResult.PlantResult(
                    "Unknown", bestProb, "Confidence too low for reliable identification");
        }

        // Map species index to name
        // In production, this would use a label file loaded from assets.
        // For now, use a placeholder format.
        String speciesName = getSpeciesName(bestIdx);
        String description = buildDescription(speciesName, bestProb, secondIdx);

        return new InferenceResult.PlantResult(speciesName, bestProb, description);
    }

    // ================================================================
    // Species Name Mapping
    // ================================================================

    /**
     * Maps a species index to a human-readable name.
     * In production, this loads from a label file (e.g., plant_labels.txt).
     *
     * @param index The species index from the model output.
     * @return The species name.
     */
    @NonNull
    private String getSpeciesName(int index) {
        // Placeholder: In production, load from assets/models/plant_labels.txt
        // which maps indices to species names like "Rosa rubiginosa"
        return "Species_" + index;
    }

    /**
     * Builds a brief description string for the identified plant.
     */
    @NonNull
    private String buildDescription(@NonNull String species, float confidence,
                                    int secondIdx) {
        StringBuilder sb = new StringBuilder();
        sb.append(species);
        sb.append(String.format(" (%.0f%% confidence)", confidence * 100));

        if (secondIdx >= 0 && confidence < 0.7f) {
            sb.append(". Also possible: ");
            sb.append(getSpeciesName(secondIdx));
        }

        return sb.toString();
    }
}
