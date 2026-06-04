/**
 * WeatherPipeline.java
 *
 * Weather/sky condition classification pipeline for the Sync Vision app.
 * Analyzes the upper portion of the camera frame (sky region) to classify
 * weather conditions. Uses a custom TFLite model when available, or
 * falls back to a heuristic-based approach using image statistics.
 *
 * Input: upper 40% of camera frame (sky region)
 * Output: WeatherResult (condition, confidence, sky features)
 *
 * Weather classes: CLEAR, PARTLY_CLOUDY, CLOUDY, OVERCAST, RAINY,
 *                  FOGGY, SUNSET_SUNRISE, STORMY
 *
 * Runs every ~2.5 seconds (weather changes slowly).
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.ml
 * Target SDK: 29+
 */

package com.syncvision.app.ml;

import android.graphics.Bitmap;
import android.graphics.Color;
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
 * Sky weather classification pipeline.
 * <p>
 * Analyzes the sky region of the camera frame to determine weather
 * conditions. This information is used for:
 * <ul>
 *   <li>Scene understanding and context</li>
 *   <li>Photography recommendations (exposure, white balance)</li>
 *   <li>Hazard warnings (storm, fog)</li>
 *   <li>Path finding adjustments (rain makes surfaces slippery)</li>
 * </ul>
 * <p>
 * If the custom weather model is not available, falls back to a
 * heuristic approach based on sky color statistics (average brightness,
 * color distribution, variance).
 */
public class WeatherPipeline {

    private static final String TAG = "SV-WeatherPipeline";

    /** Model filename in assets/models/. Must match the actual .tflite file name. */
    private static final String MODEL_NAME = "weather_classifier_int8.tflite";

    /** Number of weather condition classes. */
    private static final int NUM_CLASSES = 8;

    /** Input size for the weather model (square). */
    private static final int MODEL_INPUT_SIZE = 224;

    /** Fraction of the frame from the top to use as sky region. */
    private static final float SKY_REGION_TOP_FRACTION = 0.4f;

    /**
     * Weather condition class labels.
     * These represent distinct sky conditions relevant to outdoor awareness.
     */
    public static final String[] WEATHER_CLASSES = {
            "CLEAR",            // 0 - Clear blue sky
            "PARTLY_CLOUDY",    // 1 - Some clouds, mostly blue
            "CLOUDY",           // 2 - Significant cloud cover
            "OVERCAST",         // 3 - Fully overcast, gray sky
            "RAINY",            // 4 - Rain visible
            "FOGGY",            // 5 - Fog, mist, low visibility
            "SUNSET_SUNRISE",   // 6 - Warm orange/pink sky
            "STORMY"            // 7 - Dark storm clouds
    };

    // -----------------------------------------------------------------
    // TFLite components
    // -----------------------------------------------------------------

    @Nullable
    private Interpreter interpreter;

    /** Pre-allocated input buffer. */
    private ByteBuffer inputBuffer;

    /** Pre-allocated output buffer: 1 x NUM_CLASSES. */
    private float[][] outputBuffer;

    /** Whether the custom model is loaded. */
    private boolean modelLoaded = false;

    /** Whether the pipeline is initialized and ready. */
    private boolean initialized = false;

    /** Last computed weather result (cached for periodic access). */
    @Nullable
    private InferenceResult.WeatherResult lastResult;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new WeatherPipeline.
     * Call initialize() before processing frames.
     */
    public WeatherPipeline() {
    }

    // ================================================================
    // Lifecycle
    // ================================================================

    /**
     * Initializes the pipeline. Attempts to load the custom weather model,
     * but falls back to heuristic analysis if the model is unavailable.
     *
     * @return true if initialization succeeded (with or without model).
     */
    public boolean initialize() {
        if (initialized) {
            Log.w(TAG, "Pipeline already initialized");
            return true;
        }

        // Try to load the custom weather classification model
        try {
            ModelManager modelManager = ModelManager.getInstance();
            if (modelManager.isModelAvailable(MODEL_NAME)) {
                interpreter = modelManager.getInterpreter(MODEL_NAME,
                        CameraConfig.NUM_INFERENCE_THREADS);

                inputBuffer = ByteBuffer.allocateDirect(
                        MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3 * 4);
                inputBuffer.order(ByteOrder.nativeOrder());

                outputBuffer = new float[1][NUM_CLASSES];

                modelLoaded = true;
                Log.i(TAG, "WeatherPipeline initialized with custom model");
            } else {
                Log.i(TAG, "Weather model not found, using heuristic fallback");
            }
        } catch (IOException e) {
            Log.w(TAG, "Weather model load failed, using heuristic fallback", e);
        }

        // Always mark as initialized — heuristic fallback always works
        initialized = true;
        return true;
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
        modelLoaded = false;
        initialized = false;
        Log.i(TAG, "WeatherPipeline released");
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
     * Processes a camera frame for weather classification.
     * Extracts the sky region (upper 40%) and classifies it.
     *
     * @param bitmap The camera frame Bitmap (ARGB_8888).
     * @return WeatherResult with the classified condition, or null on error.
     */
    @Nullable
    public InferenceResult.WeatherResult processFrame(@NonNull Bitmap bitmap) {
        if (!initialized) {
            Log.w(TAG, "Pipeline not initialized");
            return null;
        }

        try {
            // Extract sky region from the upper portion of the frame
            Bitmap skyRegion = extractSkyRegion(bitmap);

            InferenceResult.WeatherResult result;

            if (modelLoaded && interpreter != null) {
                // Use the custom TFLite model
                result = processWithModel(skyRegion);
            } else {
                // Fall back to heuristic analysis
                result = processWithHeuristic(skyRegion);
            }

            // Recycle the sky region bitmap if it's a copy
            if (skyRegion != bitmap) {
                skyRegion.recycle();
            }

            lastResult = result;
            return result;

        } catch (Exception e) {
            Log.e(TAG, "Error processing frame", e);
            return lastResult;  // Return last known result on error
        }
    }

    // ================================================================
    // Sky Region Extraction
    // ================================================================

    /**
     * Extracts the sky region from the upper portion of the frame.
     *
     * @param bitmap The full camera frame.
     * @return A Bitmap containing only the sky region.
     */
    @NonNull
    private Bitmap extractSkyRegion(@NonNull Bitmap bitmap) {
        int skyHeight = (int) (bitmap.getHeight() * SKY_REGION_TOP_FRACTION);
        skyHeight = Math.max(1, skyHeight);

        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), skyHeight);
    }

    // ================================================================
    // Model-based Classification
    // ================================================================

    /**
     * Processes the sky region using the custom TFLite model.
     */
    @Nullable
    private InferenceResult.WeatherResult processWithModel(@NonNull Bitmap skyRegion) {
        // Resize to model input size
        Bitmap resized = Bitmap.createScaledBitmap(skyRegion,
                MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true);

        int[] pixels = new int[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE];
        resized.getPixels(pixels, 0, MODEL_INPUT_SIZE, 0, 0,
                MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

        if (resized != skyRegion) {
            resized.recycle();
        }

        // Fill input buffer
        inputBuffer.rewind();
        for (int pixel : pixels) {
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;
            inputBuffer.putFloat(r);
            inputBuffer.putFloat(g);
            inputBuffer.putFloat(b);
        }

        // Run inference
        Object[] inputs = {inputBuffer};
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, outputBuffer);
        interpreter.runForMultipleInputsOutputs(inputs, outputs);

        // Find the class with highest probability
        int bestClass = 0;
        float bestProb = outputBuffer[0][0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (outputBuffer[0][i] > bestProb) {
                bestProb = outputBuffer[0][i];
                bestClass = i;
            }
        }

        // Extract top-3 sky features for the result
        String[] features = extractTopFeatures(outputBuffer[0]);

        return new InferenceResult.WeatherResult(
                WEATHER_CLASSES[bestClass],
                bestProb,
                features
        );
    }

    /**
     * Extracts the top-3 weather feature names from class probabilities.
     */
    @NonNull
    private String[] extractTopFeatures(@NonNull float[] probabilities) {
        // Find top 3 indices
        int[] topIndices = new int[3];
        float[] topProbs = new float[3];

        for (int i = 0; i < probabilities.length; i++) {
            for (int j = 0; j < 3; j++) {
                if (probabilities[i] > topProbs[j]) {
                    // Shift down
                    for (int k = 2; k > j; k--) {
                        topIndices[k] = topIndices[k - 1];
                        topProbs[k] = topProbs[k - 1];
                    }
                    topIndices[j] = i;
                    topProbs[j] = probabilities[i];
                    break;
                }
            }
        }

        String[] features = new String[Math.min(3, WEATHER_CLASSES.length)];
        for (int i = 0; i < features.length; i++) {
            features[i] = WEATHER_CLASSES[topIndices[i]];
        }
        return features;
    }

    // ================================================================
    // Heuristic-based Classification (Fallback)
    // ================================================================

    /**
     * Processes the sky region using heuristic image statistics.
     * Analyzes brightness, blue ratio, saturation, and variance
     * to estimate weather conditions.
     * <p>
     * This is a simplified approach that works reasonably well for
     * common conditions (clear sky, cloudy, foggy, sunset) but
     * may misclassify ambiguous conditions.
     *
     * @param skyRegion The sky region bitmap.
     * @return A WeatherResult based on heuristic analysis.
     */
    @NonNull
    private InferenceResult.WeatherResult processWithHeuristic(@NonNull Bitmap skyRegion) {
        int width = skyRegion.getWidth();
        int height = skyRegion.getHeight();

        // Sample pixels (every 4th pixel for performance)
        long totalBrightness = 0;
        long totalBlue = 0;
        long totalRed = 0;
        long totalGreen = 0;
        long totalSaturation = 0;
        int sampleCount = 0;

        int step = 4;
        for (int y = 0; y < height; y += step) {
            for (int x = 0; x < width; x += step) {
                int pixel = skyRegion.getPixel(x, y);
                int r = Color.red(pixel);
                int g = Color.green(pixel);
                int b = Color.blue(pixel);

                float[] hsv = new float[3];
                Color.RGBToHSV(r, g, b, hsv);

                totalBrightness += (int) (hsv[2] * 255);  // Value
                totalSaturation += (int) (hsv[1] * 255);  // Saturation
                totalRed += r;
                totalGreen += g;
                totalBlue += b;
                sampleCount++;
            }
        }

        if (sampleCount == 0) {
            return new InferenceResult.WeatherResult("CLEAR", 0.3f,
                    new String[]{"CLEAR"});
        }

        // Compute averages
        float avgBrightness = (float) totalBrightness / sampleCount;
        float avgSaturation = (float) totalSaturation / sampleCount;
        float avgBlue = (float) totalBlue / sampleCount;
        float avgRed = (float) totalRed / sampleCount;
        float avgGreen = (float) totalGreen / sampleCount;

        // Heuristic classification rules
        String condition;
        float confidence;
        String[] features;

        float blueRatio = avgBlue / Math.max(1f, (avgRed + avgGreen + avgBlue) / 3f);
        float warmRatio = avgRed / Math.max(1f, avgBlue);  // Red > Blue indicates sunset/sunrise

        if (avgBrightness > 180 && avgSaturation > 100 && blueRatio > 1.3f) {
            // Bright, saturated, blue-dominant → Clear sky
            condition = "CLEAR";
            confidence = 0.7f;
            features = new String[]{"CLEAR", "PARTLY_CLOUDY"};
        } else if (avgBrightness > 150 && warmRatio > 1.4f && avgSaturation > 80) {
            // Warm-toned, moderate brightness → Sunset/Sunrise
            condition = "SUNSET_SUNRISE";
            confidence = 0.65f;
            features = new String[]{"SUNSET_SUNRISE", "CLEAR"};
        } else if (avgBrightness < 80 && avgSaturation < 30) {
            // Dark, desaturated → Overcast or Stormy
            if (avgBrightness < 50) {
                condition = "STORMY";
                confidence = 0.5f;
                features = new String[]{"STORMY", "OVERCAST"};
            } else {
                condition = "OVERCAST";
                confidence = 0.6f;
                features = new String[]{"OVERCAST", "CLOUDY"};
            }
        } else if (avgBrightness < 130 && avgSaturation < 50) {
            // Moderate brightness, low saturation → Cloudy or Foggy
            if (avgBrightness < 100) {
                condition = "FOGGY";
                confidence = 0.5f;
                features = new String[]{"FOGGY", "CLOUDY"};
            } else {
                condition = "CLOUDY";
                confidence = 0.55f;
                features = new String[]{"CLOUDY", "OVERCAST"};
            }
        } else if (avgBrightness > 140 && avgSaturation > 50 && blueRatio > 1.0f) {
            // Moderate blue → Partly cloudy
            condition = "PARTLY_CLOUDY";
            confidence = 0.6f;
            features = new String[]{"PARTLY_CLOUDY", "CLEAR"};
        } else if (avgSaturation < 40 && avgBrightness < 120) {
            // Low saturation, moderate darkness → Rainy
            condition = "RAINY";
            confidence = 0.45f;
            features = new String[]{"RAINY", "OVERCAST"};
        } else {
            // Default fallback
            condition = "PARTLY_CLOUDY";
            confidence = 0.3f;
            features = new String[]{"PARTLY_CLOUDY", "CLOUDY"};
        }

        return new InferenceResult.WeatherResult(condition, confidence, features);
    }

    // ================================================================
    // Utility
    // ================================================================

    /**
     * Returns the last computed weather result (may be null initially).
     */
    @Nullable
    public InferenceResult.WeatherResult getLastResult() {
        return lastResult;
    }

    /**
     * Returns whether the custom TFLite model is loaded.
     */
    public boolean isModelLoaded() {
        return modelLoaded;
    }

    /**
     * Returns all weather class labels.
     */
    @NonNull
    public static String[] getWeatherClasses() {
        return WEATHER_CLASSES.clone();
    }
}
