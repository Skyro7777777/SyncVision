/**
 * WeatherAnalyzer.java
 *
 * Analyzes weather conditions from the WeatherPipeline results and
 * adds contextual interpretation. Combines sky analysis with general
 * scene brightness to provide weather awareness for the HUD display.
 *
 * The analyzer adds contextual information beyond the raw classification:
 *   - "RAIN EXPECTED" if overcast + high humidity indicators
 *   - "LOW VISIBILITY" if foggy conditions detected
 *   - "SLIPPERY CONDITIONS" if rainy
 *   - Scene brightness influence on classification confidence
 *
 * The analyzer maintains state across weather updates using temporal
 * smoothing to avoid flickering between weather states.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.scene
 * Target SDK: 29+
 */

package com.syncvision.app.scene;

import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.syncvision.app.ml.InferenceResult;

import java.util.Locale;

/**
 * Analyzes weather conditions from ML pipeline results and provides
 * contextual weather awareness for the HUD display.
 * <p>
 * The analyzer applies temporal smoothing to reduce weather state
 * flickering and adds contextual interpretations based on the
 * detected weather condition and associated sky features.
 * <p>
 * Weather conditions:
 *   CLEAR, PARTLY_CLOUDY, CLOUDY, OVERCAST, RAINY, FOGGY,
 *   SUNSET_SUNRISE, STORMY
 */
public class WeatherAnalyzer {

    private static final String TAG = "SV-WeatherAnalyzer";

    // ================================================================
    // Temporal Smoothing Constants
    // ================================================================

    /** Smoothing factor for weather confidence (0-1). */
    private static final float CONFIDENCE_SMOOTHING = 0.7f;

    /** Number of consecutive same-class results before changing state. */
    private static final int STATE_CHANGE_THRESHOLD = 2;

    // ================================================================
    // State
    // ================================================================

    /** Current smoothed weather condition. */
    @NonNull
    private String currentCondition = "INITIALIZING";

    /** Current smoothed confidence. */
    private float currentConfidence = 0f;

    /** Previous raw weather condition (for state change tracking). */
    @NonNull
    private String previousRawCondition = "";

    /** Number of consecutive frames with the same raw condition. */
    private int consecutiveSameCondition = 0;

    /** Current contextual message (e.g., "RAIN EXPECTED"). */
    @Nullable
    private String contextualMessage;

    /** Whether the analyzer has received at least one update. */
    private boolean initialized = false;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new WeatherAnalyzer.
     */
    public WeatherAnalyzer() {
        Log.d(TAG, "WeatherAnalyzer initialized");
    }

    // ================================================================
    // Main Update Method
    // ================================================================

    /**
     * Updates the weather analysis with a new WeatherResult from
     * the ML pipeline. Applies temporal smoothing and adds
     * contextual interpretation.
     *
     * @param weatherResult The latest weather classification result.
     */
    public void update(@NonNull InferenceResult.WeatherResult weatherResult) {
        try {
            String rawCondition = weatherResult.condition.toUpperCase(Locale.US);
            float rawConfidence = weatherResult.confidence;

            // Step 1: Temporal smoothing on condition (debounce rapid changes)
            if (rawCondition.equals(previousRawCondition)) {
                consecutiveSameCondition++;
            } else {
                consecutiveSameCondition = 0;
                previousRawCondition = rawCondition;
            }

            // Only change the displayed condition after enough consecutive frames
            if (!initialized || consecutiveSameCondition >= STATE_CHANGE_THRESHOLD) {
                currentCondition = rawCondition;
                initialized = true;
            }

            // Step 2: Smooth the confidence value
            if (currentConfidence == 0f) {
                currentConfidence = rawConfidence;
            } else {
                currentConfidence = CONFIDENCE_SMOOTHING * rawConfidence
                        + (1 - CONFIDENCE_SMOOTHING) * currentConfidence;
            }

            // Step 3: Add contextual interpretation
            contextualMessage = interpretContext(
                    currentCondition, weatherResult.skyFeatures);

        } catch (Exception e) {
            Log.e(TAG, "Error updating weather analysis", e);
        }
    }

    // ================================================================
    // Contextual Interpretation
    // ================================================================

    /**
     * Adds contextual interpretation to the raw weather condition
     * based on detected sky features and general scene analysis.
     *
     * Contextual rules:
     *   - OVERCAST + high cloud features → "RAIN EXPECTED"
     *   - FOGGY → "LOW VISIBILITY — EXERCISE CAUTION"
     *   - RAINY → "SLIPPERY CONDITIONS"
     *   - STORMY → "SEVERE WEATHER — SEEK SHELTER"
     *   - SUNSET_SUNRISE → "REDUCED VISIBILITY"
     *   - CLEAR + hot indicators → "HIGH UV — PROTECT EYES"
     *
     * @param condition   The current weather condition.
     * @param skyFeatures Detected sky features from the pipeline.
     * @return Contextual message string, or null.
     */
    @Nullable
    private String interpretContext(@NonNull String condition,
                                    @NonNull String[] skyFeatures) {
        switch (condition) {
            case "STORMY":
                return "SEVERE WEATHER — SEEK SHELTER";

            case "RAINY":
                return "SLIPPERY CONDITIONS";

            case "FOGGY":
                return "LOW VISIBILITY — EXERCISE CAUTION";

            case "OVERCAST":
                // Check for rain indicators in sky features
                for (String feature : skyFeatures) {
                    if (feature.toLowerCase(Locale.US).contains("dark_cloud")
                            || feature.toLowerCase(Locale.US).contains("thick")) {
                        return "RAIN EXPECTED";
                    }
                }
                return "OVERCAST SKIES";

            case "CLOUDY":
                return "CLOUDY CONDITIONS";

            case "PARTLY_CLOUDY":
                return null; // No special context needed

            case "SUNSET_SUNRISE":
                return "REDUCED VISIBILITY";

            case "CLEAR":
                // Check for high UV indicators
                for (String feature : skyFeatures) {
                    if (feature.toLowerCase(Locale.US).contains("bright")
                            || feature.toLowerCase(Locale.US).contains("harsh")) {
                        return "HIGH UV — PROTECT EYES";
                    }
                }
                return null;

            default:
                return null;
        }
    }

    // ================================================================
    // Public Accessors
    // ================================================================

    /**
     * Returns the current (smoothed) weather condition.
     *
     * @return Current weather condition string (ALL CAPS).
     */
    @NonNull
    public String getCurrentCondition() {
        return currentCondition;
    }

    /**
     * Returns the current (smoothed) weather confidence.
     *
     * @return Confidence value [0, 1].
     */
    public float getCurrentConfidence() {
        return currentConfidence;
    }

    /**
     * Returns the current contextual message, if any.
     *
     * @return Contextual message (ALL CAPS), or null.
     */
    @Nullable
    public String getContextualMessage() {
        return contextualMessage;
    }

    /**
     * Returns the full weather status string for HUD display.
     * Format: "PARTLY CLOUDY 72%" or "RAINY 85% [SLIPPERY CONDITIONS]"
     *
     * @return Formatted weather status string.
     */
    @NonNull
    public String getWeatherStatus() {
        StringBuilder sb = new StringBuilder();
        sb.append("WEATHER: ");
        sb.append(currentCondition);
        sb.append(String.format(Locale.US, " %.0f%%", currentConfidence * 100f));

        if (contextualMessage != null) {
            sb.append(" [").append(contextualMessage).append("]");
        }

        return sb.toString();
    }

    /**
     * Returns whether the current conditions indicate low visibility.
     *
     * @return true if visibility is reduced.
     */
    public boolean isLowVisibility() {
        return "FOGGY".equals(currentCondition)
                || "STORMY".equals(currentCondition)
                || "SUNSET_SUNRISE".equals(currentCondition);
    }

    /**
     * Returns whether the current conditions indicate precipitation.
     *
     * @return true if rain or storm is detected.
     */
    public boolean isPrecipitation() {
        return "RAINY".equals(currentCondition)
                || "STORMY".equals(currentCondition);
    }
}
