/**
 * HazardScorer.java
 *
 * Rule-based threat assessment engine for the Sync Vision app.
 * Evaluates the danger level of detected objects based on their
 * class type, distance, and contextual rules. Provides per-object
 * hazard scores and an overall scene threat level.
 *
 * Hazard levels:
 *   HIGH (3):   car/truck/bus nearby + moving, fire, weapon-like object
 *   MEDIUM (2): obstacle in path, steep drop (depth), large animal
 *   LOW (1):    uneven surface, low ceiling, wet area
 *   NONE (0):   informational objects
 *
 * Distance factor: closer objects are more hazardous.
 * The scorer applies a distance-based multiplier to the base hazard level.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.scene
 * Target SDK: 29+
 */

package com.syncvision.app.scene;

import android.util.Log;

import androidx.annotation.NonNull;

import com.syncvision.app.ml.InferenceResult;
import com.syncvision.app.nativelib.NativeConstants;

import java.util.List;
import java.util.Locale;

/**
 * Rule-based threat assessment engine. Scores individual objects
 * and the overall scene for hazard levels based on object type,
 * proximity, and contextual rules.
 * <p>
 * Scoring is rule-based (no ML model needed) because hazard
 * assessment is fundamentally about known object types and their
 * spatial relationship to the user.
 */
public class HazardScorer {

    private static final String TAG = "SV-HazardScorer";

    // ================================================================
    // Distance Factor Constants
    // ================================================================

    /** Close range — very hazardous (within arm's reach). */
    private static final float DISTANCE_CLOSE = 2.0f;

    /** Medium range — moderately hazardous. */
    private static final float DISTANCE_MEDIUM = 5.0f;

    /** Far range — low hazard. */
    private static final float DISTANCE_FAR = 10.0f;

    /** Distance multiplier for close objects. */
    private static final float MULT_CLOSE = 1.0f;

    /** Distance multiplier for medium-distance objects. */
    private static final float MULT_MEDIUM = 0.6f;

    /** Distance multiplier for far objects. */
    private static final float MULT_FAR = 0.3f;

    /** Distance multiplier for unknown-distance objects. */
    private static final float MULT_UNKNOWN = 0.5f;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new HazardScorer.
     */
    public HazardScorer() {
        Log.d(TAG, "HazardScorer initialized");
    }

    // ================================================================
    // Per-Object Scoring
    // ================================================================

    /**
     * Scores the hazard level of a single detected object.
     *
     * @param obj The detected object to score.
     * @return Hazard level (0=NONE, 1=LOW, 2=MEDIUM, 3=HIGH).
     */
    public int scoreObject(@NonNull InferenceResult.DetectedObject obj) {
        // Get base hazard level from object class
        int baseLevel = getBaseHazardLevel(obj.name);

        // Apply distance factor
        float distMult = getDistanceMultiplier(obj.distance);

        // Compute adjusted level
        float adjustedLevel = baseLevel * distMult;

        // Round to nearest integer
        int finalLevel = Math.round(adjustedLevel);

        // Clamp to valid range
        return Math.max(NativeConstants.HAZARD_NONE,
                Math.min(NativeConstants.HAZARD_HIGH, finalLevel));
    }

    /**
     * Returns the base hazard level for an object class name.
     * This is the hazard level assuming the object is at close range.
     *
     * @param className The detected object class name.
     * @return Base hazard level (0-3).
     */
    private int getBaseHazardLevel(@NonNull String className) {
        String name = className.toLowerCase(Locale.US);

        // ---- HIGH (3): Immediate danger ----
        // Vehicles nearby (especially if moving)
        if (name.contains("car") || name.contains("truck")
                || name.contains("bus") || name.contains("train")
                || name.contains("motorcycle") || name.contains("airplane")) {
            return NativeConstants.HAZARD_HIGH;
        }

        // Fire-related (not in COCO, but future-proofing)
        if (name.contains("fire") || name.contains("flame")) {
            return NativeConstants.HAZARD_HIGH;
        }

        // Weapon-like objects
        if (name.contains("knife") || name.contains("scissors")
                || name.contains("baseball bat")) {
            return NativeConstants.HAZARD_HIGH;
        }

        // ---- MEDIUM (2): Significant risk ----
        // Large animals
        if (name.contains("horse") || name.contains("cow")
                || name.contains("bear") || name.contains("elephant")) {
            return NativeConstants.HAZARD_MEDIUM;
        }

        // Bicycles (can cause collisions)
        if (name.contains("bicycle")) {
            return NativeConstants.HAZARD_MEDIUM;
        }

        // ---- LOW (1): Minor risk ----
        // Small animals (startling but not dangerous)
        if (name.contains("dog") || name.contains("cat")
                || name.contains("bird") || name.contains("sheep")) {
            return NativeConstants.HAZARD_LOW;
        }

        // Furniture (trip hazards)
        if (name.contains("chair") || name.contains("table")
                || name.contains("bed") || name.contains("couch")
                || name.contains("bench") || name.contains("sofa")) {
            return NativeConstants.HAZARD_LOW;
        }

        // Bottles, cups (slip/trip hazard)
        if (name.contains("bottle") || name.contains("cup")
                || name.contains("wine glass") || name.contains("vase")) {
            return NativeConstants.HAZARD_LOW;
        }

        // Potted plants (trip hazard)
        if (name.contains("potted plant") || name.contains("pottedplant")) {
            return NativeConstants.HAZARD_LOW;
        }

        // ---- NONE (0): Informational ----
        // Electronics, food, accessories, etc.
        if (name.contains("tv") || name.contains("laptop")
                || name.contains("phone") || name.contains("keyboard")
                || name.contains("mouse") || name.contains("remote")
                || name.contains("book") || name.contains("clock")
                || name.contains("umbrella") || name.contains("backpack")
                || name.contains("handbag") || name.contains("tie")
                || name.contains("suitcase") || name.contains("frisbee")) {
            return NativeConstants.HAZARD_NONE;
        }

        // People are generally informational (not hazards)
        if (name.contains("person") || name.contains("people")) {
            return NativeConstants.HAZARD_NONE;
        }

        // Food items
        if (name.contains("banana") || name.contains("apple")
                || name.contains("sandwich") || name.contains("orange")
                || name.contains("pizza") || name.contains("cake")
                || name.contains("donut") || name.contains("hot dog")
                || name.contains("broccoli") || name.contains("carrot")) {
            return NativeConstants.HAZARD_NONE;
        }

        // Default: LOW for unknown objects
        return NativeConstants.HAZARD_LOW;
    }

    /**
     * Returns a distance-based multiplier for hazard scoring.
     * Closer objects are more hazardous.
     *
     * @param distance Distance in meters (-1 if unknown).
     * @return Multiplier [0, 1].
     */
    private float getDistanceMultiplier(float distance) {
        if (distance < 0) {
            // Unknown distance — use moderate multiplier
            return MULT_UNKNOWN;
        }

        if (distance <= DISTANCE_CLOSE) {
            // Very close — full hazard
            return MULT_CLOSE;
        } else if (distance <= DISTANCE_MEDIUM) {
            // Medium range — moderate hazard
            return MULT_MEDIUM;
        } else if (distance <= DISTANCE_FAR) {
            // Far — low hazard
            return MULT_FAR;
        } else {
            // Very far — minimal hazard
            return 0.1f;
        }
    }

    // ================================================================
    // Scene-Level Scoring
    // ================================================================

    /**
     * Scores the overall threat level of the scene by considering
     * the highest individual hazard level and the density of
     * hazardous objects.
     *
     * @param objects All detected objects in the scene.
     * @return Overall scene threat level (0-3).
     */
    public int scoreScene(@NonNull List<InferenceResult.DetectedObject> objects) {
        if (objects.isEmpty()) {
            return NativeConstants.HAZARD_NONE;
        }

        int maxLevel = NativeConstants.HAZARD_NONE;
        int highCount = 0;
        int mediumCount = 0;

        for (InferenceResult.DetectedObject obj : objects) {
            int level = scoreObject(obj);

            if (level > maxLevel) {
                maxLevel = level;
            }

            if (level >= NativeConstants.HAZARD_HIGH) {
                highCount++;
            } else if (level >= NativeConstants.HAZARD_MEDIUM) {
                mediumCount++;
            }
        }

        // Escalate threat level if multiple hazardous objects are present
        if (highCount >= 2) {
            // Multiple high-hazard objects — ensure we report HIGH
            maxLevel = NativeConstants.HAZARD_HIGH;
        } else if (highCount >= 1 && mediumCount >= 1) {
            // Mix of high and medium — report HIGH
            maxLevel = NativeConstants.HAZARD_HIGH;
        } else if (mediumCount >= 3) {
            // Multiple medium hazards — escalate to MEDIUM (if not already)
            maxLevel = Math.max(maxLevel, NativeConstants.HAZARD_MEDIUM);
        }

        return maxLevel;
    }

    // ================================================================
    // Detailed Scoring (for InfoPanelView)
    // ================================================================

    /**
     * Returns a detailed hazard description for an object.
     *
     * @param obj The detected object.
     * @return Detailed hazard description string.
     */
    @NonNull
    public String getHazardDescription(@NonNull InferenceResult.DetectedObject obj) {
        int level = scoreObject(obj);
        String levelStr = NativeConstants.hazardLevelToString(level);

        StringBuilder desc = new StringBuilder();
        desc.append("HAZARD: ").append(levelStr);

        if (obj.distance > 0) {
            desc.append(String.format(Locale.US, " | DISTANCE: %.1fm", obj.distance));
        }

        // Add contextual reason
        String name = obj.name.toLowerCase(Locale.US);
        if (name.contains("car") || name.contains("truck") || name.contains("bus")) {
            desc.append(" | REASON: VEHICLE IN PROXIMITY");
        } else if (name.contains("knife") || name.contains("scissors")) {
            desc.append(" | REASON: SHARP OBJECT DETECTED");
        } else if (name.contains("horse") || name.contains("cow") || name.contains("bear")) {
            desc.append(" | REASON: LARGE ANIMAL IN VICINITY");
        } else if (name.contains("chair") || name.contains("table") || name.contains("bed")) {
            desc.append(" | REASON: POTENTIAL OBSTACLE");
        } else {
            desc.append(" | REASON: OBJECT IN SCENE");
        }

        return desc.toString().toUpperCase(Locale.US);
    }
}
