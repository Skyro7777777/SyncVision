/**
 * SceneUnderstanding.java
 *
 * Central scene analysis engine for the Sync Vision app. Takes all ML
 * pipeline results and fuses them into a coherent scene understanding.
 * Maintains scene state across frames using temporal smoothing and
 * object tracking.
 *
 * Key methods:
 *   - processScene(SceneResult) → SceneResult (fused result)
 *   - getSceneSummary() → String (for HUD display)
 *   - getDetectedObjects() → List<DetectedObject>
 *   - getWeatherStatus() → String
 *   - getThreatLevel() → int (0-3)
 *   - getObjectAtPosition(nx, ny) → DetectedObject
 *
 * Maintains scene state across frames:
 *   - Temporal smoothing for detection confidence
 *   - Object tracking via IoU matching
 *   - Persistent weather status
 *   - Accumulated threat assessment
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
import com.syncvision.app.nativelib.NativeConstants;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Central scene analysis engine that fuses all ML pipeline results
 * into a coherent understanding of the current scene.
 * <p>
 * This class maintains state across frames to provide:
 * <ul>
 *   <li>Temporal smoothing — reduces flickering in detection results</li>
 *   <li>Object tracking — matches detections across frames using IoU</li>
 *   <li>Persistent weather status — carries forward when not updated</li>
 *   <li>Accumulated threat assessment — considers hazard levels of all objects</li>
 * </ul>
 */
public class SceneUnderstanding {

    private static final String TAG = "SV-SceneUnderstanding";

    // ================================================================
    // Temporal Smoothing Constants
    // ================================================================

    /** Smoothing factor for exponential moving average (0-1). */
    private static final float SMOOTHING_FACTOR = 0.3f;

    /** Minimum IoU for matching objects across frames. */
    private static final float TRACKING_IOU_THRESHOLD = 0.3f;

    /** Maximum number of frames an object can be missing before removal. */
    private static final int MAX_MISSING_FRAMES = 5;

    // ================================================================
    // Sub-Engines
    // ================================================================

    /** Object fusion engine (segmentation + detection). */
    private final ObjectFusion objectFusion;

    /** Relationship extractor. */
    private final RelationshipExtractor relationshipExtractor;

    /** Path analyzer. */
    private final PathAnalyzer pathAnalyzer;

    /** Hazard scorer. */
    private final HazardScorer hazardScorer;

    /** Weather analyzer. */
    private final WeatherAnalyzer weatherAnalyzer;

    // ================================================================
    // State
    // ================================================================

    /** Currently tracked objects with temporal smoothing. */
    private final List<TrackedObject> trackedObjects = new ArrayList<>();

    /** Last known weather condition. */
    @NonNull
    private String lastWeatherCondition = "INITIALIZING";

    /** Last known weather confidence. */
    private float lastWeatherConfidence = 0f;

    /** Current overall threat level (0-3). */
    private int currentThreatLevel = 0;

    /** Last processed scene result. */
    @Nullable
    private InferenceResult.SceneResult lastResult;

    /** Frame counter for temporal tracking. */
    private long frameCount = 0;

    // ================================================================
    // Inner Class — TrackedObject
    // ================================================================

    /**
     * Represents an object being tracked across frames.
     * Includes temporal smoothing data for confidence and position.
     */
    private static class TrackedObject {
        /** The detected object data. */
        InferenceResult.DetectedObject object;

        /** Smoothed confidence value. */
        float smoothedConfidence;

        /** Number of consecutive frames this object was not detected. */
        int missingFrames;

        /** Unique track ID. */
        int trackId;

        /** Frame when this track was last updated. */
        long lastUpdateFrame;

        TrackedObject(@NonNull InferenceResult.DetectedObject obj, int trackId, long frame) {
            this.object = obj;
            this.smoothedConfidence = obj.confidence;
            this.missingFrames = 0;
            this.trackId = trackId;
            this.lastUpdateFrame = frame;
        }
    }

    // ================================================================
    // Next track ID counter
    // ================================================================

    private int nextTrackId = 1;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new SceneUnderstanding engine.
     * Initializes all sub-engines.
     */
    public SceneUnderstanding() {
        objectFusion = new ObjectFusion();
        relationshipExtractor = new RelationshipExtractor();
        pathAnalyzer = new PathAnalyzer();
        hazardScorer = new HazardScorer();
        weatherAnalyzer = new WeatherAnalyzer();

        Log.i(TAG, "SceneUnderstanding engine initialized");
    }

    // ================================================================
    // Main Processing
    // ================================================================

    /**
     * Processes a new scene result from the ML pipelines.
     * Fuses detection and segmentation data, updates object tracking,
     * computes hazard levels, and produces a fused SceneResult.
     *
     * @param partial The raw scene result from the frame dispatcher.
     * @return A fused scene result with enhanced data.
     */
    @NonNull
    public InferenceResult.SceneResult processScene(
            @NonNull InferenceResult.SceneResult partial) {
        frameCount++;

        try {
            // Step 1: Fuse segmentation + detection
            InferenceResult.SceneResult fused = objectFusion.fuse(partial);

            // Step 2: Update object tracking with temporal smoothing
            updateTracking(fused);

            // Step 3: Compute hazard levels for each object
            if (fused.detection != null) {
                for (InferenceResult.DetectedObject obj : fused.detection.objects) {
                    obj.hazardLevel = hazardScorer.scoreObject(obj);
                }
            }

            // Step 4: Update overall threat level
            currentThreatLevel = hazardScorer.scoreScene(
                    getDetectedObjectsFromTracking());

            // Step 5: Update weather status
            if (fused.weather != null) {
                weatherAnalyzer.update(fused.weather);
                lastWeatherCondition = weatherAnalyzer.getCurrentCondition();
                lastWeatherConfidence = weatherAnalyzer.getCurrentConfidence();
            }

            // Step 6: Run path analysis if depth data is available
            if (fused.depth != null && fused.detection != null) {
                InferenceResult.PathResult pathResult =
                        pathAnalyzer.analyzePath(fused.depth, fused.detection.objects);
                if (pathResult != null) {
                    fused.path = pathResult;
                }
            }

            lastResult = fused;
            return fused;

        } catch (Exception e) {
            Log.e(TAG, "Error processing scene", e);
            return partial; // Return raw result on error
        }
    }

    // ================================================================
    // Object Tracking
    // ================================================================

    /**
     * Updates the object tracking state with new detections.
     * Matches new detections to existing tracks using IoU,
     * applies temporal smoothing, and ages out missing objects.
     *
     * @param result The latest fused scene result.
     */
    private void updateTracking(@NonNull InferenceResult.SceneResult result) {
        if (result.detection == null || result.detection.objects == null) {
            // No detections — age all tracked objects
            for (TrackedObject tracked : trackedObjects) {
                tracked.missingFrames++;
            }
            removeStaleTracks();
            return;
        }

        List<InferenceResult.DetectedObject> newDetections =
                result.detection.objects;
        boolean[] matched = new boolean[newDetections.size()];

        // Match new detections to existing tracks
        for (TrackedObject tracked : trackedObjects) {
            float bestIoU = 0f;
            int bestIdx = -1;

            for (int i = 0; i < newDetections.size(); i++) {
                if (matched[i]) continue;

                // Only match objects of the same class
                if (!newDetections.get(i).name.equals(tracked.object.name)) continue;

                float iou = computeIoU(tracked.object.bbox,
                        newDetections.get(i).bbox);
                if (iou > bestIoU && iou >= TRACKING_IOU_THRESHOLD) {
                    bestIoU = iou;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0) {
                // Match found — update the tracked object with temporal smoothing
                InferenceResult.DetectedObject newObj = newDetections.get(bestIdx);
                tracked.object = newObj;

                // Smooth the confidence
                tracked.smoothedConfidence = SMOOTHING_FACTOR * newObj.confidence
                        + (1 - SMOOTHING_FACTOR) * tracked.smoothedConfidence;
                tracked.object.confidence = tracked.smoothedConfidence;

                // Inherit track ID
                tracked.object.id = tracked.trackId;

                tracked.missingFrames = 0;
                tracked.lastUpdateFrame = frameCount;
                matched[bestIdx] = true;
            } else {
                // No match — this object may have disappeared
                tracked.missingFrames++;
            }
        }

        // Create new tracks for unmatched detections
        for (int i = 0; i < newDetections.size(); i++) {
            if (!matched[i]) {
                TrackedObject tracked = new TrackedObject(
                        newDetections.get(i), nextTrackId++, frameCount);
                tracked.object.id = tracked.trackId;
                trackedObjects.add(tracked);
            }
        }

        // Remove stale tracks
        removeStaleTracks();
    }

    /**
     * Removes tracked objects that have been missing for too many frames.
     */
    private void removeStaleTracks() {
        trackedObjects.removeIf(t -> t.missingFrames > MAX_MISSING_FRAMES);
    }

    /**
     * Computes Intersection over Union (IoU) between two bounding boxes.
     */
    private float computeIoU(@NonNull InferenceResult.RectF a,
                             @NonNull InferenceResult.RectF b) {
        float interLeft = Math.max(a.left, b.left);
        float interTop = Math.max(a.top, b.top);
        float interRight = Math.min(a.right, b.right);
        float interBottom = Math.min(a.bottom, b.bottom);

        float interWidth = Math.max(0f, interRight - interLeft);
        float interHeight = Math.max(0f, interBottom - interTop);
        float interArea = interWidth * interHeight;

        float areaA = a.area();
        float areaB = b.area();
        float unionArea = areaA + areaB - interArea;

        if (unionArea <= 0f) return 0f;
        return interArea / unionArea;
    }

    // ================================================================
    // Public Accessors
    // ================================================================

    /**
     * Returns a human-readable summary of the current scene for HUD display.
     * Format: "5 OBJECTS | WEATHER: CLEAR | THREAT: LOW"
     *
     * @return Scene summary string.
     */
    @NonNull
    public String getSceneSummary() {
        int objCount = trackedObjects.size();
        String threatStr = NativeConstants.hazardLevelToString(currentThreatLevel);
        return String.format(Locale.US, "%d OBJECTS | %s | THREAT: %s",
                objCount, lastWeatherCondition, threatStr);
    }

    /**
     * Returns the list of currently detected (tracked) objects.
     *
     * @return Unmodifiable list of detected objects.
     */
    @NonNull
    public List<InferenceResult.DetectedObject> getDetectedObjects() {
        return getDetectedObjectsFromTracking();
    }

    /**
     * Returns the current weather status string.
     * Format: "WEATHER: PARTLY CLOUDY 72%"
     *
     * @return Weather status string.
     */
    @NonNull
    public String getWeatherStatus() {
        return String.format(Locale.US, "WEATHER: %s %.0f%%",
                lastWeatherCondition.toUpperCase(Locale.US),
                lastWeatherConfidence * 100f);
    }

    /**
     * Returns the current overall threat level (0-3).
     *
     * @return Threat level: 0=NONE, 1=LOW, 2=MEDIUM, 3=HIGH.
     */
    public int getThreatLevel() {
        return currentThreatLevel;
    }

    /**
     * Finds the detected object closest to the given normalized coordinates.
     * Used for tap-to-inspect functionality.
     *
     * @param nx Normalized X [0, 1].
     * @param ny Normalized Y [0, 1].
     * @return The closest detected object, or null if none found.
     */
    @Nullable
    public InferenceResult.DetectedObject getObjectAtPosition(float nx, float ny) {
        InferenceResult.DetectedObject closest = null;
        float closestDist = Float.MAX_VALUE;

        for (TrackedObject tracked : trackedObjects) {
            InferenceResult.DetectedObject obj = tracked.object;
            float dx = obj.bbox.centerX() - nx;
            float dy = obj.bbox.centerY() - ny;
            float dist = (float) Math.sqrt(dx * dx + dy * dy);

            // Check if the point is within or very near the bounding box
            if (nx >= obj.bbox.left - 0.05f && nx <= obj.bbox.right + 0.05f
                    && ny >= obj.bbox.top - 0.05f && ny <= obj.bbox.bottom + 0.05f) {
                if (dist < closestDist) {
                    closestDist = dist;
                    closest = obj;
                }
            }
        }

        return closest;
    }

    /**
     * Returns the list of tracked objects as DetectedObject list.
     */
    @NonNull
    private List<InferenceResult.DetectedObject> getDetectedObjectsFromTracking() {
        List<InferenceResult.DetectedObject> objects = new ArrayList<>();
        for (TrackedObject tracked : trackedObjects) {
            objects.add(tracked.object);
        }
        return objects;
    }

    // ================================================================
    // Sub-Engine Accessors
    // ================================================================

    /** Returns the object fusion engine. */
    @NonNull
    public ObjectFusion getObjectFusion() {
        return objectFusion;
    }

    /** Returns the relationship extractor. */
    @NonNull
    public RelationshipExtractor getRelationshipExtractor() {
        return relationshipExtractor;
    }

    /** Returns the path analyzer. */
    @NonNull
    public PathAnalyzer getPathAnalyzer() {
        return pathAnalyzer;
    }

    /** Returns the hazard scorer. */
    @NonNull
    public HazardScorer getHazardScorer() {
        return hazardScorer;
    }

    /** Returns the weather analyzer. */
    @NonNull
    public WeatherAnalyzer getWeatherAnalyzer() {
        return weatherAnalyzer;
    }

    // ================================================================
    // Lifecycle
    // ================================================================

    /**
     * Releases all resources held by the scene understanding engine.
     */
    public void release() {
        trackedObjects.clear();
        lastResult = null;
        Log.i(TAG, "SceneUnderstanding released");
    }
}
