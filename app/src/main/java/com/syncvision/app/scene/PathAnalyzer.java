/**
 * PathAnalyzer.java
 *
 * Uses depth map data and detected objects to analyze navigational
 * paths through the scene. Calls NativeProcessor.nativeFindPath()
 * for the core A* pathfinding and then post-processes the results
 * to determine walkable ground plane, obstacle positions, and
 * path clarity.
 *
 * Key responsibilities:
 *   - Determine walkable ground plane from depth data
 *   - Identify obstacles from detected objects
 *   - Call native A* pathfinding
 *   - Post-process path results with clarity scoring
 *   - Return PathResult with waypoints and safety assessment
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
import com.syncvision.app.nativelib.NativeProcessor;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Analyzes navigational paths through the scene using depth map
 * data and detected object positions. Delegates core pathfinding
 * to the C++ native layer via NativeProcessor.nativeFindPath().
 * <p>
 * The path analysis process:
 * <ol>
 *   <li>Extract walkable ground plane from depth map</li>
 *   <li>Convert detected objects to obstacle markers</li>
 *   <li>Call native A* pathfinding</li>
 *   <li>Post-process results with clarity scoring</li>
 * </ol>
 */
public class PathAnalyzer {

    private static final String TAG = "SV-PathAnalyzer";

    // ================================================================
    // Configuration Constants
    // ================================================================

    /** Depth threshold for ground plane (normalized depth values). */
    private static final float GROUND_DEPTH_MAX = 0.4f;

    /** Depth variance threshold for ground plane consistency. */
    private static final float GROUND_VARIANCE_MAX = 0.05f;

    /** Minimum ground plane area ratio to consider pathfinding. */
    private static final float MIN_GROUND_RATIO = 0.15f;

    /** Obstacle inflation radius (in normalized coordinates). */
    private static final float OBSTACLE_RADIUS = 0.03f;

    /** Path clarity threshold — below this, path is considered BLOCKED. */
    private static final float CLARITY_BLOCKED_THRESHOLD = 0.3f;

    /** Path clarity threshold — below this, path is PARTIAL. */
    private static final float CLARITY_PARTIAL_THRESHOLD = 0.6f;

    // ================================================================
    // State
    // ================================================================

    /** Native processor for C++ pathfinding. */
    @Nullable
    private NativeProcessor nativeProcessor;

    /** Last computed path result. */
    @Nullable
    private InferenceResult.PathResult lastPathResult;

    /** Whether ground plane was detected in last analysis. */
    private boolean groundPlaneDetected = false;

    /** Ground plane area ratio in last analysis. */
    private float groundPlaneRatio = 0f;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new PathAnalyzer.
     */
    public PathAnalyzer() {
        // Try to create a NativeProcessor for C++ pathfinding
        try {
            if (NativeProcessor.isLibraryLoaded()) {
                nativeProcessor = new NativeProcessor();
            }
        } catch (Exception e) {
            Log.w(TAG, "NativeProcessor not available, using Java fallback", e);
        }

        Log.d(TAG, "PathAnalyzer initialized (native: "
                + (nativeProcessor != null ? "YES" : "NO") + ")");
    }

    // ================================================================
    // Main Analysis Method
    // ================================================================

    /**
     * Analyzes the navigational path through the scene using depth
     * map data and detected object positions.
     *
     * @param depthResult The depth estimation result.
     * @param objects     List of detected objects (potential obstacles).
     * @return PathResult with waypoints and clarity, or null on error.
     */
    @Nullable
    public InferenceResult.PathResult analyzePath(
            @NonNull InferenceResult.DepthResult depthResult,
            @NonNull List<InferenceResult.DetectedObject> objects) {
        try {
            // Step 1: Extract ground plane
            float groundRatio = extractGroundPlane(depthResult);
            groundPlaneDetected = groundRatio >= MIN_GROUND_RATIO;
            groundPlaneRatio = groundRatio;

            if (!groundPlaneDetected) {
                Log.d(TAG, "No ground plane detected (ratio: "
                        + String.format(Locale.US, "%.2f", groundRatio) + ")");
                // Return a minimal path result indicating no ground
                lastPathResult = new InferenceResult.PathResult(
                        new ArrayList<>(), Float.MAX_VALUE, false);
                return lastPathResult;
            }

            // Step 2: Build obstacle array from detected objects
            float[] obstacles = buildObstacleArray(objects);

            // Step 3: Call native pathfinding or Java fallback
            InferenceResult.PathResult result;

            if (nativeProcessor != null && NativeProcessor.isLibraryLoaded()) {
                result = findPathNative(depthResult, obstacles);
            } else {
                result = findPathFallback(depthResult, objects);
            }

            // Step 4: Score path clarity
            if (result != null) {
                float clarity = scorePathClarity(result, objects);
                boolean isClear = clarity >= CLARITY_PARTIAL_THRESHOLD;

                // Rebuild result with clarity assessment
                lastPathResult = new InferenceResult.PathResult(
                        result.waypoints,
                        result.cost,
                        isClear);
            } else {
                lastPathResult = null;
            }

            return lastPathResult;

        } catch (Exception e) {
            Log.e(TAG, "Error analyzing path", e);
            return null;
        }
    }

    // ================================================================
    // Ground Plane Extraction
    // ================================================================

    /**
     * Extracts the walkable ground plane from the depth map.
     * The ground plane is identified as the region of consistent
     * low-depth values (close to camera, representing the floor).
     *
     * @param depthResult The depth map.
     * @return Ratio of the image that is walkable ground [0, 1].
     */
    private float extractGroundPlane(
            @NonNull InferenceResult.DepthResult depthResult) {
        int width = depthResult.width;
        int height = depthResult.height;

        if (width <= 0 || height <= 0) return 0f;

        // Sample the bottom portion of the depth map for ground plane
        int startY = (int) (height * 0.6f); // Bottom 40% of image
        int groundPixels = 0;
        int totalPixels = 0;

        // Compute average depth in the bottom region
        float sumDepth = 0f;
        int sampleCount = 0;
        for (int y = startY; y < height; y += 4) {
            for (int x = 0; x < width; x += 4) {
                float depth = depthResult.depthMap[y][x];
                sumDepth += depth;
                sampleCount++;
            }
        }

        if (sampleCount == 0) return 0f;
        float avgDepth = sumDepth / sampleCount;

        // Count pixels within ground plane threshold
        for (int y = startY; y < height; y += 2) {
            for (int x = 0; x < width; x += 2) {
                float depth = depthResult.depthMap[y][x];
                totalPixels++;
                if (depth <= avgDepth + GROUND_VARIANCE_MAX
                        && depth >= avgDepth - GROUND_VARIANCE_MAX) {
                    groundPixels++;
                }
            }
        }

        return totalPixels > 0 ? (float) groundPixels / totalPixels : 0f;
    }

    // ================================================================
    // Obstacle Array Building
    // ================================================================

    /**
     * Builds a flat obstacle array from detected objects.
     * Format: [x0, y0, r0, x1, y1, r1, ...]
     * where each obstacle is (center_x, center_y, radius) in
     * normalized coordinates.
     *
     * @param objects List of detected objects.
     * @return Flat float array of obstacle positions.
     */
    @NonNull
    private float[] buildObstacleArray(
            @NonNull List<InferenceResult.DetectedObject> objects) {
        List<Float> obstacleList = new ArrayList<>();

        for (InferenceResult.DetectedObject obj : objects) {
            // Skip objects that are not obstacles (e.g., background)
            if (isObstacle(obj)) {
                float cx = obj.bbox.centerX();
                float cy = obj.bbox.centerY();
                float rx = obj.bbox.width() / 2f + OBSTACLE_RADIUS;
                float ry = obj.bbox.height() / 2f + OBSTACLE_RADIUS;
                float radius = Math.max(rx, ry);

                obstacleList.add(cx);
                obstacleList.add(cy);
                obstacleList.add(radius);
            }
        }

        // Convert to primitive array
        float[] obstacles = new float[obstacleList.size()];
        for (int i = 0; i < obstacleList.size(); i++) {
            obstacles[i] = obstacleList.get(i);
        }

        return obstacles;
    }

    /**
     * Determines if a detected object should be treated as an
     * obstacle for pathfinding purposes.
     *
     * @param obj The detected object.
     * @return true if the object is an obstacle.
     */
    private boolean isObstacle(@NonNull InferenceResult.DetectedObject obj) {
        String name = obj.name.toLowerCase(Locale.US);

        // People are not obstacles (they move)
        if (name.contains("person")) return false;

        // Small objects are not significant obstacles
        if (obj.bbox.area() < 0.005f) return false;

        // Vehicles, furniture, and large objects are obstacles
        return name.contains("car") || name.contains("truck") || name.contains("bus")
                || name.contains("chair") || name.contains("couch") || name.contains("table")
                || name.contains("bed") || name.contains("bench")
                || name.contains("bicycle") || name.contains("motorcycle")
                || name.contains("bottle") || name.contains("box");
    }

    // ================================================================
    // Native Pathfinding
    // ================================================================

    /**
     * Calls the native A* pathfinding via NativeProcessor.
     *
     * @param depthResult The depth map.
     * @param obstacles   Flat obstacle array.
     * @return PathResult from native pathfinding.
     */
    @Nullable
    private InferenceResult.PathResult findPathNative(
            @NonNull InferenceResult.DepthResult depthResult,
            @NonNull float[] obstacles) {
        try {
            float[] result = nativeProcessor.findPath2d(
                    depthResult.depthMap,
                    depthResult.width,
                    depthResult.height,
                    obstacles);

            if (result == null || result.length < NativeConstants.PATH_HEADER_SIZE) {
                Log.w(TAG, "Native pathfinding returned insufficient data");
                return null;
            }

            // Parse the native result
            return parseNativePathResult(result);

        } catch (Exception e) {
            Log.e(TAG, "Native pathfinding error", e);
            return null;
        }
    }

    /**
     * Parses the flat float array from native pathfinding into a PathResult.
     *
     * @param data Native result data: [totalCost, isClear, numWaypoints, x0, y0, cost0, ...]
     * @return Parsed PathResult.
     */
    @NonNull
    private InferenceResult.PathResult parseNativePathResult(@NonNull float[] data) {
        float totalCost = data[NativeConstants.PATH_TOTAL_COST_INDEX];
        float isClearVal = data[NativeConstants.PATH_IS_CLEAR_INDEX];
        int numWaypoints = (int) data[NativeConstants.PATH_NUM_WAYPOINTS_INDEX];

        List<float[]> waypoints = new ArrayList<>();
        int offset = NativeConstants.PATH_WAYPOINTS_START_INDEX;

        for (int i = 0; i < numWaypoints
                && offset + NativeConstants.PATH_WAYPOINT_SIZE <= data.length; i++) {
            float x = data[offset + NativeConstants.PATH_WAYPOINT_X_INDEX];
            float y = data[offset + NativeConstants.PATH_WAYPOINT_Y_INDEX];
            waypoints.add(new float[]{x, y});
            offset += NativeConstants.PATH_WAYPOINT_SIZE;
        }

        boolean isClear = isClearVal > 0.5f;

        return new InferenceResult.PathResult(waypoints, totalCost, isClear);
    }

    // ================================================================
    // Java Fallback Pathfinding
    // ================================================================

    /**
     * Simple Java fallback for pathfinding when the native library
     * is not available. Creates a straight-line path from the bottom
     * center of the image toward the top center, checking for obstacles.
     *
     * @param depthResult The depth map.
     * @param objects     Detected objects.
     * @return A simplified PathResult.
     */
    @Nullable
    private InferenceResult.PathResult findPathFallback(
            @NonNull InferenceResult.DepthResult depthResult,
            @NonNull List<InferenceResult.DetectedObject> objects) {
        // Create a simple straight-line path
        List<float[]> waypoints = new ArrayList<>();

        // Start: bottom center of image
        waypoints.add(new float[]{0.5f, 0.95f});

        // Midpoint: center of image
        waypoints.add(new float[]{0.5f, 0.5f});

        // End: top center of image
        waypoints.add(new float[]{0.5f, 0.1f});

        // Check for obstacles along the path
        boolean pathBlocked = false;
        float maxCost = 1.0f;

        for (InferenceResult.DetectedObject obj : objects) {
            if (isObstacle(obj)) {
                float cx = obj.bbox.centerX();
                float cy = obj.bbox.centerY();

                // Check if obstacle is near the path line (x ≈ 0.5)
                if (Math.abs(cx - 0.5f) < 0.15f) {
                    pathBlocked = true;
                    maxCost += 50f;
                }
            }
        }

        return new InferenceResult.PathResult(
                waypoints, maxCost, !pathBlocked);
    }

    // ================================================================
    // Path Clarity Scoring
    // ================================================================

    /**
     * Scores the clarity of a path based on obstacles and path cost.
     *
     * @param pathResult The path to score.
     * @param objects    Detected objects in the scene.
     * @return Clarity score [0, 1] where 1 = completely clear.
     */
    private float scorePathClarity(@NonNull InferenceResult.PathResult pathResult,
                                   @NonNull List<InferenceResult.DetectedObject> objects) {
        float clarity = 1.0f;

        // Reduce clarity based on path cost
        if (pathResult.cost > 100f) {
            clarity -= 0.5f;
        } else if (pathResult.cost > 50f) {
            clarity -= 0.3f;
        } else if (pathResult.cost > 20f) {
            clarity -= 0.1f;
        }

        // Reduce clarity for nearby obstacles
        for (InferenceResult.DetectedObject obj : objects) {
            if (isObstacle(obj) && obj.distance > 0 && obj.distance < 3f) {
                clarity -= 0.1f;
            }
        }

        return Math.max(0f, Math.min(1f, clarity));
    }

    // ================================================================
    // Accessors
    // ================================================================

    /** Returns whether a ground plane was detected in the last analysis. */
    public boolean isGroundPlaneDetected() {
        return groundPlaneDetected;
    }

    /** Returns the ground plane area ratio from the last analysis. */
    public float getGroundPlaneRatio() {
        return groundPlaneRatio;
    }

    /** Returns the last computed path result. */
    @Nullable
    public InferenceResult.PathResult getLastPathResult() {
        return lastPathResult;
    }
}
