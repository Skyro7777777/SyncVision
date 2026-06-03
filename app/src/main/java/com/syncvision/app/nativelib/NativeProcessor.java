/**
 * NativeProcessor.java
 *
 * JNI bridge class for the Sync Vision native library. Provides Java
 * access to the C++ processing functions for contour extraction,
 * label placement, pathfinding, diagram generation, and edge detection.
 *
 * IMPORTANT: The package is com.syncvision.app.nativelib (not "native")
 * because "native" is a reserved Java keyword and cannot be used as
 * a package name. The C++ JNI function names in jni_bridge.cpp use
 * "native" in the package path — those function signatures would need
 * to be updated to match this package (nativelib) for correct JNI
 * method resolution at runtime.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.nativelib
 * Target SDK: 29+
 */

package com.syncvision.app.nativelib;

import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * JNI bridge to the Sync Vision C++ native library.
 * <p>
 * Loads the "syncvision_native" shared library and declares native
 * methods that map to the C++ functions in jni_bridge.cpp.
 * <p>
 * All native methods are designed for on-device processing — no network
 * calls, no cloud services. All data stays on the device.
 * <p>
 * Native method signatures:
 * <ul>
 *   <li>{@link #nativeProcessContours} — Extract contour lines from a segmentation mask</li>
 *   <li>{@link #nativeSimplifyContours} — Simplify contour geometry using Douglas-Peucker</li>
 *   <li>{@link #nativePlaceLabels} — Compute optimal label placement positions</li>
 *   <li>{@link #nativeFindPath} — Find navigational path using A* on depth map</li>
 *   <li>{@link #nativeGenerateSyncDiagram} — Generate relationship diagram from objects</li>
 *   <li>{@link #nativeApplyCannyEdge} — Apply Canny edge detection to an image</li>
 * </ul>
 */
public class NativeProcessor {

    private static final String TAG = "SV-NativeProcessor";

    /** Name of the native shared library (without "lib" prefix or ".so" suffix). */
    private static final String LIBRARY_NAME = "syncvision_native";

    /** Whether the native library has been successfully loaded. */
    private static boolean libraryLoaded = false;

    // ================================================================
    // Static Initialization — Load Native Library
    // ================================================================

    static {
        try {
            System.loadLibrary(LIBRARY_NAME);
            libraryLoaded = true;
            Log.i(TAG, "Native library '" + LIBRARY_NAME + "' loaded successfully");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load native library '" + LIBRARY_NAME + "'", e);
            libraryLoaded = false;
        }
    }

    // ================================================================
    // Inner Class — DetectedObject (for JNI communication)
    // ================================================================

    /**
     * Simple data class representing a detected object for passing
     * to the native layer via JNI. Fields are accessed directly
     * by the C++ code in jni_bridge.cpp::parseDetectedObjects().
     * <p>
     * IMPORTANT: Field names and types must match exactly what the
     * C++ code expects (id, name, bboxX, bboxY, bboxW, bboxH, confidence).
     */
    public static class DetectedObject {
        /** Unique object ID. */
        public int id;

        /** Object class name (e.g., "person", "car"). */
        @NonNull
        public String name;

        /** Bounding box X coordinate in pixels. */
        public int bboxX;

        /** Bounding box Y coordinate in pixels. */
        public int bboxY;

        /** Bounding box width in pixels. */
        public int bboxW;

        /** Bounding box height in pixels. */
        public int bboxH;

        /** Detection confidence [0, 1]. */
        public float confidence;

        /**
         * Creates a new DetectedObject for JNI communication.
         *
         * @param id          Object ID.
         * @param name        Class name.
         * @param bboxX       Bounding box X.
         * @param bboxY       Bounding box Y.
         * @param bboxW       Bounding box width.
         * @param bboxH       Bounding box height.
         * @param confidence  Detection confidence.
         */
        public DetectedObject(int id, @NonNull String name,
                              int bboxX, int bboxY, int bboxW, int bboxH,
                              float confidence) {
            this.id = id;
            this.name = name;
            this.bboxX = bboxX;
            this.bboxY = bboxY;
            this.bboxW = bboxW;
            this.bboxH = bboxH;
            this.confidence = confidence;
        }
    }

    // ================================================================
    // Inner Class — LabelPlacement (for JNI communication)
    // ================================================================

    /**
     * Result from the native label placement algorithm.
     * Contains the position and font size for a single label.
     * <p>
     * This class is instantiated by the C++ code via JNI when the
     * LabelPlacement Java class is available. When it is not,
     * the C++ code falls back to returning int[3] arrays.
     * <p>
     * Expected JNI class path: com/syncvision/app/nativelib/NativeProcessor$LabelPlacement
     */
    public static class LabelPlacement {
        /** X coordinate of the label position in pixels. */
        public int x;

        /** Y coordinate of the label position in pixels. */
        public int y;

        /** Font size for the label text in pixels. */
        public int fontSize;

        /** Label text content (ALL CAPS). */
        @NonNull
        public String text = "";

        /** Default constructor for JNI instantiation. */
        public LabelPlacement() {}
    }

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new NativeProcessor.
     * Ensure the native library is loaded before calling any native methods.
     */
    public NativeProcessor() {
        if (!libraryLoaded) {
            Log.w(TAG, "Native library not loaded — native methods will throw");
        }
    }

    // ================================================================
    // Library Status
    // ================================================================

    /**
     * Returns whether the native library has been successfully loaded.
     *
     * @return true if the library is available.
     */
    public static boolean isLibraryLoaded() {
        return libraryLoaded;
    }

    // ================================================================
    // Native Method Declarations
    // ================================================================

    /**
     * Processes a segmentation mask to extract contour lines.
     * <p>
     * The C++ implementation uses either OpenCV findContours() or
     * a Moore neighborhood border tracing algorithm as fallback.
     * <p>
     * Output format (flat int array):
     * <pre>
     *   [numContours, size0, size1, ..., sizeN, x0, y0, x1, y1, ...]
     * </pre>
     * Where:
     *   - numContours: number of contour groups
     *   - size0..sizeN: number of points in each contour
     *   - x,y pairs: point coordinates in pixel space
     *
     * @param maskData Flat mask data (width * height ints, class IDs per pixel).
     * @param width    Mask width.
     * @param height   Mask height.
     * @return Flat int array with contour data (header + points), or empty array on error.
     */
    @NonNull
    public native int[] nativeProcessContours(@NonNull int[] maskData, int width, int height);

    /**
     * Simplifies a contour using the Douglas-Peucker algorithm.
     * <p>
     * Reduces the number of points in a contour while preserving
     * its overall shape, controlled by the epsilon parameter.
     *
     * @param contourPoints Flat contour point array [x0, y0, x1, y1, ...].
     * @param epsilon       Simplification tolerance (higher = more simplification).
     * @return Simplified contour points [x0, y0, x1, y1, ...], or empty array on error.
     */
    @NonNull
    public native int[] nativeSimplifyContours(@NonNull int[] contourPoints, double epsilon);

    /**
     * Computes optimal label placement positions for detected objects.
     * <p>
     * Uses multi-direction placement (right → left → above → below)
     * with collision avoidance and depth-based font sizing.
     * <p>
     * Returns either an array of LabelPlacement objects (if the Java
     * class is found via JNI) or a fallback array of int[3] arrays
     * (x, y, fontSize per label).
     *
     * @param objects Array of DetectedObject instances.
     * @param width   Canvas width in pixels.
     * @param height  Canvas height in pixels.
     * @return Array of label placements, or null on error.
     */
    @Nullable
    public native Object[] nativePlaceLabels(@NonNull Object[] objects, int width, int height);

    /**
     * Finds a navigational path using A* pathfinding on a depth map.
     * <p>
     * Uses the depth map to identify walkable ground plane regions
     * and avoids obstacles. The path is simplified using line-of-sight
     * checks (Bresenham's algorithm).
     * <p>
     * Output format (flat float array):
     * <pre>
     *   [totalCost, isClear, numWaypoints, x0, y0, cost0, x1, y1, cost1, ...]
     * </pre>
     * Where:
     *   - totalCost: total path cost
     *   - isClear: 1.0 if path is clear, 0.0 if obstructed
     *   - numWaypoints: number of waypoints
     *   - x,y,cost: waypoint coordinates and individual costs
     *
     * @param depthMap  Flat depth map (width * height floats).
     * @param width     Depth map width.
     * @param height    Depth map height.
     * @param obstacles Flat obstacle array: [x0, y0, r0, x1, y1, r1, ...]
     *                  where each obstacle is (x, y, radius), or null.
     * @return Path result as flat float array, or empty array on error.
     */
    @NonNull
    public native float[] nativeFindPath(@NonNull float[] depthMap, int width, int height,
                                          @Nullable float[] obstacles);

    /**
     * Generates a relationship diagram (sync diagram) from detected objects.
     * <p>
     * Extracts relationships (ON, NEAR, CONTAINS, BLOCKS, SUPPORTS)
     * between objects and applies force-directed layout.
     * <p>
     * Output format (flat float array):
     * <pre>
     *   [numNodes, numEdges,
     *    node0: id, x, y, iconType,
     *    node1: id, x, y, iconType,
     *    ...
     *    edge0: fromId, toId, relationshipInt,
     *    edge1: fromId, toId, relationshipInt,
     *    ...]
     * </pre>
     * Where iconType: 1=person, 2=vehicle, 3=furniture, 4=electronics, 5=nature
     * And relationshipInt: 0=ON, 1=NEAR, 2=CONTAINS, 3=BLOCKS, 4=SUPPORTS
     *
     * @param objects Array of DetectedObject instances.
     * @return Diagram data as flat float array, or empty array on error.
     */
    @NonNull
    public native float[] nativeGenerateSyncDiagram(@NonNull Object[] objects);

    /**
     * Applies Canny edge detection to an image.
     * <p>
     * Uses OpenCV Canny() when available, otherwise falls back to
     * a Sobel-based edge detection implementation.
     * <p>
     * Input image format: ARGB int array (standard Android Bitmap pixel format).
     *
     * @param imageData     Flat ARGB image data (width * height ints).
     * @param width         Image width.
     * @param height        Image height.
     * @param lowThreshold  Canny low threshold (typically 50-100).
     * @param highThreshold Canny high threshold (typically 100-200).
     * @return Edge mask (0 or 255 per pixel), or empty array on error.
     */
    @NonNull
    public native int[] nativeApplyCannyEdge(@NonNull int[] imageData, int width, int height,
                                              double lowThreshold, double highThreshold);

    // ================================================================
    // Convenience Wrappers
    // ================================================================

    /**
     * Processes a 2D segmentation mask to extract contours.
     * Convenience wrapper that flattens the 2D mask array.
     *
     * @param mask2d 2D int array [height][width] of class IDs.
     * @param width  Mask width.
     * @param height Mask height.
     * @return Flat contour data, or empty array on error.
     */
    @NonNull
    public int[] processContours2d(@NonNull int[][] mask2d, int width, int height) {
        if (!libraryLoaded) {
            Log.w(TAG, "Native library not loaded, cannot process contours");
            return new int[0];
        }

        // Flatten 2D mask to 1D
        int[] flatMask = new int[width * height];
        for (int y = 0; y < height && y < mask2d.length; y++) {
            if (mask2d[y] != null) {
                int copyLen = Math.min(mask2d[y].length, width);
                System.arraycopy(mask2d[y], 0, flatMask, y * width, copyLen);
            }
        }

        return nativeProcessContours(flatMask, width, height);
    }

    /**
     * Finds a path using a 2D depth map.
     * Convenience wrapper that flattens the 2D depth array.
     *
     * @param depthMap2d 2D float array [height][width] of depth values.
     * @param width      Depth map width.
     * @param height     Depth map height.
     * @param obstacles  Flat obstacle array or null.
     * @return Path result as flat float array, or empty array on error.
     */
    @NonNull
    public float[] findPath2d(@NonNull float[][] depthMap2d, int width, int height,
                               @Nullable float[] obstacles) {
        if (!libraryLoaded) {
            Log.w(TAG, "Native library not loaded, cannot find path");
            return new float[0];
        }

        // Flatten 2D depth map to 1D
        float[] flatDepth = new float[width * height];
        for (int y = 0; y < height && y < depthMap2d.length; y++) {
            if (depthMap2d[y] != null) {
                int copyLen = Math.min(depthMap2d[y].length, width);
                System.arraycopy(depthMap2d[y], 0, flatDepth, y * width, copyLen);
            }
        }

        return nativeFindPath(flatDepth, width, height, obstacles);
    }
}
