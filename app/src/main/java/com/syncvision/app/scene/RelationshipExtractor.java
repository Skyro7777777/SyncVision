/**
 * RelationshipExtractor.java
 *
 * Analyzes spatial relationships between detected objects in the scene.
 * Determines relationships like ON, NEAR, CONTAINS, BLOCKS, SUPPORTS
 * based on bounding box overlap, proximity, and depth ordering.
 *
 * This Java-side extractor is used when the C++ native sync_diagram
 * is not available, or as a complement to it. It provides the same
 * relationship types as the C++ implementation:
 *   - ON:       Object A is vertically above and overlapping object B
 *   - NEAR:     Objects A and B are in close proximity
 *   - CONTAINS: Object A's bbox fully contains object B's bbox
 *   - BLOCKS:   Object A overlaps and is closer (from depth) than B
 *   - SUPPORTS: Object B rests on top of object A (inverse of ON)
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

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Extracts spatial relationships between pairs of detected objects.
 * <p>
 * Relationships are determined by analyzing:
 * <ul>
 *   <li>Bounding box overlap (IoU)</li>
 *   <li>Relative vertical position (one above the other)</li>
 *   <li>Containment (one bbox fully inside another)</li>
 *   <li>Depth ordering (which object is closer to camera)</li>
 *   <li>Proximity (distance between bbox centers)</li>
 * </ul>
 */
public class RelationshipExtractor {

    private static final String TAG = "SV-RelationshipExtractor";

    // ================================================================
    // Thresholds
    // ================================================================

    /** IoU threshold for considering objects as overlapping. */
    private static final float OVERLAP_IOU_THRESHOLD = 0.1f;

    /** IoU threshold for containment relationship. */
    private static final float CONTAINMENT_IOU_THRESHOLD = 0.7f;

    /** Vertical offset threshold for ON/SUPPORTS (normalized). */
    private static final float VERTICAL_OFFSET_THRESHOLD = 0.1f;

    /** Proximity threshold for NEAR relationship (normalized). */
    private static final float NEAR_DISTANCE_THRESHOLD = 0.3f;

    // ================================================================
    // Inner Class — Relationship
    // ================================================================

    /**
     * Represents a spatial relationship between two objects.
     */
    public static class Relationship {
        /** Source object ID. */
        public int fromId;

        /** Destination object ID. */
        public int toId;

        /** Relationship type (ON, NEAR, CONTAINS, BLOCKS, SUPPORTS). */
        public int type;

        /** Confidence in this relationship [0, 1]. */
        public float confidence;

        /** Human-readable description. */
        @NonNull
        public String description;

        public Relationship(int fromId, int toId, int type,
                            float confidence, @NonNull String description) {
            this.fromId = fromId;
            this.toId = toId;
            this.type = type;
            this.confidence = confidence;
            this.description = description;
        }
    }

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new RelationshipExtractor.
     */
    public RelationshipExtractor() {
        Log.d(TAG, "RelationshipExtractor initialized");
    }

    // ================================================================
    // Main Extraction Method
    // ================================================================

    /**
     * Extracts spatial relationships between all pairs of detected objects.
     *
     * @param objects List of detected objects.
     * @param depth   Depth result (nullable, used for BLOCKS determination).
     * @return List of extracted relationships.
     */
    @NonNull
    public List<Relationship> extractRelationships(
            @NonNull List<InferenceResult.DetectedObject> objects,
            @Nullable InferenceResult.DepthResult depth) {
        List<Relationship> relationships = new ArrayList<>();

        if (objects.size() < 2) {
            return relationships;
        }

        try {
            for (int i = 0; i < objects.size(); i++) {
                for (int j = i + 1; j < objects.size(); j++) {
                    InferenceResult.DetectedObject a = objects.get(i);
                    InferenceResult.DetectedObject b = objects.get(j);

                    Relationship rel = determineRelationship(a, b, depth);
                    if (rel != null) {
                        relationships.add(rel);
                    }

                    // Check the reverse relationship
                    Relationship relReverse = determineRelationship(b, a, depth);
                    if (relReverse != null) {
                        relationships.add(relReverse);
                    }
                }
            }

        } catch (Exception e) {
            Log.e(TAG, "Error extracting relationships", e);
        }

        return relationships;
    }

    /**
     * Convenience overload without depth information.
     */
    @NonNull
    public List<Relationship> extractRelationships(
            @NonNull List<InferenceResult.DetectedObject> objects) {
        return extractRelationships(objects, null);
    }

    // ================================================================
    // Relationship Determination
    // ================================================================

    /**
     * Determines the relationship from object A to object B.
     * Only one relationship type is returned per direction.
     *
     * Priority order:
     *   1. CONTAINS (A fully contains B)
     *   2. BLOCKS (A overlaps B and is closer)
     *   3. ON (A is above B with vertical overlap)
     *   4. SUPPORTS (B is above A with vertical overlap)
     *   5. NEAR (A and B are in close proximity)
     *
     * @param a     First object.
     * @param b     Second object.
     * @param depth Depth result (nullable).
     * @return A relationship, or null if no significant relationship exists.
     */
    @Nullable
    private Relationship determineRelationship(
            @NonNull InferenceResult.DetectedObject a,
            @NonNull InferenceResult.DetectedObject b,
            @Nullable InferenceResult.DepthResult depth) {

        float iou = computeIoU(a.bbox, b.bbox);
        float centerDist = computeCenterDistance(a.bbox, b.bbox);

        // 1. CONTAINS: A's bbox fully contains B's bbox
        if (containsFully(a.bbox, b.bbox)) {
            return new Relationship(a.id, b.id,
                    NativeConstants.RELATIONSHIP_CONTAINS,
                    Math.min(1.0f, 0.7f + iou * 0.3f),
                    a.name + " CONTAINS " + b.name);
        }

        // 2. BLOCKS: A overlaps B and is closer (from depth)
        if (iou > OVERLAP_IOU_THRESHOLD && depth != null) {
            float depthA = depth.getDepthAt(a.bbox.centerX(), a.bbox.centerY());
            float depthB = depth.getDepthAt(b.bbox.centerX(), b.bbox.centerY());
            if (depthA > depthB) {
                // A is closer (MiDaS: higher = closer)
                return new Relationship(a.id, b.id,
                        NativeConstants.RELATIONSHIP_BLOCKS,
                        Math.min(1.0f, 0.5f + iou * 0.5f),
                        a.name + " BLOCKS " + b.name);
            }
        }

        // 3. ON: A is vertically above B with horizontal overlap
        float verticalOffset = a.bbox.centerY() - b.bbox.centerY();
        boolean horizontallyOverlapping = a.bbox.intersects(b.bbox);

        if (verticalOffset < -VERTICAL_OFFSET_THRESHOLD && horizontallyOverlapping) {
            // A is above B
            return new Relationship(a.id, b.id,
                    NativeConstants.RELATIONSHIP_ON,
                    Math.min(1.0f, 0.4f + iou * 0.6f),
                    a.name + " ON " + b.name);
        }

        // 4. SUPPORTS: B is above A (A supports B)
        if (verticalOffset > VERTICAL_OFFSET_THRESHOLD && horizontallyOverlapping) {
            return new Relationship(a.id, b.id,
                    NativeConstants.RELATIONSHIP_SUPPORTS,
                    Math.min(1.0f, 0.4f + iou * 0.6f),
                    a.name + " SUPPORTS " + b.name);
        }

        // 5. NEAR: A and B are in close proximity
        if (centerDist < NEAR_DISTANCE_THRESHOLD) {
            float conf = 1.0f - (centerDist / NEAR_DISTANCE_THRESHOLD);
            return new Relationship(a.id, b.id,
                    NativeConstants.RELATIONSHIP_NEAR,
                    Math.max(0.2f, conf),
                    a.name + " NEAR " + b.name);
        }

        return null; // No significant relationship
    }

    // ================================================================
    // Geometry Helpers
    // ================================================================

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

    /**
     * Computes the Euclidean distance between the centers of two bboxes.
     *
     * @return Distance in normalized coordinates [0, ~1.4].
     */
    private float computeCenterDistance(@NonNull InferenceResult.RectF a,
                                       @NonNull InferenceResult.RectF b) {
        float dx = a.centerX() - b.centerX();
        float dy = a.centerY() - b.centerY();
        return (float) Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Checks if bounding box a fully contains bounding box b.
     * A contains B if all corners of B are within A, with some tolerance.
     *
     * @param a Outer bounding box.
     * @param b Inner bounding box.
     * @return true if A contains B (with some margin).
     */
    private boolean containsFully(@NonNull InferenceResult.RectF a,
                                  @NonNull InferenceResult.RectF b) {
        float margin = 0.02f; // Small margin for numerical tolerance
        return a.left <= b.left + margin
                && a.top <= b.top + margin
                && a.right >= b.right - margin
                && a.bottom >= b.bottom - margin;
    }
}
