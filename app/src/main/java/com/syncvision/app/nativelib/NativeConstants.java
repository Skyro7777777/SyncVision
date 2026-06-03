/**
 * NativeConstants.java
 *
 * Constants for JNI communication between the Java and C++ layers
 * of the Sync Vision native processing system. These constants define
 * the data format indices, header sizes, and enumerated values used
 * in the data arrays exchanged between Java and C++ via JNI.
 *
 * These constants MUST be kept in sync with the C++ implementations
 * in contour_processor.cpp, path_finder.cpp, sync_diagram.cpp,
 * label_placer.cpp, and jni_bridge.cpp.
 *
 * Sync Vision â€” Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.nativelib
 * Target SDK: 29+
 */

package com.syncvision.app.nativelib;

import androidx.annotation.NonNull;

/**
 * Constants that define the binary data format exchanged between the
 * Java and C++ layers via JNI. All array formats are documented here
 * so that both sides can correctly parse the data.
 *
 * <h3>Contour Data Format:</h3>
 * <pre>
 *   [numContours, size0, size1, ..., sizeN, x0, y0, x1, y1, ...]
 * </pre>
 *
 * <h3>Path Result Format:</h3>
 * <pre>
 *   [totalCost, isClear, numWaypoints, x0, y0, cost0, x1, y1, cost1, ...]
 * </pre>
 *
 * <h3>Diagram Result Format:</h3>
 * <pre>
 *   [numNodes, numEdges, node0(id,x,y,iconType), ..., edge0(fromId,toId,rel), ...]
 * </pre>
 */
public final class NativeConstants {

    // Private constructor â€” this is a constants-only class
    private NativeConstants() {
        throw new AssertionError("NativeConstants is a constants class; do not instantiate.");
    }

    // ================================================================
    // Contour Data Format Constants
    // ================================================================

    /**
     * Header size for contour data arrays.
     * The header contains: [numContours, size0, size1, ..., sizeN]
     * Size = 1 (for numContours) + numContours (for each contour's point count).
     * This is the minimum header size when numContours = 0.
     */
    public static final int CONTOUR_HEADER_MIN_SIZE = 1;

    /**
     * Index of the numContours field in the contour data header.
     */
    public static final int CONTOUR_NUM_CONTOURS_INDEX = 0;

    /**
     * Starting index for contour size entries in the header.
     * Each entry gives the number of points in that contour.
     */
    public static final int CONTOUR_SIZES_START_INDEX = 1;

    /**
     * Number of floats/ints per contour point (x, y).
     */
    public static final int CONTOUR_POINT_SIZE = 2;

    /**
     * Index of X coordinate within a point pair.
     */
    public static final int CONTOUR_POINT_X_INDEX = 0;

    /**
     * Index of Y coordinate within a point pair.
     */
    public static final int CONTOUR_POINT_Y_INDEX = 1;

    // ================================================================
    // Path Result Format Constants
    // ================================================================

    /**
     * Header size for path result arrays.
     * The header contains: [totalCost, isClear, numWaypoints]
     */
    public static final int PATH_HEADER_SIZE = 3;

    /**
     * Index of the totalCost field in the path result header.
     */
    public static final int PATH_TOTAL_COST_INDEX = 0;

    /**
     * Index of the isClear field in the path result header.
     * Value: 1.0 = clear path, 0.0 = obstructed path.
     */
    public static final int PATH_IS_CLEAR_INDEX = 1;

    /**
     * Index of the numWaypoints field in the path result header.
     */
    public static final int PATH_NUM_WAYPOINTS_INDEX = 2;

    /**
     * Starting index for waypoint data (after the header).
     */
    public static final int PATH_WAYPOINTS_START_INDEX = 3;

    /**
     * Number of floats per waypoint entry (x, y, cost).
     */
    public static final int PATH_WAYPOINT_SIZE = 3;

    /**
     * Index of X coordinate within a waypoint entry.
     */
    public static final int PATH_WAYPOINT_X_INDEX = 0;

    /**
     * Index of Y coordinate within a waypoint entry.
     */
    public static final int PATH_WAYPOINT_Y_INDEX = 1;

    /**
     * Index of cost value within a waypoint entry.
     */
    public static final int PATH_WAYPOINT_COST_INDEX = 2;

    // ================================================================
    // Diagram Result Format Constants
    // ================================================================

    /**
     * Header size for diagram result arrays.
     * The header contains: [numNodes, numEdges]
     */
    public static final int DIAGRAM_HEADER_SIZE = 2;

    /**
     * Index of the numNodes field in the diagram result header.
     */
    public static final int DIAGRAM_NUM_NODES_INDEX = 0;

    /**
     * Index of the numEdges field in the diagram result header.
     */
    public static final int DIAGRAM_NUM_EDGES_INDEX = 1;

    /**
     * Starting index for node data (after the header).
     */
    public static final int DIAGRAM_NODES_START_INDEX = 2;

    /**
     * Number of floats per node entry (id, x, y, iconType).
     */
    public static final int DIAGRAM_NODE_SIZE = 4;

    /**
     * Index of node ID within a node entry.
     */
    public static final int DIAGRAM_NODE_ID_INDEX = 0;

    /**
     * Index of node X coordinate within a node entry.
     */
    public static final int DIAGRAM_NODE_X_INDEX = 1;

    /**
     * Index of node Y coordinate within a node entry.
     */
    public static final int DIAGRAM_NODE_Y_INDEX = 2;

    /**
     * Index of node icon type within a node entry.
     */
    public static final int DIAGRAM_NODE_ICON_TYPE_INDEX = 3;

    /**
     * Number of floats per edge entry (fromId, toId, relationship).
     */
    public static final int DIAGRAM_EDGE_SIZE = 3;

    /**
     * Index of source node ID within an edge entry.
     */
    public static final int DIAGRAM_EDGE_FROM_ID_INDEX = 0;

    /**
     * Index of destination node ID within an edge entry.
     */
    public static final int DIAGRAM_EDGE_TO_ID_INDEX = 1;

    /**
     * Index of relationship type within an edge entry.
     */
    public static final int DIAGRAM_EDGE_RELATIONSHIP_INDEX = 2;

    // ================================================================
    // Diagram Icon Types
    // ================================================================

    /** Icon type: person / human figure. */
    public static final int ICON_PERSON = 1;

    /** Icon type: vehicle (car, truck, bus, etc.). */
    public static final int ICON_VEHICLE = 2;

    /** Icon type: furniture (chair, couch, table, etc.). */
    public static final int ICON_FURNITURE = 3;

    /** Icon type: electronics (phone, laptop, TV, etc.). */
    public static final int ICON_ELECTRONICS = 4;

    /** Icon type: nature (plant, tree, flower, etc.). */
    public static final int ICON_NATURE = 5;

    // ================================================================
    // Diagram Relationship Types
    // ================================================================

    /** Relationship: object A is ON top of object B. */
    public static final int RELATIONSHIP_ON = 0;

    /** Relationship: object A is NEAR object B (proximity). */
    public static final int RELATIONSHIP_NEAR = 1;

    /** Relationship: object A CONTAINS object B. */
    public static final int RELATIONSHIP_CONTAINS = 2;

    /** Relationship: object A BLOCKS (occludes) object B. */
    public static final int RELATIONSHIP_BLOCKS = 3;

    /** Relationship: object A SUPPORTS object B (structural). */
    public static final int RELATIONSHIP_SUPPORTS = 4;

    // ================================================================
    // Hazard Levels
    // ================================================================

    /** No hazard â€” safe to traverse/approach. */
    public static final int HAZARD_NONE = 0;

    /** Low hazard â€” minor risk, proceed with caution. */
    public static final int HAZARD_LOW = 1;

    /** Medium hazard â€” moderate risk, consider alternative route. */
    public static final int HAZARD_MEDIUM = 2;

    /** High hazard â€” significant risk, avoid if possible. */
    public static final int HAZARD_HIGH = 3;

    // ================================================================
    // Label Placement Format Constants
    // ================================================================

    /**
     * Number of ints per label placement in the fallback format
     * (when LabelPlacement class is not available via JNI).
     * Format: [x, y, fontSize]
     */
    public static final int LABEL_PLACEMENT_FALLBACK_SIZE = 3;

    /**
     * Index of X coordinate in fallback label placement.
     */
    public static final int LABEL_PLACEMENT_X_INDEX = 0;

    /**
     * Index of Y coordinate in fallback label placement.
     */
    public static final int LABEL_PLACEMENT_Y_INDEX = 1;

    /**
     * Index of font size in fallback label placement.
     */
    public static final int LABEL_PLACEMENT_FONT_SIZE_INDEX = 2;

    // ================================================================
    // Edge Detection Constants
    // ================================================================

    /** Default low threshold for Canny edge detection. */
    public static final double CANNY_DEFAULT_LOW_THRESHOLD = 50.0;

    /** Default high threshold for Canny edge detection. */
    public static final double CANNY_DEFAULT_HIGH_THRESHOLD = 150.0;

    /** Edge mask value indicating an edge pixel. */
    public static final int EDGE_PIXEL_VALUE = 255;

    /** Edge mask value indicating a non-edge pixel. */
    public static final int NON_EDGE_PIXEL_VALUE = 0;

    // ================================================================
    // Path Status Constants (matches path_fragment.glsl)
    // ================================================================

    /** Path status: clear of obstacles. */
    public static final float PATH_STATUS_CLEAR = 0.0f;

    /** Path status: partially obstructed. */
    public static final float PATH_STATUS_PARTIAL = 1.0f;

    /** Path status: fully blocked. */
    public static final float PATH_STATUS_BLOCKED = 2.0f;

    // ================================================================
    // Utility Methods
    // ================================================================

    /**
     * Returns a human-readable string for a hazard level.
     *
     * @param level Hazard level constant.
     * @return Descriptive string.
     */
    @NonNull
    public static String hazardLevelToString(int level) {
        switch (level) {
            case HAZARD_NONE:   return "NONE";
            case HAZARD_LOW:    return "LOW";
            case HAZARD_MEDIUM: return "MEDIUM";
            case HAZARD_HIGH:   return "HIGH";
            default:            return "UNKNOWN(" + level + ")";
        }
    }

    /**
     * Returns a human-readable string for a diagram relationship type.
     *
     * @param relationship Relationship constant.
     * @return Descriptive string.
     */
    @NonNull
    public static String relationshipToString(int relationship) {
        switch (relationship) {
            case RELATIONSHIP_ON:       return "ON";
            case RELATIONSHIP_NEAR:     return "NEAR";
            case RELATIONSHIP_CONTAINS: return "CONTAINS";
            case RELATIONSHIP_BLOCKS:   return "BLOCKS";
            case RELATIONSHIP_SUPPORTS: return "SUPPORTS";
            default:                    return "UNKNOWN(" + relationship + ")";
        }
    }

    /**
     * Returns a human-readable string for a diagram icon type.
     *
     * @param iconType Icon type constant.
     * @return Descriptive string.
     */
    @NonNull
    public static String iconTypeToString(int iconType) {
        switch (iconType) {
            case ICON_PERSON:      return "PERSON";
            case ICON_VEHICLE:     return "VEHICLE";
            case ICON_FURNITURE:   return "FURNITURE";
            case ICON_ELECTRONICS: return "ELECTRONICS";
            case ICON_NATURE:      return "NATURE";
            default:               return "UNKNOWN(" + iconType + ")";
        }
    }

    /**
     * Returns a human-readable string for a path status.
     *
     * @param status Path status constant.
     * @return Descriptive string.
     */
    @NonNull
    public static String pathStatusToString(float status) {
        if (status < 0.5f) return "CLEAR";
        if (status < 1.5f) return "PARTIAL";
        return "BLOCKED";
    }
}
