/**
 * SyncDiagramGenerator.java
 *
 * Java wrapper that calls NativeProcessor.nativeGenerateSyncDiagram()
 * to generate the relationship diagram from detected objects.
 * Converts the C++ results into DiagramNode and DiagramEdge objects
 * used by the SyncDiagramView for rendering.
 *
 * The generator:
 *   1. Converts DetectedObject list to NativeProcessor.DetectedObject[]
 *   2. Calls nativeGenerateSyncDiagram() via JNI
 *   3. Parses the flat float result array into structured objects
 *   4. Updates the SyncDiagramView with the new diagram data
 *
 * If the native library is not available, falls back to a simple
 * Java-based diagram generator that creates proximity-based edges.
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
import com.syncvision.app.ui.SyncDiagramView;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Generates relationship diagrams from detected objects by calling
 * the C++ native sync_diagram generator and converting results
 * into Java objects for the SyncDiagramView.
 * <p>
 * The C++ implementation uses:
 *   - Relationship extraction (containment, overlap, proximity)
 *   - Force-directed layout with Coulomb repulsion, Hooke attraction
 *   - 100 iterations with damping for smooth convergence
 * <p>
 * If the native library is unavailable, a Java fallback generates
 * a simpler proximity-based diagram.
 */
public class SyncDiagramGenerator {

    private static final String TAG = "SV-SyncDiagramGenerator";

    // ================================================================
    // Components
    // ================================================================

    /** Native processor for C++ diagram generation. */
    @Nullable
    private NativeProcessor nativeProcessor;

    /** Random for fallback layout initialization. */
    private final Random random = new Random();

    /** Relationship extractor for Java fallback. */
    private final RelationshipExtractor relationshipExtractor;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new SyncDiagramGenerator.
     */
    public SyncDiagramGenerator() {
        // Try to create a NativeProcessor for C++ diagram generation
        try {
            if (NativeProcessor.isLibraryLoaded()) {
                nativeProcessor = new NativeProcessor();
            }
        } catch (Exception e) {
            Log.w(TAG, "NativeProcessor not available, using Java fallback", e);
        }

        relationshipExtractor = new RelationshipExtractor();

        Log.d(TAG, "SyncDiagramGenerator initialized (native: "
                + (nativeProcessor != null ? "YES" : "NO") + ")");
    }

    // ================================================================
    // Main Generation Method
    // ================================================================

    /**
     * Generates a relationship diagram from the given scene result
     * and updates the SyncDiagramView.
     *
     * @param result The scene result with detection data.
     * @param view   The SyncDiagramView to update.
     */
    public void generateAndUpdate(@NonNull InferenceResult.SceneResult result,
                                  @NonNull SyncDiagramView view) {
        if (result.detection == null || result.detection.objects == null
                || result.detection.objects.isEmpty()) {
            view.clear();
            return;
        }

        try {
            List<SyncDiagramView.DiagramNode> nodes;
            List<SyncDiagramView.DiagramEdge> edges;

            if (nativeProcessor != null && NativeProcessor.isLibraryLoaded()) {
                // Use C++ native diagram generation
                generateNative(result, view);
                return;
            } else {
                // Use Java fallback
                List<SyncDiagramView.DiagramNode> fallbackNodes = generateNodesFallback(result);
                List<SyncDiagramView.DiagramEdge> fallbackEdges = generateEdgesFallback(result);
                nodes = fallbackNodes;
                edges = fallbackEdges;
            }

            view.setDiagramData(nodes, edges);

        } catch (Exception e) {
            Log.e(TAG, "Error generating sync diagram", e);
        }
    }

    // ================================================================
    // Native Diagram Generation
    // ================================================================

    /**
     * Generates the diagram using the C++ native library.
     * Converts detected objects to the JNI format, calls the native
     * method, and parses the results into DiagramNode/Edge objects.
     *
     * @param result The scene result with detection data.
     * @param view   The view to update.
     */
    private void generateNative(@NonNull InferenceResult.SceneResult result,
                                @NonNull SyncDiagramView view) {
        try {
            // Convert DetectedObject list to NativeProcessor.DetectedObject[]
            NativeProcessor.DetectedObject[] nativeObjects =
                    convertToNativeFormat(result.detection.objects);

            // Call the native method
            float[] diagramData = nativeProcessor.nativeGenerateSyncDiagramSafe(nativeObjects);

            if (diagramData == null || diagramData.length < NativeConstants.DIAGRAM_HEADER_SIZE) {
                Log.w(TAG, "Native diagram generation returned insufficient data");
                // Fall back to Java generation
                List<SyncDiagramView.DiagramNode> nodes = generateNodesFallback(result);
                List<SyncDiagramView.DiagramEdge> edges = generateEdgesFallback(result);
                view.setDiagramData(nodes, edges);
                return;
            }

            // Parse the native result into DiagramNode and DiagramEdge lists
            List<SyncDiagramView.DiagramNode> nodes = new ArrayList<>();
            List<SyncDiagramView.DiagramEdge> edges = new ArrayList<>();

            int numNodes = (int) diagramData[NativeConstants.DIAGRAM_NUM_NODES_INDEX];
            int numEdges = (int) diagramData[NativeConstants.DIAGRAM_NUM_EDGES_INDEX];

            // Parse nodes
            int offset = NativeConstants.DIAGRAM_NODES_START_INDEX;
            for (int i = 0; i < numNodes
                    && offset + NativeConstants.DIAGRAM_NODE_SIZE <= diagramData.length; i++) {
                int id = (int) diagramData[offset + NativeConstants.DIAGRAM_NODE_ID_INDEX];
                float x = diagramData[offset + NativeConstants.DIAGRAM_NODE_X_INDEX];
                float y = diagramData[offset + NativeConstants.DIAGRAM_NODE_Y_INDEX];
                int iconType = (int) diagramData[offset + NativeConstants.DIAGRAM_NODE_ICON_TYPE_INDEX];

                // Find the object name from the detection list
                String name = "OBJECT";
                if (i < result.detection.objects.size()) {
                    name = result.detection.objects.get(i).name.toUpperCase(Locale.US);
                }

                SyncDiagramView.DiagramNode node =
                        new SyncDiagramView.DiagramNode(id, name, x, y, iconType);
                nodes.add(node);

                offset += NativeConstants.DIAGRAM_NODE_SIZE;
            }

            // Parse edges
            for (int i = 0; i < numEdges
                    && offset + NativeConstants.DIAGRAM_EDGE_SIZE <= diagramData.length; i++) {
                int fromId = (int) diagramData[offset + NativeConstants.DIAGRAM_EDGE_FROM_ID_INDEX];
                int toId = (int) diagramData[offset + NativeConstants.DIAGRAM_EDGE_TO_ID_INDEX];
                int relationship = (int) diagramData[offset
                        + NativeConstants.DIAGRAM_EDGE_RELATIONSHIP_INDEX];

                SyncDiagramView.DiagramEdge edge =
                        new SyncDiagramView.DiagramEdge(fromId, toId, relationship);
                edges.add(edge);

                offset += NativeConstants.DIAGRAM_EDGE_SIZE;
            }

            view.setDiagramData(nodes, edges);

        } catch (Exception e) {
            Log.e(TAG, "Error in native diagram generation", e);
            // Fall back to Java generation
            List<SyncDiagramView.DiagramNode> nodes = generateNodesFallback(result);
            List<SyncDiagramView.DiagramEdge> edges = generateEdgesFallback(result);
            view.setDiagramData(nodes, edges);
        }
    }

    // ================================================================
    // Java Fallback Generation
    // ================================================================

    /**
     * Generates diagram nodes from detected objects (Java fallback).
     * Places nodes in a circular arrangement for initial positioning;
     * the SyncDiagramView's force-directed layout will refine positions.
     *
     * @param result The scene result.
     * @return List of diagram nodes.
     */
    @NonNull
    private List<SyncDiagramView.DiagramNode> generateNodesFallback(
            @NonNull InferenceResult.SceneResult result) {
        List<SyncDiagramView.DiagramNode> nodes = new ArrayList<>();

        List<InferenceResult.DetectedObject> objects = result.detection.objects;
        int count = objects.size();

        for (int i = 0; i < count; i++) {
            InferenceResult.DetectedObject obj = objects.get(i);

            // Classify icon type based on object name
            int iconType = classifyIconType(obj.name);

            // Position nodes at their detection center (normalized coords)
            float x = obj.bbox.centerX();
            float y = obj.bbox.centerY();

            SyncDiagramView.DiagramNode node = new SyncDiagramView.DiagramNode(
                    obj.id,
                    obj.name.toUpperCase(Locale.US),
                    x, y,
                    iconType
            );

            nodes.add(node);
        }

        return nodes;
    }

    /**
     * Generates diagram edges from spatial relationships (Java fallback).
     * Uses the RelationshipExtractor to determine relationships between
     * detected object pairs.
     *
     * @param result The scene result.
     * @return List of diagram edges.
     */
    @NonNull
    private List<SyncDiagramView.DiagramEdge> generateEdgesFallback(
            @NonNull InferenceResult.SceneResult result) {
        List<SyncDiagramView.DiagramEdge> edges = new ArrayList<>();

        List<RelationshipExtractor.Relationship> relationships =
                relationshipExtractor.extractRelationships(
                        result.detection.objects, result.depth);

        for (RelationshipExtractor.Relationship rel : relationships) {
            SyncDiagramView.DiagramEdge edge = new SyncDiagramView.DiagramEdge(
                    rel.fromId, rel.toId, rel.type);
            edges.add(edge);
        }

        return edges;
    }

    // ================================================================
    // Object Conversion
    // ================================================================

    /**
     * Converts InferenceResult.DetectedObject list to the JNI format
     * required by NativeProcessor.nativeGenerateSyncDiagram().
     *
     * @param objects List of detected objects from the ML pipeline.
     * @return Array of NativeProcessor.DetectedObject for JNI.
     */
    @NonNull
    private NativeProcessor.DetectedObject[] convertToNativeFormat(
            @NonNull List<InferenceResult.DetectedObject> objects) {
        // We need to convert normalized coordinates to pixel coordinates
        // for the native layer. Use a fixed viewport size.
        int viewportW = 640;
        int viewportH = 480;

        NativeProcessor.DetectedObject[] nativeObjects =
                new NativeProcessor.DetectedObject[objects.size()];

        for (int i = 0; i < objects.size(); i++) {
            InferenceResult.DetectedObject obj = objects.get(i);

            // Convert normalized bbox to pixel coordinates
            int bboxX = (int) (obj.bbox.left * viewportW);
            int bboxY = (int) (obj.bbox.top * viewportH);
            int bboxW = (int) (obj.bbox.width() * viewportW);
            int bboxH = (int) (obj.bbox.height() * viewportH);

            nativeObjects[i] = new NativeProcessor.DetectedObject(
                    obj.id,
                    obj.name,
                    bboxX, bboxY, bboxW, bboxH,
                    obj.confidence
            );
        }

        return nativeObjects;
    }

    // ================================================================
    // Icon Type Classification
    // ================================================================

    /**
     * Classifies an object name into a diagram icon type.
     * Must match the C++ icon type classification in sync_diagram.cpp.
     *
     * @param name The detected object name.
     * @return Icon type constant from NativeConstants.
     */
    private int classifyIconType(@NonNull String name) {
        String lower = name.toLowerCase(Locale.US);

        // Person
        if (lower.contains("person") || lower.contains("people")) {
            return NativeConstants.ICON_PERSON;
        }
        // Vehicle
        if (lower.contains("car") || lower.contains("truck") || lower.contains("bus")
                || lower.contains("bicycle") || lower.contains("motorcycle")
                || lower.contains("train") || lower.contains("boat")
                || lower.contains("airplane")) {
            return NativeConstants.ICON_VEHICLE;
        }
        // Furniture
        if (lower.contains("chair") || lower.contains("couch") || lower.contains("sofa")
                || lower.contains("bed") || lower.contains("table")
                || lower.contains("dining")) {
            return NativeConstants.ICON_FURNITURE;
        }
        // Electronics
        if (lower.contains("tv") || lower.contains("laptop") || lower.contains("phone")
                || lower.contains("keyboard") || lower.contains("mouse")
                || lower.contains("remote") || lower.contains("microwave")) {
            return NativeConstants.ICON_ELECTRONICS;
        }
        // Nature (animals, plants)
        if (lower.contains("plant") || lower.contains("tree") || lower.contains("flower")
                || lower.contains("dog") || lower.contains("cat") || lower.contains("bird")
                || lower.contains("horse") || lower.contains("sheep")
                || lower.contains("cow") || lower.contains("bear")
                || lower.contains("potted")) {
            return NativeConstants.ICON_NATURE;
        }

        return NativeConstants.ICON_FURNITURE; // Default
    }
}
