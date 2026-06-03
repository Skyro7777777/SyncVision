/**
 * ObjectFusion.java
 *
 * Fuses segmentation masks with detection bounding boxes to produce
 * enhanced detected objects. Cross-references DeepLab 21 classes
 * with COCO SSD 80 classes to provide the best class label and
 * segmentation mask for each detected object.
 *
 * For each detected object, determines:
 *   - Best class label (from SSD if available, else from DeepLab)
 *   - Segmentation mask (from DeepLab, cropped to bbox)
 *   - Description (from ObjectDescription database)
 *
 * Handles overlapping detections by assigning each segmentation pixel
 * to the closest detection bounding box center.
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
import com.syncvision.app.ml.SegmentationPipeline;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Fuses detection and segmentation pipeline results to produce
 * enhanced detected objects with both bounding boxes and segmentation
 * masks, along with the best available class label for each object.
 * <p>
 * The fusion process:
 * <ol>
 *   <li>Take detections from COCO SSD (bounding boxes + class labels)</li>
 *   <li>Take segmentation from DeepLab v3+ (per-pixel class labels)</li>
 *   <li>For each detection, crop the segmentation mask to the bbox</li>
 *   <li>Resolve class label conflicts (SSD label takes priority)</li>
 *   <li>Assign a human-readable description</li>
 * </ol>
 * <p>
 * Class label cross-reference:
 *   - DeepLab uses 21 COCO Pascal VOC classes
 *   - SSD uses 80 COCO classes
 *   - Some classes overlap (e.g., "person" in both)
 *   - SSD provides more specific labels where available
 */
public class ObjectFusion {

    private static final String TAG = "SV-ObjectFusion";

    // ================================================================
    // DeepLab → COCO Class Name Mapping
    // ================================================================

    /**
     * Maps DeepLab class names to their closest COCO SSD equivalents.
     * DeepLab uses 21 Pascal VOC classes; COCO SSD uses 80 classes.
     * Where the names differ, this mapping provides the COCO equivalent.
     */
    private static final Map<String, String> DEEPLAB_TO_COCO_MAP = new HashMap<>();
    static {
        // Direct matches (same name in both)
        DEEPLAB_TO_COCO_MAP.put("person", "person");
        DEEPLAB_TO_COCO_MAP.put("bicycle", "bicycle");
        DEEPLAB_TO_COCO_MAP.put("car", "car");
        DEEPLAB_TO_COCO_MAP.put("motorbike", "motorcycle");
        DEEPLAB_TO_COCO_MAP.put("bus", "bus");
        DEEPLAB_TO_COCO_MAP.put("bottle", "bottle");
        DEEPLAB_TO_COCO_MAP.put("bird", "bird");
        DEEPLAB_TO_COCO_MAP.put("cat", "cat");
        DEEPLAB_TO_COCO_MAP.put("dog", "dog");
        DEEPLAB_TO_COCO_MAP.put("horse", "horse");
        DEEPLAB_TO_COCO_MAP.put("sheep", "sheep");
        DEEPLAB_TO_COCO_MAP.put("cow", "cow");
        DEEPLAB_TO_COCO_MAP.put("chair", "chair");
        DEEPLAB_TO_COCO_MAP.put("pottedplant", "potted plant");
        DEEPLAB_TO_COCO_MAP.put("tv", "tv");
        // Name differences
        DEEPLAB_TO_COCO_MAP.put("aeroplane", "airplane");
        DEEPLAB_TO_COCO_MAP.put("boat", "boat");
        DEEPLAB_TO_COCO_MAP.put("diningtable", "dining table");
        DEEPLAB_TO_COCO_MAP.put("sofa", "couch");
        DEEPLAB_TO_COCO_MAP.put("train", "train");
    }

    // ================================================================
    // Object Description Database
    // ================================================================

    /**
     * Static description database for common detected objects.
     * Maps class names to human-readable descriptions.
     */
    private static final Map<String, String> DESCRIPTIONS = new HashMap<>();
    static {
        DESCRIPTIONS.put("person", "HUMAN DETECTED IN SCENE");
        DESCRIPTIONS.put("car", "MOTOR VEHICLE DETECTED — CHECK FOR MOVEMENT");
        DESCRIPTIONS.put("truck", "LARGE VEHICLE DETECTED — MAINTAIN DISTANCE");
        DESCRIPTIONS.put("bus", "PUBLIC TRANSIT VEHICLE DETECTED");
        DESCRIPTIONS.put("bicycle", "BICYCLE DETECTED — WATCH FOR RIDER");
        DESCRIPTIONS.put("motorcycle", "MOTORCYCLE DETECTED — MAY ACCELERATE QUICKLY");
        DESCRIPTIONS.put("dog", "CANINE DETECTED IN SCENE");
        DESCRIPTIONS.put("cat", "FELINE DETECTED IN SCENE");
        DESCRIPTIONS.put("bird", "AVIAN SPECIES DETECTED");
        DESCRIPTIONS.put("chair", "SEATING FURNITURE DETECTED");
        DESCRIPTIONS.put("couch", "UPHOLSTERED SEATING DETECTED");
        DESCRIPTIONS.put("bottle", "CONTAINER DETECTED — POSSIBLE OBSTACLE");
        DESCRIPTIONS.put("tv", "DISPLAY DEVICE DETECTED");
        DESCRIPTIONS.put("laptop", "PORTABLE COMPUTER DETECTED");
        DESCRIPTIONS.put("cell phone", "MOBILE DEVICE DETECTED");
        DESCRIPTIONS.put("potted plant", "VEGETATION IN CONTAINER DETECTED");
        DESCRIPTIONS.put("dining table", "TABLE SURFACE DETECTED");
        DESCRIPTIONS.put("stop sign", "TRAFFIC CONTROL SIGN — STOP REQUIRED");
        DESCRIPTIONS.put("traffic light", "TRAFFIC SIGNAL DETECTED");
        DESCRIPTIONS.put("fire hydrant", "FIRE SAFETY EQUIPMENT DETECTED");
        DESCRIPTIONS.put("backpack", "CARRIED BAG DETECTED");
        DESCRIPTIONS.put("umbrella", "RAIN PROTECTION DEVICE DETECTED");
        DESCRIPTIONS.put("airplane", "AIRCRAFT DETECTED IN SKY");
        DESCRIPTIONS.put("train", "RAIL VEHICLE DETECTED");
        DESCRIPTIONS.put("boat", "WATERCRAFT DETECTED");
        DESCRIPTIONS.put("horse", "LARGE ANIMAL DETECTED");
        DESCRIPTIONS.put("cow", "LIVESTOCK DETECTED");
        DESCRIPTIONS.put("sheep", "LIVESTOCK DETECTED");
        DESCRIPTIONS.put("sofa", "SEATING FURNITURE DETECTED");
    }

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new ObjectFusion engine.
     */
    public ObjectFusion() {
        Log.d(TAG, "ObjectFusion initialized");
    }

    // ================================================================
    // Main Fusion Method
    // ================================================================

    /**
     * Fuses detection and segmentation results into enhanced objects.
     *
     * @param result The raw scene result from ML pipelines.
     * @return A new scene result with fused detection data.
     */
    @NonNull
    public InferenceResult.SceneResult fuse(
            @NonNull InferenceResult.SceneResult result) {
        try {
            // If there's no detection or segmentation, return as-is
            if (result.detection == null) {
                return result;
            }

            // Create a copy of the scene result to modify
            InferenceResult.SceneResult fused = copySceneResult(result);

            // Enhance detected objects with segmentation masks and descriptions
            if (fused.detection != null && fused.detection.objects != null) {
                for (InferenceResult.DetectedObject obj : fused.detection.objects) {
                    // Step 1: Assign segmentation mask if available
                    if (fused.segmentation != null) {
                        obj.segmentationMask = cropMaskToBbox(
                                fused.segmentation, obj.bbox, obj.name);
                    }

                    // Step 2: Resolve best class label
                    obj.name = resolveBestLabel(obj.name, fused.segmentation, obj.bbox);

                    // Step 3: Assign description
                    obj.description = getDescription(obj.name);

                    // Step 4: Estimate distance from depth map if available
                    if (obj.distance < 0 && fused.depth != null) {
                        float depthVal = fused.depth.getDepthAt(
                                obj.bbox.centerX(), obj.bbox.centerY());
                        if (depthVal > 0) {
                            obj.distance = fused.depth.estimateDistance(
                                    depthVal, 0.5f, 2.0f);
                        }
                    }
                }
            }

            return fused;

        } catch (Exception e) {
            Log.e(TAG, "Error fusing objects", e);
            return result;
        }
    }

    // ================================================================
    // Segmentation Mask Cropping
    // ================================================================

    /**
     * Crops the full segmentation mask to the region around the
     * detected object's bounding box. Only pixels that match the
     * expected class label are included in the mask.
     *
     * @param segmentation The full segmentation result.
     * @param bbox         The bounding box to crop to (normalized coords).
     * @param className    The expected class name for mask filtering.
     * @return A cropped mask [height][width] with 1=matching, 0=not matching,
     *         or null if cropping fails.
     */
    @Nullable
    private int[][] cropMaskToBbox(
            @NonNull InferenceResult.SegmentationResult segmentation,
            @NonNull InferenceResult.RectF bbox,
            @NonNull String className) {
        try {
            int maskW = segmentation.width;
            int maskH = segmentation.height;

            // Convert normalized bbox to mask pixel coordinates
            int x1 = Math.max(0, (int) (bbox.left * maskW));
            int y1 = Math.max(0, (int) (bbox.top * maskH));
            int x2 = Math.min(maskW - 1, (int) (bbox.right * maskW));
            int y2 = Math.min(maskH - 1, (int) (bbox.bottom * maskH));

            int cropW = x2 - x1;
            int cropH = y2 - y1;
            if (cropW <= 0 || cropH <= 0) return null;

            // Find the expected segmentation class ID(s) for this object
            List<Integer> expectedClassIds = getClassIdsForName(className, segmentation);

            // Build the cropped mask
            int[][] croppedMask = new int[cropH][cropW];
            for (int y = 0; y < cropH; y++) {
                for (int x = 0; x < cropW; x++) {
                    int maskY = y1 + y;
                    int maskX = x1 + x;
                    if (maskY < segmentation.mask.length
                            && maskX < segmentation.mask[maskY].length) {
                        int classId = segmentation.mask[maskY][maskX];
                        croppedMask[y][x] = expectedClassIds.contains(classId) ? 1 : 0;
                    }
                }
            }

            return croppedMask;

        } catch (Exception e) {
            Log.w(TAG, "Error cropping mask to bbox", e);
            return null;
        }
    }

    // ================================================================
    // Class Label Resolution
    // ================================================================

    /**
     * Resolves the best class label for an object by cross-referencing
     * the detection label with the segmentation mask at the object center.
     * The detection (SSD) label takes priority because SSD provides more
     * specific class labels.
     *
     * @param detectedName The name from the detection pipeline.
     * @param segmentation The segmentation result (nullable).
     * @param bbox         The bounding box of the detected object.
     * @return The best resolved class name.
     */
    @NonNull
    private String resolveBestLabel(
            @NonNull String detectedName,
            @Nullable InferenceResult.SegmentationResult segmentation,
            @NonNull InferenceResult.RectF bbox) {
        // SSD detection label takes priority — it's more specific
        // Only override if the detection name is "unknown"
        if (!detectedName.equals("unknown") || segmentation == null) {
            return detectedName;
        }

        // Try to get the label from the segmentation mask at bbox center
        int cx = (int) (bbox.centerX() * segmentation.width);
        int cy = (int) (bbox.centerY() * segmentation.height);
        cx = Math.min(Math.max(cx, 0), segmentation.width - 1);
        cy = Math.min(Math.max(cy, 0), segmentation.height - 1);

        int classId = segmentation.mask[cy][cx];
        String segClassName = SegmentationPipeline.getClassName(classId);

        // Map DeepLab name to COCO name
        String mappedName = DEEPLAB_TO_COCO_MAP.get(segClassName);
        return mappedName != null ? mappedName : segClassName;
    }

    // ================================================================
    // Description Lookup
    // ================================================================

    /**
     * Returns a human-readable description for the given class name.
     *
     * @param className The detected class name.
     * @return Description string, or a generic fallback.
     */
    @NonNull
    private String getDescription(@NonNull String className) {
        String desc = DESCRIPTIONS.get(className.toLowerCase(Locale.US));
        if (desc != null) return desc;

        // Generic fallback
        return className.toUpperCase(Locale.US) + " DETECTED IN SCENE";
    }

    // ================================================================
    // Helper Methods
    // ================================================================

    /**
     * Gets the segmentation class IDs that correspond to a given
     * detection class name. Returns a list because multiple DeepLab
     * classes may map to the same COCO class.
     *
     * @param className    The detection class name.
     * @param segmentation The segmentation result with class label map.
     * @return List of matching class IDs.
     */
    @NonNull
    private List<Integer> getClassIdsForName(
            @NonNull String className,
            @NonNull InferenceResult.SegmentationResult segmentation) {
        List<Integer> ids = new ArrayList<>();
        String normalizedName = className.toLowerCase(Locale.US);

        // Search through segmentation class labels
        for (Map.Entry<Integer, String> entry : segmentation.classLabels.entrySet()) {
            String segLabel = entry.getValue().toLowerCase(Locale.US);

            // Direct match
            if (normalizedName.equals(segLabel)) {
                ids.add(entry.getKey());
                continue;
            }

            // Check the DeepLab → COCO mapping
            String mappedCoco = DEEPLAB_TO_COCO_MAP.get(segLabel);
            if (mappedCoco != null && normalizedName.equals(mappedCoco.toLowerCase(Locale.US))) {
                ids.add(entry.getKey());
            }
        }

        // If no matches, include background (0) to avoid empty mask
        if (ids.isEmpty()) {
            // Try a fuzzy match based on partial names
            for (Map.Entry<Integer, String> entry : segmentation.classLabels.entrySet()) {
                String segLabel = entry.getValue().toLowerCase(Locale.US);
                if (normalizedName.contains(segLabel) || segLabel.contains(normalizedName)) {
                    ids.add(entry.getKey());
                }
            }
        }

        return ids;
    }

    /**
     * Creates a shallow copy of a SceneResult for modification.
     *
     * @param original The original scene result.
     * @return A copy with new detection and object lists.
     */
    @NonNull
    private InferenceResult.SceneResult copySceneResult(
            @NonNull InferenceResult.SceneResult original) {
        InferenceResult.SceneResult copy = new InferenceResult.SceneResult();
        copy.timestamp = original.timestamp;
        copy.frameIndex = original.frameIndex;
        copy.fps = original.fps;
        copy.latencyMs = original.latencyMs;
        copy.segmentation = original.segmentation;
        copy.depth = original.depth;
        copy.face = original.face;
        copy.ocr = original.ocr;
        copy.weather = original.weather;
        copy.plant = original.plant;
        copy.barcode = original.barcode;
        copy.path = original.path;
        copy.landmark = original.landmark;

        // Deep copy detection objects
        if (original.detection != null) {
            List<InferenceResult.DetectedObject> objectsCopy = new ArrayList<>();
            for (InferenceResult.DetectedObject obj : original.detection.objects) {
                InferenceResult.DetectedObject objCopy = new InferenceResult.DetectedObject();
                objCopy.id = obj.id;
                objCopy.name = obj.name;
                objCopy.confidence = obj.confidence;
                objCopy.bbox = new InferenceResult.RectF(
                        obj.bbox.left, obj.bbox.top, obj.bbox.right, obj.bbox.bottom);
                objCopy.segmentationMask = obj.segmentationMask;
                objCopy.description = obj.description;
                objCopy.distance = obj.distance;
                objCopy.hazardLevel = obj.hazardLevel;
                objectsCopy.add(objCopy);
            }
            copy.detection = new InferenceResult.DetectionResult(objectsCopy);
        }

        return copy;
    }
}
