/**
 * InferenceResult.java
 *
 * Data classes for all ML inference results in the Sync Vision app.
 * Contains structured result types for each pipeline: detection,
 * segmentation, depth, face, OCR, weather, plant, barcode, path,
 * landmark, and the combined SceneResult.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.ml
 * Target SDK: 29+
 */

package com.syncvision.app.ml;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.List;
import java.util.Map;

/**
 * Container for all ML inference result types.
 * Each inner class represents the output of a specific pipeline.
 */
public final class InferenceResult {

    // Private constructor — this is a data class container
    private InferenceResult() {
        throw new AssertionError("InferenceResult is a container class; do not instantiate.");
    }

    // ================================================================
    // DetectedObject — Single detected object with optional mask
    // ================================================================

    /**
     * Represents a single detected object from the detection + segmentation pipelines.
     * Combines bounding box, class label, confidence, optional segmentation mask,
     * and derived properties like distance estimate and hazard level.
     */
    public static class DetectedObject {
        /** Unique identifier for this detection within the frame. */
        public int id;

        /** Human-readable class name (e.g., "person", "car"). */
        @NonNull
        public String name = "";

        /** Detection confidence in [0, 1]. */
        public float confidence;

        /** Bounding box in normalized coordinates [0, 1]. */
        @NonNull
        public RectF bbox = new RectF();

        /**
         * Per-pixel segmentation mask for this object (nullable).
         * Non-null only when segmentation pipeline provides a mask
         * that corresponds to this detection.
         * Dimensions: [height][width], values are 0 or 1.
         */
        @Nullable
        public int[][] segmentationMask;

        /** Brief description of the object for display. */
        @Nullable
        public String description;

        /**
         * Estimated distance to the object in meters.
         * Derived from depth map analysis. -1 if unknown.
         */
        public float distance = -1f;

        /**
         * Hazard level for accessibility/navigational warnings.
         * 0 = none, 1 = low, 2 = medium, 3 = high.
         */
        public int hazardLevel = 0;

        public DetectedObject() {}

        public DetectedObject(int id, @NonNull String name, float confidence,
                              @NonNull RectF bbox) {
            this.id = id;
            this.name = name;
            this.confidence = confidence;
            this.bbox = bbox;
        }
    }

    // ================================================================
    // RectF — Simplified bounding box (avoid Android dependency in ML)
    // ================================================================

    /**
     * Lightweight bounding box in normalized coordinates [0, 1].
     * This avoids a hard dependency on android.graphics.RectF in
     * pure-Java ML code, while being easily convertible.
     */
    public static class RectF {
        public float left;
        public float top;
        public float right;
        public float bottom;

        public RectF() {}

        public RectF(float left, float top, float right, float bottom) {
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
        }

        /** Width of the bounding box. */
        public float width() {
            return right - left;
        }

        /** Height of the bounding box. */
        public float height() {
            return bottom - top;
        }

        /** Center X coordinate. */
        public float centerX() {
            return left + width() / 2f;
        }

        /** Center Y coordinate. */
        public float centerY() {
            return top + height() / 2f;
        }

        /** Area of the bounding box. */
        public float area() {
            return width() * height();
        }

        /** Checks if this rect intersects with another. */
        public boolean intersects(@NonNull RectF other) {
            return this.left < other.right
                    && this.right > other.left
                    && this.top < other.bottom
                    && this.bottom > other.top;
        }

        /** Converts to an Android graphics RectF. */
        @NonNull
        public android.graphics.RectF toAndroidRectF() {
            return new android.graphics.RectF(left, top, right, bottom);
        }

        /** Creates from an Android graphics RectF. */
        @NonNull
        public static RectF fromAndroidRectF(@NonNull android.graphics.RectF r) {
            return new RectF(r.left, r.top, r.right, r.bottom);
        }
    }

    // ================================================================
    // SegmentationResult — Per-pixel semantic segmentation
    // ================================================================

    /**
     * Result from the DeepLab v3+ segmentation pipeline.
     * Contains a per-pixel class label mask and the mapping
     * from class IDs to human-readable labels.
     */
    public static class SegmentationResult {
        /**
         * Per-pixel class label mask.
         * Dimensions: [height][width], values are class IDs (0-20).
         */
        @NonNull
        public int[][] mask;

        /** Width of the mask. */
        public int width;

        /** Height of the mask. */
        public int height;

        /**
         * Mapping from class ID to class label string.
         * E.g., 0 → "background", 15 → "person".
         */
        @NonNull
        public Map<Integer, String> classLabels;

        public SegmentationResult(@NonNull int[][] mask, int width, int height,
                                  @NonNull Map<Integer, String> classLabels) {
            this.mask = mask;
            this.width = width;
            this.height = height;
            this.classLabels = classLabels;
        }
    }

    // ================================================================
    // DetectionResult — Bounding box object detection
    // ================================================================

    /**
     * Result from the COCO SSD object detection pipeline.
     * Contains a list of detected objects with bounding boxes
     * and class labels.
     */
    public static class DetectionResult {
        /** List of detected objects. */
        @NonNull
        public List<DetectedObject> objects;

        public DetectionResult(@NonNull List<DetectedObject> objects) {
            this.objects = objects;
        }

        /** Returns the number of detected objects. */
        public int getCount() {
            return objects.size();
        }

        /**
         * Finds the first detected object with the given class name.
         *
         * @param className The class name to search for.
         * @return The first matching object, or null.
         */
        @Nullable
        public DetectedObject findByName(@NonNull String className) {
            for (DetectedObject obj : objects) {
                if (className.equalsIgnoreCase(obj.name)) {
                    return obj;
                }
            }
            return null;
        }
    }

    // ================================================================
    // DepthResult — Monocular depth estimation
    // ================================================================

    /**
     * Result from the MiDaS depth estimation pipeline.
     * Contains a relative inverse depth map where higher values
     * indicate closer objects.
     */
    public static class DepthResult {
        /**
         * Depth map in relative inverse depth values.
         * Dimensions: [height][width].
         * Higher values = closer to camera.
         * Values are not in absolute meters — they require calibration.
         */
        @NonNull
        public float[][] depthMap;

        /** Width of the depth map. */
        public int width;

        /** Height of the depth map. */
        public int height;

        public DepthResult(@NonNull float[][] depthMap, int width, int height) {
            this.depthMap = depthMap;
            this.width = width;
            this.height = height;
        }

        /**
         * Gets the depth value at the given normalized coordinates.
         *
         * @param nx Normalized x [0, 1].
         * @param ny Normalized y [0, 1].
         * @return Depth value, or 0 if out of bounds.
         */
        public float getDepthAt(float nx, float ny) {
            int x = Math.min(Math.max((int) (nx * width), 0), width - 1);
            int y = Math.min(Math.max((int) (ny * height), 0), height - 1);
            return depthMap[y][x];
        }

        /**
         * Estimates approximate distance in meters for a point
         * given a known reference distance and its depth value.
         * This is a simple linear scaling approach.
         *
         * @param depthValue        The depth value at the point of interest.
         * @param referenceDepth    The depth value at the reference point.
         * @param referenceDistanceM The known distance at the reference point in meters.
         * @return Estimated distance in meters, or -1 if reference is invalid.
         */
        public float estimateDistance(float depthValue, float referenceDepth,
                                     float referenceDistanceM) {
            if (referenceDepth <= 0 || referenceDistanceM <= 0) {
                return -1f;
            }
            // MiDaS produces inverse depth, so higher = closer
            // Distance ∝ 1/depth
            return referenceDistanceM * (referenceDepth / depthValue);
        }
    }

    // ================================================================
    // FaceResult — Face detection (no identity)
    // ================================================================

    /**
     * Result from the MediaPipe BlazeFace detection pipeline.
     * IMPORTANT: This contains NO facial recognition or identity data.
     * Faces are only counted and outlined for awareness.
     */
    public static class FaceResult {
        /** Bounding rectangles for detected faces, in normalized coordinates. */
        @NonNull
        public List<RectF> faceRects;

        /** Number of faces detected in this frame. */
        public int faceCount;

        public FaceResult(@NonNull List<RectF> faceRects) {
            this.faceRects = faceRects;
            this.faceCount = faceRects.size();
        }
    }

    // ================================================================
    // OcrResult — Text recognition
    // ================================================================

    /**
     * Result from the ML Kit text recognition pipeline.
     * Contains detected text blocks with their positions.
     */
    public static class OcrResult {
        /** List of detected text blocks. */
        @NonNull
        public List<TextBlock> textBlocks;

        public OcrResult(@NonNull List<TextBlock> textBlocks) {
            this.textBlocks = textBlocks;
        }

        /** Returns the full recognized text as a single string. */
        @NonNull
        public String getFullText() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < textBlocks.size(); i++) {
                if (i > 0) sb.append("\n");
                sb.append(textBlocks.get(i).text);
            }
            return sb.toString();
        }
    }

    /**
     * Represents a single detected text block with position and confidence.
     */
    public static class TextBlock {
        /** Recognized text content. */
        @NonNull
        public String text;

        /** Bounding box in normalized coordinates. */
        @NonNull
        public RectF bbox;

        /** Recognition confidence in [0, 1]. */
        public float confidence;

        public TextBlock(@NonNull String text, @NonNull RectF bbox, float confidence) {
            this.text = text;
            this.bbox = bbox;
            this.confidence = confidence;
        }
    }

    // ================================================================
    // WeatherResult — Sky/weather classification
    // ================================================================

    /**
     * Result from the weather classification pipeline.
     * Describes the current weather condition based on sky analysis.
     */
    public static class WeatherResult {
        /** Classified weather condition string (e.g., "CLEAR", "FOGGY"). */
        @NonNull
        public String condition;

        /** Classification confidence in [0, 1]. */
        public float confidence;

        /** Detected sky features (e.g., "clouds", "sunset_colors"). */
        @NonNull
        public String[] skyFeatures;

        public WeatherResult(@NonNull String condition, float confidence,
                             @NonNull String[] skyFeatures) {
            this.condition = condition;
            this.confidence = confidence;
            this.skyFeatures = skyFeatures;
        }
    }

    // ================================================================
    // PlantResult — Plant/species identification
    // ================================================================

    /**
     * Result from the iNaturalist plant identification pipeline.
     * Contains the identified species and a brief description.
     */
    public static class PlantResult {
        /** Identified species name (e.g., "Rosa rubiginosa"). */
        @NonNull
        public String species;

        /** Identification confidence in [0, 1]. */
        public float confidence;

        /** Brief description of the identified plant. */
        @Nullable
        public String description;

        public PlantResult(@NonNull String species, float confidence,
                           @Nullable String description) {
            this.species = species;
            this.confidence = confidence;
            this.description = description;
        }
    }

    // ================================================================
    // BarcodeResult — Barcode/QR code scanning
    // ================================================================

    /**
     * Result from the ML Kit barcode scanning pipeline.
     */
    public static class BarcodeResult {
        /** Barcode format (e.g., "QR_CODE", "EAN_13", "CODE_128"). */
        @NonNull
        public String format;

        /** Raw decoded value of the barcode. */
        @NonNull
        public String rawValue;

        /** Bounding box in normalized coordinates. */
        @NonNull
        public RectF bbox;

        public BarcodeResult(@NonNull String format, @NonNull String rawValue,
                             @NonNull RectF bbox) {
            this.format = format;
            this.rawValue = rawValue;
            this.bbox = bbox;
        }
    }

    // ================================================================
    // PathResult — Navigational path/wayfinding
    // ================================================================

    /**
     * Result from the path finding pipeline.
     * Contains a series of waypoints from current position to target,
     * along with path cost and clearance status.
     */
    public static class PathResult {
        /** Waypoints as [x, y] pairs in normalized coordinates. */
        @NonNull
        public List<float[]> waypoints;

        /** Total path cost (lower is better/easier). */
        public float cost;

        /** Whether the path is clear of obstacles. */
        public boolean isClear;

        public PathResult(@NonNull List<float[]> waypoints, float cost, boolean isClear) {
            this.waypoints = waypoints;
            this.cost = cost;
            this.isClear = isClear;
        }

        /** Returns the number of waypoints. */
        public int getWaypointCount() {
            return waypoints.size();
        }
    }

    // ================================================================
    // LandmarkResult — Landmark recognition
    // ================================================================

    /**
     * Result from the landmark recognition pipeline.
     * Follows the same structure as PlantResult for consistency.
     */
    public static class LandmarkResult {
        /** Recognized landmark name (e.g., "Eiffel Tower", "Statue of Liberty"). */
        @NonNull
        public String name;

        /** Recognition confidence in [0, 1]. */
        public float confidence;

        /** Brief description of the landmark. */
        @Nullable
        public String description;

        public LandmarkResult(@NonNull String name, float confidence,
                              @Nullable String description) {
            this.name = name;
            this.confidence = confidence;
            this.description = description;
        }
    }

    // ================================================================
    // SceneResult — Combined result for an entire frame
    // ================================================================

    /**
     * Aggregated ML results for a single camera frame.
     * Combines all pipeline outputs with timing metadata.
     * Fields are nullable because not all pipelines run on every frame.
     */
    public static class SceneResult {
        /** Timestamp when this result was created (millis since epoch). */
        public long timestamp;

        /** Frame index (monotonically increasing). */
        public long frameIndex;

        /** Current estimated FPS at the time of this result. */
        public float fps;

        /** Total processing latency for this frame in milliseconds. */
        public float latencyMs;

        // --- Pipeline results (nullable based on schedule) ---

        /** Semantic segmentation result. Available every frame. */
        @Nullable
        public SegmentationResult segmentation;

        /** Object detection result. Available every frame. */
        @Nullable
        public DetectionResult detection;

        /** Depth estimation result. Available every 3rd frame. */
        @Nullable
        public DepthResult depth;

        /** Face detection result. Available every 2nd frame. */
        @Nullable
        public FaceResult face;

        /** OCR result. Available on-demand. */
        @Nullable
        public OcrResult ocr;

        /** Weather classification result. Available periodically. */
        @Nullable
        public WeatherResult weather;

        /** Plant identification result. Available on-demand. */
        @Nullable
        public PlantResult plant;

        /** Barcode scan result. Available on-demand. */
        @Nullable
        public BarcodeResult barcode;

        /** Path finding result. Available on-demand. */
        @Nullable
        public PathResult path;

        /** Landmark recognition result. Available on-demand. */
        @Nullable
        public LandmarkResult landmark;

        /** Default constructor. */
        public SceneResult() {}
    }
}
