/**
 * DetectionPipeline.java
 *
 * COCO SSD MobileNet V2 object detection pipeline for the Sync Vision app.
 * Takes a camera frame Bitmap, resizes to 300x300 RGB input, runs TFLite
 * inference, and produces bounding boxes with class labels and confidence
 * scores. Includes Non-Maximum Suppression (NMS) to remove duplicate
 * detections.
 *
 * Model: coco_ssd_mobilenet_v2_300.tflite
 * Input: 1x300x300x3 float32 (RGB, normalized [0, 1])
 * Output: bounding boxes, class labels, confidence scores, num detections
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.ml
 * Target SDK: 29+
 */

package com.syncvision.app.ml;

import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.syncvision.app.camera.CameraConfig;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Object detection pipeline using COCO SSD MobileNet V2.
 * <p>
 * Processes every camera frame to detect objects with bounding boxes.
 * Results are filtered by confidence threshold and deduplicated using
 * Non-Maximum Suppression (NMS).
 * <p>
 * The 80 COCO class labels cover common objects like person, car, dog, etc.
 */
public class DetectionPipeline {

    private static final String TAG = "SV-DetectionPipeline";

    /** Model filename in assets/models/. */
    private static final String MODEL_NAME = "coco_ssd_mobilenet_v2_300.tflite";

    /** Number of COCO object classes. */
    private static final int NUM_CLASSES = 80;

    /** Maximum number of detections the model can output. */
    private static final int MAX_OUTPUT_DETECTIONS = 10;

    /** Input image size (square). */
    private static final int INPUT_SIZE = CameraConfig.DETECTION_INPUT_SIZE; // 300

    /**
     * 80 COCO class labels for SSD MobileNet V2.
     * Indexed from 1 (class 0 is reserved/background in some models).
     */
    private static final String[] COCO_CLASSES = {
            "person",            // 1
            "bicycle",           // 2
            "car",               // 3
            "motorcycle",        // 4
            "airplane",          // 5
            "bus",               // 6
            "train",             // 7
            "truck",             // 8
            "boat",              // 9
            "traffic light",     // 10
            "fire hydrant",      // 11
            "stop sign",         // 13
            "parking meter",     // 14
            "bench",             // 15
            "bird",              // 16
            "cat",               // 17
            "dog",               // 18
            "horse",             // 19
            "sheep",             // 20
            "cow",               // 21
            "elephant",          // 22
            "bear",              // 23
            "zebra",             // 24
            "giraffe",           // 25
            "backpack",          // 27
            "umbrella",          // 28
            "handbag",           // 31
            "tie",               // 32
            "suitcase",          // 33
            "frisbee",           // 34
            "skis",              // 35
            "snowboard",         // 36
            "sports ball",       // 37
            "kite",              // 38
            "baseball bat",      // 39
            "baseball glove",    // 40
            "skateboard",        // 41
            "surfboard",         // 42
            "tennis racket",     // 43
            "bottle",            // 44
            "wine glass",        // 46
            "cup",               // 47
            "fork",              // 48
            "knife",             // 49
            "spoon",             // 50
            "bowl",              // 51
            "banana",            // 52
            "apple",             // 53
            "sandwich",          // 54
            "orange",            // 55
            "broccoli",          // 56
            "carrot",            // 57
            "hot dog",           // 58
            "pizza",             // 59
            "donut",             // 60
            "cake",              // 61
            "chair",             // 62
            "couch",             // 63
            "potted plant",      // 64
            "bed",               // 65
            "dining table",      // 67
            "toilet",            // 70
            "tv",                // 72
            "laptop",            // 73
            "mouse",             // 74
            "remote",            // 75
            "keyboard",          // 76
            "cell phone",        // 77
            "microwave",         // 78
            "oven",              // 79
            "toaster",           // 80
            "sink",              // 81
            "refrigerator",      // 82
            "book",              // 84
            "clock",             // 85
            "vase",              // 86
            "scissors",          // 87
            "teddy bear",        // 88
            "hair drier",        // 89
            "toothbrush"         // 90
    };

    // -----------------------------------------------------------------
    // TFLite components
    // -----------------------------------------------------------------

    @Nullable
    private Interpreter interpreter;

    /** Pre-allocated input buffer: 1 x INPUT_SIZE x INPUT_SIZE x 3. */
    private ByteBuffer inputBuffer;

    /** Output: bounding boxes [1][MAX_DETECTIONS][4] (ymin, xmin, ymax, xmax). */
    private float[][][] outputLocations;

    /** Output: class IDs [1][MAX_DETECTIONS]. */
    private float[][] outputClasses;

    /** Output: confidence scores [1][MAX_DETECTIONS]. */
    private float[][] outputScores;

    /** Output: number of detections [1]. */
    private float[] outputNumDetections;

    /** Confidence threshold for filtering detections. */
    private float confidenceThreshold = CameraConfig.DETECTION_CONFIDENCE_THRESHOLD;

    /** IoU threshold for NMS. */
    private float nmsIoUThreshold = CameraConfig.NMS_IOU_THRESHOLD;

    /** Whether the pipeline is initialized and ready. */
    private boolean initialized = false;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new DetectionPipeline.
     * Call initialize() before processing frames.
     */
    public DetectionPipeline() {
    }

    // ================================================================
    // Lifecycle
    // ================================================================

    /**
     * Initializes the pipeline by loading the TFLite model.
     *
     * @return true if initialization succeeded.
     */
    public boolean initialize() {
        if (initialized) {
            Log.w(TAG, "Pipeline already initialized");
            return true;
        }

        try {
            ModelManager modelManager = ModelManager.getInstance();
            interpreter = modelManager.getInterpreter(MODEL_NAME, CameraConfig.NUM_INFERENCE_THREADS);

            // Pre-allocate input buffer
            inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
            inputBuffer.order(ByteOrder.nativeOrder());

            // Pre-allocate output buffers
            outputLocations = new float[1][MAX_OUTPUT_DETECTIONS][4];
            outputClasses = new float[1][MAX_OUTPUT_DETECTIONS];
            outputScores = new float[1][MAX_OUTPUT_DETECTIONS];
            outputNumDetections = new float[1];

            initialized = true;
            Log.i(TAG, "DetectionPipeline initialized (input: " + INPUT_SIZE + "x" + INPUT_SIZE + ")");
            return true;

        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize DetectionPipeline", e);
            return false;
        }
    }

    /**
     * Releases pipeline resources.
     */
    public void release() {
        if (interpreter != null) {
            ModelManager.getInstance().releaseInterpreter(MODEL_NAME);
            interpreter = null;
        }
        inputBuffer = null;
        outputLocations = null;
        outputClasses = null;
        outputScores = null;
        outputNumDetections = null;
        initialized = false;
        Log.i(TAG, "DetectionPipeline released");
    }

    /**
     * Returns whether the pipeline is initialized and ready.
     */
    public boolean isInitialized() {
        return initialized;
    }

    // ================================================================
    // Configuration
    // ================================================================

    /**
     * Sets the confidence threshold for detection filtering.
     *
     * @param threshold Confidence threshold in [0, 1]. Default: 0.5.
     */
    public void setConfidenceThreshold(float threshold) {
        this.confidenceThreshold = Math.max(0f, Math.min(1f, threshold));
    }

    /**
     * Sets the IoU threshold for Non-Maximum Suppression.
     *
     * @param threshold IoU threshold in [0, 1]. Default: 0.45.
     */
    public void setNmsIoUThreshold(float threshold) {
        this.nmsIoUThreshold = Math.max(0f, Math.min(1f, threshold));
    }

    // ================================================================
    // Frame Processing
    // ================================================================

    /**
     * Processes a camera frame through the object detection model.
     *
     * @param bitmap The camera frame Bitmap (ARGB_8888).
     * @return DetectionResult with detected objects, or null on error.
     */
    @Nullable
    public InferenceResult.DetectionResult processFrame(@NonNull Bitmap bitmap) {
        if (!initialized || interpreter == null) {
            Log.w(TAG, "Pipeline not initialized");
            return null;
        }

        try {
            // Step 1: Preprocess input
            preprocessInput(bitmap);

            // Step 2: Run inference
            Object[] inputs = {inputBuffer};

            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, outputLocations);
            outputs.put(1, outputClasses);
            outputs.put(2, outputScores);
            outputs.put(3, outputNumDetections);

            interpreter.runForMultipleInputsOutputs(inputs, outputs);

            // Step 3: Post-process with NMS
            return postprocessOutput();

        } catch (Exception e) {
            Log.e(TAG, "Error processing frame", e);
            return null;
        }
    }

    // ================================================================
    // Pre-processing
    // ================================================================

    /**
     * Preprocesses the input bitmap for the detection model.
     * Resizes to INPUT_SIZE x INPUT_SIZE and normalizes RGB to [0, 1].
     *
     * @param bitmap The original camera frame.
     */
    private void preprocessInput(@NonNull Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        if (resized != bitmap) {
            resized.recycle();
        }

        inputBuffer.rewind();
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;
            inputBuffer.putFloat(r);
            inputBuffer.putFloat(g);
            inputBuffer.putFloat(b);
        }
    }

    // ================================================================
    // Post-processing with NMS
    // ================================================================

    /**
     * Post-processes the model output with confidence filtering and NMS.
     *
     * @return DetectionResult with filtered and deduplicated objects.
     */
    @NonNull
    private InferenceResult.DetectionResult postprocessOutput() {
        int numDetections = (int) outputNumDetections[0];
        numDetections = Math.min(numDetections, MAX_OUTPUT_DETECTIONS);

        // Step 1: Collect detections above confidence threshold
        List<InferenceResult.DetectedObject> candidates = new ArrayList<>();

        for (int i = 0; i < numDetections; i++) {
            float score = outputScores[0][i];
            if (score < confidenceThreshold) {
                continue;
            }

            int classId = (int) outputClasses[0][i];
            String className = getClassName(classId);

            // Bounding box: [ymin, xmin, ymax, xmax] in normalized [0, 1]
            float ymin = outputLocations[0][i][0];
            float xmin = outputLocations[0][i][1];
            float ymax = outputLocations[0][i][2];
            float xmax = outputLocations[0][i][3];

            // Clamp to [0, 1]
            xmin = Math.max(0f, Math.min(1f, xmin));
            ymin = Math.max(0f, Math.min(1f, ymin));
            xmax = Math.max(0f, Math.min(1f, xmax));
            ymax = Math.max(0f, Math.min(1f, ymax));

            InferenceResult.DetectedObject obj = new InferenceResult.DetectedObject();
            obj.id = i;
            obj.name = className;
            obj.confidence = score;
            obj.bbox = new InferenceResult.RectF(xmin, ymin, xmax, ymax);

            candidates.add(obj);
        }

        // Step 2: Apply Non-Maximum Suppression
        List<InferenceResult.DetectedObject> filtered = applyNMS(candidates);

        // Step 3: Limit to max detections
        if (filtered.size() > CameraConfig.MAX_DETECTIONS) {
            filtered = filtered.subList(0, CameraConfig.MAX_DETECTIONS);
        }

        return new InferenceResult.DetectionResult(filtered);
    }

    /**
     * Applies Non-Maximum Suppression to remove overlapping detections
     * of the same class.
     *
     * @param candidates List of candidate detections (already filtered by confidence).
     * @return Filtered list with duplicates removed.
     */
    @NonNull
    private List<InferenceResult.DetectedObject> applyNMS(
            @NonNull List<InferenceResult.DetectedObject> candidates) {

        // Sort by confidence (descending)
        List<InferenceResult.DetectedObject> sorted = new ArrayList<>(candidates);
        sorted.sort((a, b) -> Float.compare(b.confidence, a.confidence));

        List<InferenceResult.DetectedObject> result = new ArrayList<>();
        boolean[] suppressed = new boolean[sorted.size()];

        for (int i = 0; i < sorted.size(); i++) {
            if (suppressed[i]) continue;

            InferenceResult.DetectedObject current = sorted.get(i);
            result.add(current);

            // Suppress overlapping detections of the same class
            for (int j = i + 1; j < sorted.size(); j++) {
                if (suppressed[j]) continue;

                InferenceResult.DetectedObject other = sorted.get(j);

                // Only suppress same-class detections
                if (!current.name.equals(other.name)) continue;

                float iou = computeIoU(current.bbox, other.bbox);
                if (iou > nmsIoUThreshold) {
                    suppressed[j] = true;
                }
            }
        }

        return result;
    }

    /**
     * Computes Intersection over Union (IoU) between two bounding boxes.
     *
     * @param a First bounding box.
     * @param b Second bounding box.
     * @return IoU value in [0, 1].
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
    // Utility
    // ================================================================

    /**
     * Returns the class name for a given COCO class ID.
     *
     * @param classId The class ID (1-indexed in COCO).
     * @return The class name, or "unknown" if out of range.
     */
    @NonNull
    public static String getClassName(int classId) {
        // COCO SSD model uses 1-indexed class IDs
        int index = classId - 1;
        if (index >= 0 && index < COCO_CLASSES.length) {
            return COCO_CLASSES[index];
        }
        return "unknown";
    }

    /**
     * Returns all COCO class labels as an array.
     */
    @NonNull
    public static String[] getCocoClasses() {
        return COCO_CLASSES.clone();
    }
}
