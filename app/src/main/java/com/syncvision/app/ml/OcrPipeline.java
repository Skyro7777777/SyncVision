/**
 * OcrPipeline.java
 *
 * ML Kit Text Recognition V2 (on-device) pipeline for the Sync Vision app.
 * Uses Google's ML Kit TextRecognizer to detect and recognize text in
 * camera frames. Operates on-demand only (not continuous) to save
 * resources — triggered when the user taps the OCR button.
 *
 * Input: camera frame InputImage
 * Output: OcrResult (text blocks with bounding boxes and confidence)
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

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * On-demand OCR pipeline using ML Kit Text Recognition V2.
 * <p>
 * This pipeline is NOT run continuously — it only activates when the
 * user explicitly requests text recognition (e.g., by tapping the
 * OCR button or pointing at text for a sustained period).
 * <p>
 * ML Kit's on-device text recognizer supports Latin script by default
 * and works without network connectivity.
 */
public class OcrPipeline {

    private static final String TAG = "SV-OcrPipeline";

    /** Maximum time to wait for OCR result (milliseconds). */
    private static final long OCR_TIMEOUT_MS = 3000L;

    // -----------------------------------------------------------------
    // ML Kit components
    // -----------------------------------------------------------------

    @Nullable
    private TextRecognizer textRecognizer;

    /** Whether the pipeline is initialized and ready. */
    private boolean initialized = false;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new OcrPipeline.
     * Call initialize() before processing frames.
     */
    public OcrPipeline() {
    }

    // ================================================================
    // Lifecycle
    // ================================================================

    /**
     * Initializes the pipeline by creating the ML Kit TextRecognizer.
     *
     * @return true if initialization succeeded.
     */
    public boolean initialize() {
        if (initialized) {
            Log.w(TAG, "Pipeline already initialized");
            return true;
        }

        try {
            // Use Latin script recognizer (on-device, no downloads needed)
            textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
            initialized = true;
            Log.i(TAG, "OcrPipeline initialized (ML Kit Latin)");
            return true;

        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize OcrPipeline", e);
            return false;
        }
    }

    /**
     * Releases pipeline resources.
     */
    public void release() {
        if (textRecognizer != null) {
            textRecognizer.close();
            textRecognizer = null;
        }
        initialized = false;
        Log.i(TAG, "OcrPipeline released");
    }

    /**
     * Returns whether the pipeline is initialized and ready.
     */
    public boolean isInitialized() {
        return initialized;
    }

    // ================================================================
    // Frame Processing
    // ================================================================

    /**
     * Processes a camera frame for text recognition.
     * This is a synchronous (blocking) call that waits for ML Kit
     * to complete processing. Use on a background thread.
     *
     * @param bitmap The camera frame Bitmap (ARGB_8888).
     * @return OcrResult with detected text blocks, or null on error.
     */
    @Nullable
    public InferenceResult.OcrResult processFrame(@NonNull Bitmap bitmap) {
        if (!initialized || textRecognizer == null) {
            Log.w(TAG, "Pipeline not initialized");
            return null;
        }

        try {
            // Create InputImage from Bitmap
            InputImage inputImage = InputImage.fromBitmap(bitmap, 0);

            // Use a latch to make the async ML Kit call synchronous
            final CountDownLatch latch = new CountDownLatch(1);
            final AtomicReference<Text> textResult = new AtomicReference<>();
            final AtomicReference<Exception> errorRef = new AtomicReference<>();

            textRecognizer.process(inputImage)
                    .addOnSuccessListener(text -> {
                        textResult.set(text);
                        latch.countDown();
                    })
                    .addOnFailureListener(e -> {
                        errorRef.set(e);
                        latch.countDown();
                    });

            // Wait for result with timeout
            boolean completed = latch.await(OCR_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            if (!completed) {
                Log.w(TAG, "OCR timed out after " + OCR_TIMEOUT_MS + "ms");
                return null;
            }

            // Check for errors
            Exception error = errorRef.get();
            if (error != null) {
                Log.e(TAG, "OCR processing error", error);
                return null;
            }

            // Convert result
            Text text = textResult.get();
            if (text == null) {
                return null;
            }

            return convertResult(text, bitmap.getWidth(), bitmap.getHeight());

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.w(TAG, "OCR interrupted");
            return null;
        } catch (Exception e) {
            Log.e(TAG, "Error processing frame", e);
            return null;
        }
    }

    // ================================================================
    // Result Conversion
    // ================================================================

    /**
     * Converts ML Kit Text result to our OcrResult format.
     * Normalizes bounding box coordinates from pixel space to [0, 1].
     *
     * @param text       ML Kit Text result.
     * @param frameWidth  Frame width in pixels.
     * @param frameHeight Frame height in pixels.
     * @return Normalized OcrResult.
     */
    @NonNull
    private InferenceResult.OcrResult convertResult(@NonNull Text text,
                                                     int frameWidth, int frameHeight) {
        List<InferenceResult.TextBlock> textBlocks = new ArrayList<>();

        for (Text.TextBlock block : text.getTextBlocks()) {
            String blockText = block.getText();
            if (blockText == null || blockText.trim().isEmpty()) {
                continue;
            }

            // Get bounding box
            android.graphics.Rect bbox = block.getBoundingBox();
            if (bbox == null) continue;

            // Normalize to [0, 1]
            float left = (float) bbox.left / frameWidth;
            float top = (float) bbox.top / frameHeight;
            float right = (float) bbox.right / frameWidth;
            float bottom = (float) bbox.bottom / frameHeight;

            // Clamp
            left = Math.max(0f, Math.min(1f, left));
            top = Math.max(0f, Math.min(1f, top));
            right = Math.max(0f, Math.min(1f, right));
            bottom = Math.max(0f, Math.min(1f, bottom));

            // Confidence: ML Kit doesn't always provide per-block confidence.
            // Use a heuristic based on corner points availability.
            float confidence = (block.getCornerPoints() != null) ? 0.9f : 0.5f;

            textBlocks.add(new InferenceResult.TextBlock(
                    blockText.trim(),
                    new InferenceResult.RectF(left, top, right, bottom),
                    confidence
            ));
        }

        return new InferenceResult.OcrResult(textBlocks);
    }
}
