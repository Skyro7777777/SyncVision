/**
 * BarcodePipeline.java
 *
 * ML Kit Barcode Scanning (on-device) pipeline for the Sync Vision app.
 * Uses Google's ML Kit BarcodeScanner to detect and decode barcodes
 * and QR codes in camera frames. Supports all common barcode formats.
 *
 * Input: camera frame InputImage
 * Output: BarcodeResult (format, raw value, bounding box)
 *
 * On-demand only (triggered by user tap or auto-detect mode).
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

import com.google.mlkit.vision.barcode.common.Barcode;
import com.google.mlkit.vision.barcode.BarcodeScanning;
import com.google.mlkit.vision.barcode.BarcodeScanner;
import com.google.mlkit.vision.barcode.BarcodeScannerOptions;
import com.google.mlkit.vision.common.InputImage;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * On-demand barcode scanning pipeline using ML Kit.
 * <p>
 * This pipeline is NOT run continuously — it only activates when the
 * user requests barcode scanning or when auto-detect mode is enabled.
 * <p>
 * Supported barcode formats:
 * <ul>
 *   <li>QR Code</li>
 *   <li>Data Matrix</li>
 *   <li>PDF417</li>
 *   <li>AZTEC</li>
 *   <li>Code 128, Code 39, Code 93, Codabar</li>
 *   <li>EAN-13, EAN-8</li>
 *   <li>ITF</li>
 *   <li>UPC-A, UPC-E</li>
 * </ul>
 */
public class BarcodePipeline {

    private static final String TAG = "SV-BarcodePipeline";

    /** Maximum time to wait for barcode scanning result (milliseconds). */
    private static final long SCAN_TIMEOUT_MS = 3000L;

    // -----------------------------------------------------------------
    // ML Kit components
    // -----------------------------------------------------------------

    @Nullable
    private BarcodeScanner barcodeScanner;

    /** Whether the pipeline is initialized and ready. */
    private boolean initialized = false;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new BarcodePipeline.
     * Call initialize() before processing frames.
     */
    public BarcodePipeline() {
    }

    // ================================================================
    // Lifecycle
    // ================================================================

    /**
     * Initializes the pipeline by creating the ML Kit BarcodeScanner.
     * Configured to support all common barcode formats.
     *
     * @return true if initialization succeeded.
     */
    public boolean initialize() {
        if (initialized) {
            Log.w(TAG, "Pipeline already initialized");
            return true;
        }

        try {
            // Configure scanner for all common formats
            BarcodeScannerOptions options = new BarcodeScannerOptions.Builder()
                    .setBarcodeFormats(Barcode.FORMAT_ALL_FORMATS)
                    .build();

            barcodeScanner = BarcodeScanning.getClient(options);
            initialized = true;
            Log.i(TAG, "BarcodePipeline initialized (ML Kit, all formats)");
            return true;

        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize BarcodePipeline", e);
            return false;
        }
    }

    /**
     * Releases pipeline resources.
     */
    public void release() {
        if (barcodeScanner != null) {
            barcodeScanner.close();
            barcodeScanner = null;
        }
        initialized = false;
        Log.i(TAG, "BarcodePipeline released");
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
     * Processes a camera frame for barcode detection.
     * This is a synchronous (blocking) call that waits for ML Kit
     * to complete processing. Use on a background thread.
     *
     * @param bitmap The camera frame Bitmap (ARGB_8888).
     * @return List of BarcodeResult for all detected barcodes, or null on error.
     *         Returns an empty list if no barcodes are found.
     */
    @Nullable
    public List<InferenceResult.BarcodeResult> processFrame(@NonNull Bitmap bitmap) {
        if (!initialized || barcodeScanner == null) {
            Log.w(TAG, "Pipeline not initialized");
            return null;
        }

        try {
            InputImage inputImage = InputImage.fromBitmap(bitmap, 0);

            // Use a latch to make the async ML Kit call synchronous
            final CountDownLatch latch = new CountDownLatch(1);
            final AtomicReference<List<Barcode>> barcodeResult = new AtomicReference<>();
            final AtomicReference<Exception> errorRef = new AtomicReference<>();

            barcodeScanner.process(inputImage)
                    .addOnSuccessListener(barcodes -> {
                        barcodeResult.set(barcodes);
                        latch.countDown();
                    })
                    .addOnFailureListener(e -> {
                        errorRef.set(e);
                        latch.countDown();
                    });

            // Wait for result with timeout
            boolean completed = latch.await(SCAN_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            if (!completed) {
                Log.w(TAG, "Barcode scan timed out after " + SCAN_TIMEOUT_MS + "ms");
                return null;
            }

            // Check for errors
            Exception error = errorRef.get();
            if (error != null) {
                Log.e(TAG, "Barcode scanning error", error);
                return null;
            }

            // Convert results
            List<Barcode> barcodes = barcodeResult.get();
            if (barcodes == null || barcodes.isEmpty()) {
                return new ArrayList<>();
            }

            return convertResults(barcodes, bitmap.getWidth(), bitmap.getHeight());

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.w(TAG, "Barcode scan interrupted");
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
     * Converts ML Kit Barcode results to our BarcodeResult format.
     * Normalizes bounding box coordinates from pixel space to [0, 1].
     *
     * @param barcodes    ML Kit Barcode list.
     * @param frameWidth  Frame width in pixels.
     * @param frameHeight Frame height in pixels.
     * @return List of normalized BarcodeResult objects.
     */
    @NonNull
    private List<InferenceResult.BarcodeResult> convertResults(
            @NonNull List<Barcode> barcodes, int frameWidth, int frameHeight) {
        List<InferenceResult.BarcodeResult> results = new ArrayList<>();

        for (Barcode barcode : barcodes) {
            String format = getFormatName(barcode.getFormat());
            String rawValue = barcode.getRawValue();

            if (rawValue == null || rawValue.isEmpty()) {
                // Try display value as fallback
                rawValue = barcode.getDisplayValue();
                if (rawValue == null || rawValue.isEmpty()) {
                    continue;
                }
            }

            // Get bounding box
            InferenceResult.RectF bbox;
            android.graphics.Rect cornerBox = barcode.getBoundingBox();
            if (cornerBox != null) {
                float left = Math.max(0f, Math.min(1f, (float) cornerBox.left / frameWidth));
                float top = Math.max(0f, Math.min(1f, (float) cornerBox.top / frameHeight));
                float right = Math.max(0f, Math.min(1f, (float) cornerBox.right / frameWidth));
                float bottom = Math.max(0f, Math.min(1f, (float) cornerBox.bottom / frameHeight));
                bbox = new InferenceResult.RectF(left, top, right, bottom);
            } else {
                // No bounding box available — use a default centered rect
                bbox = new InferenceResult.RectF(0.3f, 0.3f, 0.7f, 0.7f);
            }

            results.add(new InferenceResult.BarcodeResult(format, rawValue, bbox));
        }

        return results;
    }

    // ================================================================
    // Utility
    // ================================================================

    /**
     * Converts ML Kit barcode format constant to a human-readable string.
     *
     * @param format ML Kit barcode format constant.
     * @return Format name string.
     */
    @NonNull
    public static String getFormatName(int format) {
        switch (format) {
            case Barcode.FORMAT_QR_CODE:
                return "QR_CODE";
            case Barcode.FORMAT_DATA_MATRIX:
                return "DATA_MATRIX";
            case Barcode.FORMAT_PDF417:
                return "PDF417";
            case Barcode.FORMAT_AZTEC:
                return "AZTEC";
            case Barcode.FORMAT_CODE_128:
                return "CODE_128";
            case Barcode.FORMAT_CODE_39:
                return "CODE_39";
            case Barcode.FORMAT_CODE_93:
                return "CODE_93";
            case Barcode.FORMAT_CODABAR:
                return "CODABAR";
            case Barcode.FORMAT_EAN_13:
                return "EAN_13";
            case Barcode.FORMAT_EAN_8:
                return "EAN_8";
            case Barcode.FORMAT_ITF:
                return "ITF";
            case Barcode.FORMAT_UPC_A:
                return "UPC_A";
            case Barcode.FORMAT_UPC_E:
                return "UPC_E";
            default:
                return "UNKNOWN";
        }
    }
}
