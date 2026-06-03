/**
 * FrameDispatcher.java
 *
 * Central frame dispatch hub for the Sync Vision ML pipeline.
 * Receives ImageProxy frames from CameraManager, converts them to
 * Bitmap/ByteBuffer, and dispatches to the appropriate ML pipelines
 * based on frame counter scheduling. Maintains a ring buffer of
 * latest results and tracks FPS/latency metrics.
 *
 * Sync Vision â€” Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.camera
 * Target SDK: 29+
 */

package com.syncvision.app.camera;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.ImageProxy;

import com.syncvision.app.ml.InferenceResult;
import com.syncvision.app.ml.SegmentationPipeline;
import com.syncvision.app.ml.DetectionPipeline;
import com.syncvision.app.ml.DepthPipeline;
import com.syncvision.app.ml.FacePipeline;
import com.syncvision.app.ml.OcrPipeline;
import com.syncvision.app.ml.PlantPipeline;
import com.syncvision.app.ml.BarcodePipeline;
import com.syncvision.app.ml.WeatherPipeline;
import com.syncvision.app.ml.LandmarkPipeline;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Dispatches camera frames to ML pipelines based on a frame-counter schedule.
 * <p>
 * Scheduling strategy:
 * <ul>
 *   <li>Every frame: SegmentationPipeline + DetectionPipeline</li>
 *   <li>Every 3rd frame: DepthPipeline</li>
 *   <li>Every 2nd frame: FacePipeline</li>
 *   <li>On-demand: OcrPipeline, PlantPipeline, BarcodePipeline, LandmarkPipeline</li>
 *   <li>Periodic: WeatherPipeline (every ~2.5 seconds)</li>
 * </ul>
 * <p>
 * Results are stored in a ring buffer and delivered via OnFrameProcessedListener.
 */
public class FrameDispatcher {

    private static final String TAG = "SV-FrameDispatcher";

    // -----------------------------------------------------------------
    // Pipelines
    // -----------------------------------------------------------------

    @Nullable
    private SegmentationPipeline segmentationPipeline;

    @Nullable
    private DetectionPipeline detectionPipeline;

    @Nullable
    private DepthPipeline depthPipeline;

    @Nullable
    private FacePipeline facePipeline;

    @Nullable
    private OcrPipeline ocrPipeline;

    @Nullable
    private PlantPipeline plantPipeline;

    @Nullable
    private BarcodePipeline barcodePipeline;

    @Nullable
    private WeatherPipeline weatherPipeline;

    @Nullable
    private LandmarkPipeline landmarkPipeline;

    // -----------------------------------------------------------------
    // State
    // -----------------------------------------------------------------

    /** Monotonically increasing frame counter. */
    private final AtomicLong frameCounter = new AtomicLong(0);

    /** Timestamp of the last weather pipeline run. */
    private volatile long lastWeatherRunTimeMs = 0;

    /** Background executor for ML pipeline dispatch. */
    private final ExecutorService pipelineExecutor;

    /** Ring buffer for latest SceneResult objects. */
    private final InferenceResult.SceneResult[] resultRingBuffer;

    /** Current write index into the ring buffer. */
    private int ringBufferIndex = 0;

    /** Whether the dispatcher is active. */
    private volatile boolean active = true;

    // -----------------------------------------------------------------
    // FPS & Latency Tracking
    // -----------------------------------------------------------------

    private final AtomicLong framesProcessed = new AtomicLong(0);
    private volatile long fpsTrackingStartTimeMs = 0;
    private volatile float currentFps = 0f;
    private volatile long lastFrameTimestampNs = 0;
    private volatile float avgLatencyMs = 0f;

    // -----------------------------------------------------------------
    // On-demand pipeline triggers
    // -----------------------------------------------------------------

    private volatile boolean ocrRequested = false;
    private volatile boolean plantRequested = false;
    private volatile boolean barcodeRequested = false;
    private volatile boolean landmarkRequested = false;

    // -----------------------------------------------------------------
    // Listener
    // -----------------------------------------------------------------

    /**
     * Callback interface invoked when a frame has been fully processed
     * by all scheduled pipelines.
     */
    public interface OnFrameProcessedListener {
        /**
         * Called with the combined scene result after pipeline processing.
         *
         * @param result The aggregated ML results for this frame.
         */
        void onFrameProcessed(@NonNull InferenceResult.SceneResult result);
    }

    @Nullable
    private OnFrameProcessedListener listener;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new FrameDispatcher.
     * Initializes the pipeline executor and ring buffer.
     */
    public FrameDispatcher() {
        this.pipelineExecutor = Executors.newFixedThreadPool(2, r -> {
            Thread t = new Thread(r, "SV-PipelineWorker");
            t.setPriority(Thread.NORM_PRIORITY + 2);
            return t;
        });

        this.resultRingBuffer = new InferenceResult.SceneResult[CameraConfig.RESULT_RING_BUFFER_SIZE];
        this.fpsTrackingStartTimeMs = System.currentTimeMillis();
    }

    // ================================================================
    // Pipeline Registration
    // ================================================================

    public void setSegmentationPipeline(@Nullable SegmentationPipeline pipeline) {
        this.segmentationPipeline = pipeline;
    }

    public void setDetectionPipeline(@Nullable DetectionPipeline pipeline) {
        this.detectionPipeline = pipeline;
    }

    public void setDepthPipeline(@Nullable DepthPipeline pipeline) {
        this.depthPipeline = pipeline;
    }

    public void setFacePipeline(@Nullable FacePipeline pipeline) {
        this.facePipeline = pipeline;
    }

    public void setOcrPipeline(@Nullable OcrPipeline pipeline) {
        this.ocrPipeline = pipeline;
    }

    public void setPlantPipeline(@Nullable PlantPipeline pipeline) {
        this.plantPipeline = pipeline;
    }

    public void setBarcodePipeline(@Nullable BarcodePipeline pipeline) {
        this.barcodePipeline = pipeline;
    }

    public void setWeatherPipeline(@Nullable WeatherPipeline pipeline) {
        this.weatherPipeline = pipeline;
    }

    public void setLandmarkPipeline(@Nullable LandmarkPipeline pipeline) {
        this.landmarkPipeline = pipeline;
    }

    // ================================================================
    // Listener
    // ================================================================

    /**
     * Sets the listener for processed frame results.
     *
     * @param listener The listener, or null to remove.
     */
    public void setOnFrameProcessedListener(@Nullable OnFrameProcessedListener listener) {
        this.listener = listener;
    }

    // ================================================================
    // On-Demand Pipeline Triggers
    // ================================================================

    /** Request OCR on the next available frame. */
    public void requestOcr() {
        ocrRequested = true;
    }

    /** Request plant identification on the next available frame. */
    public void requestPlantIdentification() {
        plantRequested = true;
    }

    /** Request barcode scanning on the next available frame. */
    public void requestBarcodeScan() {
        barcodeRequested = true;
    }

    /** Request landmark recognition on the next available frame. */
    public void requestLandmarkRecognition() {
        landmarkRequested = true;
    }

    // ================================================================
    // Core Dispatch
    // ================================================================

    /**
     * Called by CameraManager for each new camera frame.
     * Converts the ImageProxy to Bitmap and dispatches to pipelines
     * based on the current frame counter.
     *
     * @param imageProxy The camera frame. The caller is responsible for closing it.
     */
    public void dispatchFrame(@NonNull ImageProxy imageProxy) {
        if (!active) {
            return;
        }

        final long frameIndex = frameCounter.getAndIncrement();
        final long startTimeNs = System.nanoTime();

        // Convert ImageProxy to Bitmap
        Bitmap frameBitmap = imageProxyToBitmap(imageProxy);
        if (frameBitmap == null) {
            Log.w(TAG, "Failed to convert frame " + frameIndex + " to Bitmap");
            return;
        }

        // Store the bitmap reference for on-demand pipelines
        final Bitmap finalBitmap = frameBitmap;

        // Dispatch to pipelines on background thread
        pipelineExecutor.execute(() -> processFrame(finalBitmap, frameIndex, startTimeNs));
    }

    /**
     * Processes a single frame through all scheduled pipelines.
     * Runs on a background thread.
     */
    private void processFrame(@NonNull Bitmap bitmap, long frameIndex, long startTimeNs) {
        try {
            // --- Build the combined SceneResult ---
            InferenceResult.SceneResult sceneResult = new InferenceResult.SceneResult();
            sceneResult.timestamp = System.currentTimeMillis();
            sceneResult.frameIndex = frameIndex;

            // --- Every frame: Segmentation ---
            if (segmentationPipeline != null) {
                try {
                    sceneResult.segmentation = segmentationPipeline.processFrame(bitmap);
                } catch (Exception e) {
                    Log.e(TAG, "Segmentation pipeline error", e);
                }
            }

            // --- Every frame: Detection ---
            if (detectionPipeline != null) {
                try {
                    sceneResult.detection = detectionPipeline.processFrame(bitmap);
                } catch (Exception e) {
                    Log.e(TAG, "Detection pipeline error", e);
                }
            }

            // --- Every 3rd frame: Depth ---
            if (depthPipeline != null && frameIndex % CameraConfig.DEPTH_FRAME_SKIP == 0) {
                try {
                    sceneResult.depth = depthPipeline.processFrame(bitmap);
                } catch (Exception e) {
                    Log.e(TAG, "Depth pipeline error", e);
                }
            }

            // --- Every 2nd frame: Face ---
            if (facePipeline != null && frameIndex % CameraConfig.FACE_FRAME_SKIP == 0) {
                try {
                    sceneResult.face = facePipeline.processFrame(bitmap);
                } catch (Exception e) {
                    Log.e(TAG, "Face pipeline error", e);
                }
            }

            // --- Periodic: Weather (every ~2.5 seconds) ---
            if (weatherPipeline != null) {
                long now = System.currentTimeMillis();
                if (now - lastWeatherRunTimeMs > CameraConfig.WEATHER_INTERVAL_SECONDS * 1000) {
                    try {
                        sceneResult.weather = weatherPipeline.processFrame(bitmap);
                        lastWeatherRunTimeMs = now;
                    } catch (Exception e) {
                        Log.e(TAG, "Weather pipeline error", e);
                    }
                }
            }

            // --- On-demand: OCR ---
            if (ocrPipeline != null && ocrRequested) {
                try {
                    sceneResult.ocr = ocrPipeline.processFrame(bitmap);
                } catch (Exception e) {
                    Log.e(TAG, "OCR pipeline error", e);
                }
                ocrRequested = false;
            }

            // --- On-demand: Plant ---
            if (plantPipeline != null && plantRequested) {
                try {
                    sceneResult.plant = plantPipeline.processFrame(bitmap);
                } catch (Exception e) {
                    Log.e(TAG, "Plant pipeline error", e);
                }
                plantRequested = false;
            }

            // --- On-demand: Barcode ---
            if (barcodePipeline != null && barcodeRequested) {
                try {
                    List<InferenceResult.BarcodeResult> barcodes = barcodePipeline.processFrame(bitmap);
                    sceneResult.barcode = (barcodes != null && !barcodes.isEmpty())
                            ? barcodes.get(0) : null;
                } catch (Exception e) {
                    Log.e(TAG, "Barcode pipeline error", e);
                }
                barcodeRequested = false;
            }

            // --- On-demand: Landmark ---
            if (landmarkPipeline != null && landmarkRequested) {
                try {
                    sceneResult.landmark = landmarkPipeline.processFrame(bitmap);
                } catch (Exception e) {
                    Log.e(TAG, "Landmark pipeline error", e);
                }
                landmarkRequested = false;
            }

            // --- Compute FPS and latency ---
            long endTimeNs = System.nanoTime();
            float latencyMs = (endTimeNs - startTimeNs) / 1_000_000f;
            updateMetrics(latencyMs);
            sceneResult.fps = currentFps;
            sceneResult.latencyMs = latencyMs;

            // --- Store in ring buffer ---
            synchronized (resultRingBuffer) {
                resultRingBuffer[ringBufferIndex] = sceneResult;
                ringBufferIndex = (ringBufferIndex + 1) % resultRingBuffer.length;
            }

            // --- Notify listener ---
            if (listener != null) {
                listener.onFrameProcessed(sceneResult);
            }

        } catch (Exception e) {
            Log.e(TAG, "Unexpected error processing frame " + frameIndex, e);
        }
    }

    // ================================================================
    // Image Conversion
    // ================================================================

    /**
     * Converts an ImageProxy (YUV_420_888 or RGBA_8888) to a Bitmap.
     * Handles rotation based on the image proxy's rotation info.
     *
     * @param imageProxy The camera frame.
     * @return A Bitmap in ARGB_8888 format, or null on failure.
     */
    @Nullable
    private Bitmap imageProxyToBitmap(@NonNull ImageProxy imageProxy) {
        try {
            Bitmap bitmap;

            // Check the image format
            if (imageProxy.getFormat() == ImageFormat.YUV_420_888) {
                bitmap = yuvImageProxyToBitmap(imageProxy);
            } else {
                // RGBA_8888 or other formats â€” use direct ByteBuffer approach
                bitmap = rgbaImageProxyToBitmap(imageProxy);
            }

            if (bitmap == null) {
                return null;
            }

            // Apply rotation if needed
            int rotation = imageProxy.getImageInfo().getRotationDegrees();
            if (rotation != 0) {
                Matrix matrix = new Matrix();
                matrix.postRotate(rotation);
                Bitmap rotated = Bitmap.createBitmap(
                        bitmap, 0, 0,
                        bitmap.getWidth(), bitmap.getHeight(),
                        matrix, true
                );
                if (rotated != bitmap) {
                    bitmap.recycle();
                }
                bitmap = rotated;
            }

            return bitmap;

        } catch (Exception e) {
            Log.e(TAG, "Error converting ImageProxy to Bitmap", e);
            return null;
        }
    }

    /**
     * Converts a YUV_420_888 ImageProxy to Bitmap using YuvImage.
     */
    @Nullable
    private Bitmap yuvImageProxyToBitmap(@NonNull ImageProxy imageProxy) {
        Image image = imageProxy.getImage();
        if (image == null) {
            return null;
        }

        // Convert YUV_420_888 to NV21 byte array
        byte[] nv21 = yuv420ToNv21(image);

        // Decode NV21 to Bitmap via YuvImage
        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21,
                image.getWidth(), image.getHeight(), null);

        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, image.getWidth(), image.getHeight()), 95, outputStream);
        byte[] jpegBytes = outputStream.toByteArray();

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;

        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.length, options);
    }

    /**
     * Converts YUV_420_888 Image planes to NV21 byte array.
     */
    private byte[] yuv420ToNv21(@NonNull Image image) {
        Image.Plane[] planes = image.getPlanes();
        int ySize = planes[0].getRowStride() * image.getHeight();
        int uvSize = planes[1].getRowStride() * image.getHeight() / 2;

        // Conservative buffer size
        byte[] nv21 = new byte[ySize + uvSize];

        // Y plane
        ByteBuffer yBuffer = planes[0].getBuffer();
        yBuffer.position(0);
        int yRowStride = planes[0].getRowStride();
        int yPixelStride = planes[0].getPixelStride();

        // U and V planes
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();
        int uvRowStride = planes[1].getRowStride();
        int uvPixelStride = planes[1].getPixelStride();

        // Copy Y data
        int pos = 0;
        if (yRowStride == image.getWidth() && yPixelStride == 1) {
            // Contiguous Y data
            yBuffer.get(nv21, 0, yRowStride * image.getHeight());
            pos = yRowStride * image.getHeight();
        } else {
            for (int row = 0; row < image.getHeight(); row++) {
                for (int col = 0; col < image.getWidth(); col++) {
                    nv21[pos++] = yBuffer.get(row * yRowStride + col * yPixelStride);
                }
            }
        }

        // Interleave V and U for NV21 (VU ordering)
        for (int row = 0; row < image.getHeight() / 2; row++) {
            for (int col = 0; col < image.getWidth() / 2; col++) {
                int uvIndex = row * uvRowStride + col * uvPixelStride;
                nv21[pos++] = vBuffer.get(uvIndex);  // V
                nv21[pos++] = uBuffer.get(uvIndex);  // U
            }
        }

        return nv21;
    }

    /**
     * Converts an RGBA_8888 ImageProxy to Bitmap using direct buffer copy.
     */
    @Nullable
    private Bitmap rgbaImageProxyToBitmap(@NonNull ImageProxy imageProxy) {
        Image image = imageProxy.getImage();
        if (image == null) {
            return null;
        }

        Image.Plane plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();

        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();

        int width = imageProxy.getWidth();
        int height = imageProxy.getHeight();

        // If the buffer is tightly packed, we can create the Bitmap directly
        if (rowStride == width * pixelStride) {
            buffer.rewind();
            int remaining = buffer.remaining();
            byte[] data = new byte[remaining];
            buffer.get(data);

            // Wrap as int buffer for ARGB
            int[] pixels = new int[width * height];
            ByteBuffer wrapped = ByteBuffer.wrap(data);

            for (int i = 0; i < pixels.length; i++) {
                int r = wrapped.get() & 0xFF;
                int g = wrapped.get() & 0xFF;
                int b = wrapped.get() & 0xFF;
                int a = (pixelStride == 4) ? (wrapped.get() & 0xFF) : 0xFF;
                pixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
            }

            return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);
        } else {
            // Handle row padding
            Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            buffer.rewind();

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int index = y * rowStride + x * pixelStride;
                    int r = buffer.get(index) & 0xFF;
                    int g = buffer.get(index + 1) & 0xFF;
                    int b = buffer.get(index + 2) & 0xFF;
                    int a = (pixelStride == 4) ? (buffer.get(index + 3) & 0xFF) : 0xFF;
                    bitmap.setPixel(x, y, (a << 24) | (r << 16) | (g << 8) | b);
                }
            }

            return bitmap;
        }
    }

    // ================================================================
    // Metrics
    // ================================================================

    /**
     * Updates FPS and latency metrics after a frame is processed.
     */
    private void updateMetrics(float latencyMs) {
        framesProcessed.incrementAndGet();

        // Exponential moving average for latency
        if (avgLatencyMs == 0) {
            avgLatencyMs = latencyMs;
        } else {
            avgLatencyMs = avgLatencyMs * 0.9f + latencyMs * 0.1f;
        }

        // Compute FPS every second
        long now = System.currentTimeMillis();
        long elapsed = now - fpsTrackingStartTimeMs;
        if (elapsed >= 1000) {
            currentFps = (framesProcessed.get() * 1000f) / elapsed;
            framesProcessed.set(0);
            fpsTrackingStartTimeMs = now;
        }
    }

    // ================================================================
    // Ring Buffer Access
    // ================================================================

    /**
     * Returns the most recent SceneResult from the ring buffer.
     *
     * @return The latest scene result, or null if none available.
     */
    @Nullable
    public InferenceResult.SceneResult getLatestResult() {
        synchronized (resultRingBuffer) {
            int index = (ringBufferIndex - 1 + resultRingBuffer.length) % resultRingBuffer.length;
            return resultRingBuffer[index];
        }
    }

    /**
     * Returns all results currently in the ring buffer, ordered from
     * oldest to newest.
     *
     * @return Array of scene results (may contain nulls if buffer not full).
     */
    @NonNull
    public InferenceResult.SceneResult[] getAllResults() {
        synchronized (resultRingBuffer) {
            InferenceResult.SceneResult[] results =
                    new InferenceResult.SceneResult[resultRingBuffer.length];
            for (int i = 0; i < resultRingBuffer.length; i++) {
                int idx = (ringBufferIndex + i) % resultRingBuffer.length;
                results[i] = resultRingBuffer[idx];
            }
            return results;
        }
    }

    // ================================================================
    // Metrics Access
    // ================================================================

    /** Returns the current estimated FPS. */
    public float getCurrentFps() {
        return currentFps;
    }

    /** Returns the average frame processing latency in milliseconds. */
    public float getAvgLatencyMs() {
        return avgLatencyMs;
    }

    /** Returns the total number of frames dispatched. */
    public long getTotalFrameCount() {
        return frameCounter.get();
    }

    // ================================================================
    // Lifecycle
    // ================================================================

    /**
     * Pauses frame dispatching. Pipeline processing is suspended but
     * the executor remains alive.
     */
    public void pause() {
        active = false;
        Log.i(TAG, "FrameDispatcher paused");
    }

    /**
     * Resumes frame dispatching after a pause.
     */
    public void resume() {
        active = true;
        fpsTrackingStartTimeMs = System.currentTimeMillis();
        framesProcessed.set(0);
        Log.i(TAG, "FrameDispatcher resumed");
    }

    /**
     * Shuts down the dispatcher and releases resources.
     * Call this in Activity.onDestroy().
     */
    public void shutdown() {
        active = false;
        pipelineExecutor.shutdownNow();
        Log.i(TAG, "FrameDispatcher shut down");
    }
}
