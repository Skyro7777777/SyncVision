/**
 * PerformanceMonitor.java
 *
 * Thread-safe performance monitoring utility for the Sync Vision app.
 * Tracks FPS, frame timing, and ML inference latency across multiple
 * pipelines. Uses a circular buffer of the last 60 frames for
 * responsive statistics without memory growth.
 *
 * Usage:
 *   PerformanceMonitor monitor = PerformanceMonitor.getInstance();
 *
 *   // In the render loop:
 *   monitor.recordFrameTime(System.currentTimeMillis());
 *
 *   // After ML inference:
 *   monitor.recordInferenceTime("SegmentationPipeline", 28);
 *
 *   // For HUD display:
 *   String stats = monitor.getStats();
 *
 * The monitor is designed for real-time HUD overlay rendering with
 * minimal overhead. All methods are thread-safe and lock-free where
 * possible (using volatile reads) for the hot path (recordFrameTime).
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.util
 * Target SDK: 29+
 */

package com.syncvision.app.util;

import android.util.Log;

import androidx.annotation.NonNull;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Thread-safe performance monitor tracking FPS, frame times, and ML
 * inference latencies. Uses circular buffers for bounded memory usage
 * and provides formatted statistics for the HUD overlay.
 * <p>
 * Design goals:
 *   - Zero allocation on the hot path (recordFrameTime)
 *   - Bounded memory via fixed-size circular buffers
 *   - Thread-safe for concurrent access from render + ML threads
 *   - Formatted output suitable for terminal-style HUD rendering
 */
public class PerformanceMonitor {

    private static final String TAG = "SV-PerfMonitor";

    // ================================================================
    // Constants
    // ================================================================

    /** Number of frames to track in the circular buffer. */
    private static final int BUFFER_SIZE = 60;

    /** Number of inference samples to track per pipeline. */
    private static final int INFERENCE_BUFFER_SIZE = 30;

    // ================================================================
    // Singleton
    // ================================================================

    private static volatile PerformanceMonitor sInstance;

    /**
     * Returns the singleton PerformanceMonitor instance.
     *
     * @return The PerformanceMonitor singleton.
     */
    @NonNull
    public static PerformanceMonitor getInstance() {
        if (sInstance == null) {
            synchronized (PerformanceMonitor.class) {
                if (sInstance == null) {
                    sInstance = new PerformanceMonitor();
                }
            }
        }
        return sInstance;
    }

    // ================================================================
    // Frame Time Tracking (Circular Buffer)
    // ================================================================

    /** Circular buffer of frame timestamps in milliseconds. */
    private final long[] mFrameTimestamps = new long[BUFFER_SIZE];

    /** Current write index into the frame timestamp buffer. */
    private int mFrameIndex = 0;

    /** Number of frames recorded (for warm-up detection). */
    private int mFrameCount = 0;

    /** Circular buffer of frame durations in milliseconds. */
    private final float[] mFrameTimes = new float[BUFFER_SIZE];

    /** Current write index into the frame time buffer. */
    private int mFrameTimeIndex = 0;

    // ================================================================
    // Inference Time Tracking (Per-Pipeline)
    // ================================================================

    /**
     * Map of pipeline name → circular buffer of inference durations.
     * Uses ConcurrentHashMap for thread-safe concurrent access.
     */
    private final ConcurrentHashMap<String, InferenceTracker> mInferenceTrackers
            = new ConcurrentHashMap<>();

    // ================================================================
    // Constructor
    // ================================================================

    /** Private constructor for singleton. */
    private PerformanceMonitor() {
        Log.d(TAG, "PerformanceMonitor initialized (buffer=" + BUFFER_SIZE
                + ", inferenceBuffer=" + INFERENCE_BUFFER_SIZE + ")");
    }

    // ================================================================
    // Frame Recording
    // ================================================================

    /**
     * Records a frame timestamp for FPS and frame time calculation.
     * Should be called once per rendered frame, typically from the
     * GLSurfaceView.Renderer.onDrawFrame() callback or equivalent.
     * <p>
     * This method is lightweight and suitable for the render hot path.
     *
     * @param timestampMs Current timestamp in milliseconds
     *                    (e.g., System.currentTimeMillis()).
     */
    public void recordFrameTime(long timestampMs) {
        synchronized (mFrameTimestamps) {
            // Compute frame time from consecutive timestamps
            if (mFrameCount > 0) {
                int prevIndex = (mFrameIndex - 1 + BUFFER_SIZE) % BUFFER_SIZE;
                float delta = timestampMs - mFrameTimestamps[prevIndex];
                // Sanity check: ignore huge deltas (e.g., app was paused)
                if (delta >= 0 && delta < 1000) {
                    mFrameTimes[mFrameTimeIndex] = delta;
                    mFrameTimeIndex = (mFrameTimeIndex + 1) % BUFFER_SIZE;
                }
            }

            mFrameTimestamps[mFrameIndex] = timestampMs;
            mFrameIndex = (mFrameIndex + 1) % BUFFER_SIZE;
            mFrameCount++;
        }
    }

    // ================================================================
    // Inference Recording
    // ================================================================

    /**
     * Records an ML inference duration for a specific pipeline.
     * Should be called after each ML inference call completes.
     *
     * @param pipelineName Name of the ML pipeline
     *                     (e.g., "SegmentationPipeline").
     * @param durationMs   Inference duration in milliseconds.
     */
    public void recordInferenceTime(@NonNull String pipelineName, long durationMs) {
        InferenceTracker tracker = mInferenceTrackers.get(pipelineName);
        if (tracker == null) {
            tracker = new InferenceTracker(INFERENCE_BUFFER_SIZE);
            InferenceTracker existing = mInferenceTrackers.putIfAbsent(pipelineName, tracker);
            if (existing != null) {
                tracker = existing;
            }
        }
        tracker.record(durationMs);
    }

    // ================================================================
    // Statistics Queries
    // ================================================================

    /**
     * Returns the current frames per second based on the last
     * BUFFER_SIZE frames. Returns 0 if insufficient data.
     *
     * @return Current FPS estimate.
     */
    public float getFps() {
        synchronized (mFrameTimestamps) {
            if (mFrameCount < 2) {
                return 0f;
            }

            // Find the time span of the buffer
            int count = Math.min(mFrameCount, BUFFER_SIZE);
            if (count < 2) {
                return 0f;
            }

            // Get oldest and newest timestamps
            int oldestIndex;
            if (mFrameCount <= BUFFER_SIZE) {
                oldestIndex = 0;
            } else {
                oldestIndex = mFrameIndex; // Next write position = oldest
            }

            int newestIndex = (mFrameIndex - 1 + BUFFER_SIZE) % BUFFER_SIZE;
            long oldest = mFrameTimestamps[oldestIndex];
            long newest = mFrameTimestamps[newestIndex];

            if (newest <= oldest) {
                return 0f;
            }

            float elapsedSec = (newest - oldest) / 1000f;
            return (count - 1) / elapsedSec;
        }
    }

    /**
     * Returns the average frame time in milliseconds over the last
     * BUFFER_SIZE frames. Returns 0 if insufficient data.
     *
     * @return Average frame time in ms.
     */
    public float getAverageFrameTime() {
        synchronized (mFrameTimestamps) {
            int count = Math.min(mFrameCount - 1, BUFFER_SIZE);
            if (count <= 0) {
                return 0f;
            }

            float sum = 0f;
            for (int i = 0; i < count; i++) {
                int idx = (mFrameTimeIndex - count + i + BUFFER_SIZE) % BUFFER_SIZE;
                sum += mFrameTimes[idx];
            }

            return sum / count;
        }
    }

    /**
     * Returns the average inference time in milliseconds for a
     * specific pipeline over the last INFERENCE_BUFFER_SIZE samples.
     *
     * @param pipelineName Name of the ML pipeline.
     * @return Average inference time in ms, or 0 if no data.
     */
    public float getInferenceTime(@NonNull String pipelineName) {
        InferenceTracker tracker = mInferenceTrackers.get(pipelineName);
        if (tracker == null) {
            return 0f;
        }
        return tracker.getAverage();
    }

    /**
     * Returns a formatted multi-line statistics string suitable for
     * the HUD overlay display. Uses terminal green ALL CAPS format
     * consistent with the app's visual theme.
     * <p>
     * Output format:
     * <pre>
     * FPS: 28.5 | FRAME: 35.1MS
     * SEGMENT: 28MS | DETECT: 22MS
     * DEPTH: 45MS | FACE: 18MS
     * </pre>
     *
     * @return Formatted statistics string.
     */
    @NonNull
    public String getStats() {
        StringBuilder sb = new StringBuilder(128);

        // Frame stats
        float fps = getFps();
        float frameTime = getAverageFrameTime();
        sb.append(String.format(Locale.US, "FPS: %.1f | FRAME: %.1fMS",
                fps, frameTime));

        // Inference stats — one line per pipeline (max 4 most recent)
        int pipelinesShown = 0;
        for (Map.Entry<String, InferenceTracker> entry : mInferenceTrackers.entrySet()) {
            if (pipelinesShown >= 4) break;

            if (pipelinesShown == 0) {
                sb.append('\n');
            } else {
                sb.append(" | ");
            }

            // Abbreviate pipeline name for display
            String name = abbreviatePipelineName(entry.getKey());
            float avgMs = entry.getValue().getAverage();
            sb.append(String.format(Locale.US, "%s: %.0fMS", name, avgMs));
            pipelinesShown++;
        }

        // Second line if more than 2 pipelines
        if (mInferenceTrackers.size() > 2) {
            sb.append('\n');
            int count = 0;
            boolean first = true;
            for (Map.Entry<String, InferenceTracker> entry : mInferenceTrackers.entrySet()) {
                if (count < 2) {
                    count++;
                    continue;
                }
                if (count >= 6) break;

                if (!first) {
                    sb.append(" | ");
                }
                first = false;

                String name = abbreviatePipelineName(entry.getKey());
                float avgMs = entry.getValue().getAverage();
                sb.append(String.format(Locale.US, "%s: %.0fMS", name, avgMs));
                count++;
            }
        }

        return sb.toString();
    }

    /**
     * Abbreviates a pipeline name for HUD display.
     * "SegmentationPipeline" → "SEGMENT"
     * "DetectionPipeline" → "DETECT"
     *
     * @param name Full pipeline class name.
     * @return Abbreviated ALL CAPS name.
     */
    @NonNull
    private String abbreviatePipelineName(@NonNull String name) {
        // Remove "Pipeline" suffix if present
        String abbreviated = name.replace("Pipeline", "");

        // Known abbreviations for common pipelines
        switch (abbreviated.toLowerCase(Locale.US)) {
            case "segmentation":  return "SEGMENT";
            case "detection":     return "DETECT";
            case "depth":         return "DEPTH";
            case "face":          return "FACE";
            case "ocr":           return "OCR";
            case "weather":       return "WEATHER";
            case "plant":         return "PLANT";
            case "barcode":       return "BARCODE";
            case "landmark":      return "LANDMARK";
            default:
                // Take first 8 chars, uppercase
                if (abbreviated.length() <= 8) {
                    return abbreviated.toUpperCase(Locale.US);
                }
                return abbreviated.substring(0, 8).toUpperCase(Locale.US);
        }
    }

    // ================================================================
    // Reset
    // ================================================================

    /**
     * Resets all performance statistics. Useful when the camera
     * is restarted or the app enters a new operating mode.
     */
    public void reset() {
        synchronized (mFrameTimestamps) {
            mFrameIndex = 0;
            mFrameTimeIndex = 0;
            mFrameCount = 0;
            for (int i = 0; i < BUFFER_SIZE; i++) {
                mFrameTimestamps[i] = 0;
                mFrameTimes[i] = 0;
            }
        }
        mInferenceTrackers.clear();
        Log.d(TAG, "Performance stats reset");
    }

    // ================================================================
    // Inner Class: InferenceTracker
    // ================================================================

    /**
     * Circular buffer tracker for inference timing of a single
     * ML pipeline. Thread-safe via synchronized methods.
     */
    private static class InferenceTracker {

        /** Circular buffer of inference durations in milliseconds. */
        private final long[] mDurations;

        /** Current write index. */
        private int mIndex = 0;

        /** Number of samples recorded. */
        private int mCount = 0;

        /**
         * Creates a tracker with the given buffer size.
         *
         * @param bufferSize Number of samples to track.
         */
        InferenceTracker(int bufferSize) {
            mDurations = new long[bufferSize];
        }

        /**
         * Records an inference duration.
         *
         * @param durationMs Duration in milliseconds.
         */
        synchronized void record(long durationMs) {
            mDurations[mIndex] = durationMs;
            mIndex = (mIndex + 1) % mDurations.length;
            mCount++;
        }

        /**
         * Returns the average inference duration over the buffer.
         *
         * @return Average duration in ms, or 0 if no data.
         */
        synchronized float getAverage() {
            int count = Math.min(mCount, mDurations.length);
            if (count == 0) return 0f;

            long sum = 0;
            for (int i = 0; i < count; i++) {
                sum += mDurations[i];
            }

            return (float) sum / count;
        }

        /**
         * Returns the most recently recorded inference duration.
         *
         * @return Last duration in ms, or 0 if no data.
         */
        synchronized float getLast() {
            if (mCount == 0) return 0f;
            int lastIdx = (mIndex - 1 + mDurations.length) % mDurations.length;
            return mDurations[lastIdx];
        }
    }
}
