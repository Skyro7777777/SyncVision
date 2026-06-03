/**
 * OverlayRenderer.java
 *
 * High-level API for the Sync Vision rendering system. This class bridges
 * the ML pipeline results with the OpenGL renderer, converting ML outputs
 * (segmentation masks, depth maps, detected labels, path data) into
 * GPU textures and vertex buffers for the GLRenderer.
 *
 * Responsibilities:
 *   - Convert ML results (SceneResult) into GL textures
 *   - Generate label bitmaps using Canvas (monospace font, ALL CAPS, #00FF41)
 *   - Create semi-transparent dark panel backgrounds behind labels
 *   - Upload Bitmap data to GL textures via the GLRenderer
 *   - Convert path waypoints into vertex buffers for path visualization
 *   - Manage night mode and scanline effect parameters
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.rendering
 * Target SDK: 29+
 */

package com.syncvision.app.rendering;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.syncvision.app.ml.InferenceResult;
import com.syncvision.app.nativelib.NativeProcessor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * High-level rendering API that converts ML pipeline outputs into
 * GPU-ready textures and geometry for the GLRenderer.
 * <p>
 * This class is called from the FrameDispatcher (or UI thread) to
 * update the rendering state. It handles the CPU-side work of
 * converting ML results to visual data, which is then uploaded
 * to the GPU on the next frame by the GLRenderer.
 * <p>
 * Label rendering style:
 *   - Font: Monospace, ALL CAPS
 *   - Text color: Terminal green #00FF41
 *   - Background: Semi-transparent dark panel
 *   - Text shadow: 1px diagonal offset for readability
 */
public class OverlayRenderer {

    private static final String TAG = "SV-OverlayRenderer";

    // -----------------------------------------------------------------
    // Terminal green color constants (#00FF41)
    // -----------------------------------------------------------------

    /** Terminal green as Android Color int. */
    private static final int TERMINAL_GREEN = Color.rgb(0, 255, 65);

    /** Terminal green as float array for GL (R, G, B). */
    private static final float[] TERMINAL_GREEN_F = {0.0f, 1.0f, 0.255f};

    // -----------------------------------------------------------------
    // Label rendering constants
    // -----------------------------------------------------------------

    /** Minimum font size in pixels for labels. */
    private static final int MIN_FONT_SIZE = 12;

    /** Maximum font size in pixels for labels. */
    private static final int MAX_FONT_SIZE = 24;

    /** Padding around label text in pixels. */
    private static final int LABEL_PADDING = 8;

    /** Background panel alpha (0-255). */
    private static final int BG_ALPHA = 140;

    /** Text shadow offset in pixels. */
    private static final int SHADOW_OFFSET = 1;

    // -----------------------------------------------------------------
    // Rendering components
    // -----------------------------------------------------------------

    /** The GL renderer that handles actual GPU rendering. */
    private final GLRenderer glRenderer;

    /** The native processor for C++ operations. */
    private final NativeProcessor nativeProcessor;

    // -----------------------------------------------------------------
    // Label bitmap cache
    // -----------------------------------------------------------------

    /** Current label bitmap (may be reused across frames). */
    @Nullable
    private Bitmap labelBitmap;

    /** Canvas for rendering labels. */
    @Nullable
    private Canvas labelCanvas;

    /** Paint for label text rendering. */
    private final Paint labelPaint;

    /** Paint for label text shadow. */
    private final Paint shadowPaint;

    /** Paint for background panel. */
    private final Paint bgPaint;

    // -----------------------------------------------------------------
    // State
    // -----------------------------------------------------------------

    /** Viewport width for coordinate mapping. */
    private int viewportWidth = 640;

    /** Viewport height for coordinate mapping. */
    private int viewportHeight = 480;

    /** Whether a label update is pending. */
    private volatile boolean labelUpdatePending = false;

    /** Last processed label list (for incremental updates). */
    @Nullable
    private List<LabelInfo> currentLabels;

    // ================================================================
    // Inner class — LabelInfo
    // ================================================================

    /**
     * Information about a single label to be rendered.
     * Created from DetectedObject data by the OverlayRenderer.
     */
    public static class LabelInfo {
        /** Object ID. */
        public int id;

        /** Display name (will be rendered in ALL CAPS). */
        @NonNull
        public String name;

        /** Detection confidence [0, 1]. */
        public float confidence;

        /** Bounding box center X in normalized coords [0, 1]. */
        public float centerX;

        /** Bounding box center Y in normalized coords [0, 1]. */
        public float centerY;

        /** Bounding box width in normalized coords [0, 1]. */
        public float width;

        /** Bounding box height in normalized coords [0, 1]. */
        public float height;

        /** Estimated distance in meters (-1 if unknown). */
        public float distance;

        /** Hazard level (0=none, 1=low, 2=medium, 3=high). */
        public int hazardLevel;

        public LabelInfo(int id, @NonNull String name, float confidence,
                         float centerX, float centerY, float width, float height,
                         float distance, int hazardLevel) {
            this.id = id;
            this.name = name;
            this.confidence = confidence;
            this.centerX = centerX;
            this.centerY = centerY;
            this.width = width;
            this.height = height;
            this.distance = distance;
            this.hazardLevel = hazardLevel;
        }
    }

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new OverlayRenderer.
     *
     * @param glRenderer      The GL renderer for texture uploads.
     * @param nativeProcessor The native processor for C++ operations.
     */
    public OverlayRenderer(@NonNull GLRenderer glRenderer,
                           @NonNull NativeProcessor nativeProcessor) {
        this.glRenderer = glRenderer;
        this.nativeProcessor = nativeProcessor;

        // Initialize label rendering paints
        labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        labelPaint.setColor(TERMINAL_GREEN);
        labelPaint.setTypeface(Typeface.MONOSPACE);
        labelPaint.setTextSize(14f);
        labelPaint.setTextAlign(Paint.Align.LEFT);

        shadowPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        shadowPaint.setColor(Color.argb(180, 0, 40, 10));
        shadowPaint.setTypeface(Typeface.MONOSPACE);
        shadowPaint.setTextSize(14f);
        shadowPaint.setTextAlign(Paint.Align.LEFT);

        bgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        bgPaint.setColor(Color.argb(BG_ALPHA, 0, 10, 2));
        bgPaint.setStyle(Paint.Style.FILL);
    }

    // ================================================================
    // High-Level API — Scene Updates
    // ================================================================

    /**
     * Updates the entire rendering state from a SceneResult.
     * Called from the FrameDispatcher when new ML results are available.
     *
     * @param result The combined ML scene result.
     */
    public void updateScene(@NonNull InferenceResult.SceneResult result) {
        // Update segmentation mask
        if (result.segmentation != null) {
            updateSegmentationMask(result.segmentation.mask,
                    result.segmentation.width,
                    result.segmentation.height);
        }

        // Update depth map
        if (result.depth != null) {
            updateDepthMap(result.depth.depthMap,
                    result.depth.width,
                    result.depth.height);
        }

        // Update labels from detection results
        if (result.detection != null) {
            List<LabelInfo> labels = createLabelInfoList(result);
            updateLabels(labels);
        }

        // Update path
        if (result.path != null) {
            updatePath(result.path);
        }
    }

    /**
     * Updates the segmentation mask texture.
     * The mask data is queued for upload to the GPU on the next frame.
     *
     * @param mask   2D int array of class IDs [height][width].
     * @param width  Mask width.
     * @param height Mask height.
     */
    public void updateSegmentationMask(@NonNull int[][] mask, int width, int height) {
        try {
            glRenderer.updateMaskData(mask, width, height);
        } catch (Exception e) {
            Log.e(TAG, "Error updating segmentation mask", e);
        }
    }

    /**
     * Updates the depth map texture.
     * The depth data is queued for upload to the GPU on the next frame.
     *
     * @param depthMap 2D float array of depth values [height][width].
     * @param width    Depth map width.
     * @param height   Depth map height.
     */
    public void updateDepthMap(@NonNull float[][] depthMap, int width, int height) {
        try {
            glRenderer.updateDepthData(depthMap, width, height);
        } catch (Exception e) {
            Log.e(TAG, "Error updating depth map", e);
        }
    }

    /**
     * Updates the text labels overlay.
     * Creates a new label bitmap using Canvas rendering with terminal green
     * ALL CAPS text on semi-transparent dark panels, then queues it for
     * upload to the GPU.
     *
     * @param labels List of LabelInfo objects to render.
     */
    public void updateLabels(@NonNull List<LabelInfo> labels) {
        if (labels.isEmpty()) {
            // Clear the label texture
            clearLabelTexture();
            currentLabels = null;
            return;
        }

        this.currentLabels = new ArrayList<>(labels);

        try {
            // Render labels to bitmap
            ByteBuffer pixelData = renderLabelsToBitmap(labels);
            if (pixelData != null) {
                glRenderer.updateLabelData(pixelData,
                        viewportWidth, viewportHeight);
                labelUpdatePending = false;
            }
        } catch (Exception e) {
            Log.e(TAG, "Error updating labels", e);
        }
    }

    /**
     * Updates the path visualization.
     * Converts path waypoints into vertex data for line rendering.
     *
     * @param pathResult The path result from the ML pipeline.
     */
    public void updatePath(@NonNull InferenceResult.PathResult pathResult) {
        try {
            FloatBuffer pathBuffer = createPathVertexBuffer(pathResult);
            if (pathBuffer != null) {
                int vertexCount = pathResult.getWaypointCount() * 2 - 2;
                if (vertexCount > 0) {
                    glRenderer.setPathData(pathBuffer, vertexCount);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Error updating path", e);
        }
    }

    /**
     * Sets night mode on or off.
     *
     * @param enabled true to enable night mode.
     */
    public void setNightMode(boolean enabled) {
        glRenderer.setNightMode(enabled);
    }

    /**
     * Sets the CRT scanline effect intensity.
     *
     * @param intensity Intensity [0.0 = off, 1.0 = maximum].
     */
    public void setScanlineIntensity(float intensity) {
        glRenderer.setScanlineIntensity(intensity);
    }

    /**
     * Sets the viewport dimensions for coordinate mapping.
     * Must be called when the surface size changes.
     *
     * @param width  Viewport width in pixels.
     * @param height Viewport height in pixels.
     */
    public void setViewportSize(int width, int height) {
        this.viewportWidth = width;
        this.viewportHeight = height;
    }

    // ================================================================
    // Label Bitmap Generation
    // ================================================================

    /**
     * Renders the given labels onto a bitmap using Canvas.
     * Each label is rendered as:
     *   - Semi-transparent dark background panel
     *   - ALL CAPS monospace text in terminal green #00FF41
     *   - 1px diagonal text shadow for readability
     *   - Confidence percentage and distance (if available)
     *
     * @param labels The labels to render.
     * @return ByteBuffer of RGBA pixel data, or null on failure.
     */
    @Nullable
    private ByteBuffer renderLabelsToBitmap(@NonNull List<LabelInfo> labels) {
        int w = viewportWidth;
        int h = viewportHeight;

        if (w <= 0 || h <= 0) {
            Log.w(TAG, "Invalid viewport size for label rendering: " + w + "x" + h);
            return null;
        }

        // Create or reuse the label bitmap
        if (labelBitmap == null || labelBitmap.getWidth() != w
                || labelBitmap.getHeight() != h) {
            if (labelBitmap != null && !labelBitmap.isRecycled()) {
                labelBitmap.recycle();
            }
            labelBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
            labelCanvas = new Canvas(labelBitmap);
        }

        // Clear the bitmap (fully transparent)
        labelBitmap.eraseColor(Color.TRANSPARENT);

        // Render each label
        for (LabelInfo label : labels) {
            renderSingleLabel(labelCanvas, label, w, h);
        }

        // Convert bitmap to ByteBuffer (RGBA)
        ByteBuffer buffer = ByteBuffer.allocateDirect(w * h * 4)
                .order(ByteOrder.nativeOrder());
        labelBitmap.copyPixelsToBuffer(buffer);
        buffer.position(0);

        return buffer;
    }

    /**
     * Renders a single label onto the canvas.
     *
     * @param canvas The canvas to draw on.
     * @param label  The label information.
     * @param canvasWidth  Canvas width.
     * @param canvasHeight Canvas height.
     */
    private void renderSingleLabel(@NonNull Canvas canvas, @NonNull LabelInfo label,
                                   int canvasWidth, int canvasHeight) {
        // Compute label position in pixel coordinates
        float posX = label.centerX * canvasWidth;
        float posY = label.centerY * canvasHeight;

        // Compute font size based on depth/distance
        int fontSize = computeFontSize(label.distance);
        labelPaint.setTextSize(fontSize);
        shadowPaint.setTextSize(fontSize);

        // Build the label text (ALL CAPS)
        String mainText = label.name.toUpperCase(Locale.US);
        String detailText = buildDetailText(label);

        // Measure text bounds
        Rect mainBounds = new Rect();
        labelPaint.getTextBounds(mainText, 0, mainText.length(), mainBounds);

        Rect detailBounds = new Rect();
        if (!detailText.isEmpty()) {
            labelPaint.getTextBounds(detailText, 0, detailText.length(), detailBounds);
        }

        // Compute panel dimensions
        int textWidth = Math.max(mainBounds.width(),
                detailText.isEmpty() ? 0 : detailBounds.width());
        int totalTextHeight = mainBounds.height()
                + (detailText.isEmpty() ? 0 : detailBounds.height() + 4);

        int panelWidth = textWidth + LABEL_PADDING * 2;
        int panelHeight = totalTextHeight + LABEL_PADDING * 2;

        // Position: place label above the object center
        float panelLeft = posX - panelWidth / 2f;
        float panelTop = posY - label.height * canvasHeight / 2f - panelHeight - 4;

        // Clamp to screen bounds
        panelLeft = Math.max(4, Math.min(panelLeft, canvasWidth - panelWidth - 4));
        panelTop = Math.max(4, Math.min(panelTop, canvasHeight - panelHeight - 4));

        // Draw background panel
        canvas.drawRect(panelLeft, panelTop,
                panelLeft + panelWidth, panelTop + panelHeight, bgPaint);

        // Draw border glow (subtle green outline)
        Paint borderPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        borderPaint.setColor(Color.argb(60, 0, 255, 65));
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(1f);
        canvas.drawRect(panelLeft, panelTop,
                panelLeft + panelWidth, panelTop + panelHeight, borderPaint);

        // Draw text shadow
        float textX = panelLeft + LABEL_PADDING;
        float textY = panelTop + LABEL_PADDING + mainBounds.height();

        canvas.drawText(mainText, textX + SHADOW_OFFSET, textY + SHADOW_OFFSET,
                shadowPaint);

        // Draw main text (terminal green, ALL CAPS)
        canvas.drawText(mainText, textX, textY, labelPaint);

        // Draw detail text (smaller, dimmer green)
        if (!detailText.isEmpty()) {
            float detailY = textY + detailBounds.height() + 4;
            Paint detailPaint = new Paint(labelPaint);
            detailPaint.setTextSize(fontSize * 0.8f);
            detailPaint.setColor(Color.argb(200, 0, 200, 50));
            canvas.drawText(detailText, textX, detailY, detailPaint);
        }
    }

    /**
     * Builds the detail text line for a label (confidence + distance + hazard).
     *
     * @param label The label info.
     * @return The detail text string, or empty string if no details.
     */
    @NonNull
    private String buildDetailText(@NonNull LabelInfo label) {
        StringBuilder sb = new StringBuilder();

        // Confidence percentage
        sb.append(String.format(Locale.US, "%.0f%%", label.confidence * 100));

        // Distance
        if (label.distance > 0) {
            sb.append(String.format(Locale.US, " %.1fm", label.distance));
        }

        // Hazard indicator
        switch (label.hazardLevel) {
            case 1:
                sb.append(" [LOW]");
                break;
            case 2:
                sb.append(" [MED]");
                break;
            case 3:
                sb.append(" [HIGH]");
                break;
        }

        return sb.toString().toUpperCase(Locale.US);
    }

    /**
     * Computes the font size based on estimated distance.
     * Closer objects get larger labels; farther objects get smaller ones.
     *
     * @param distance Estimated distance in meters (-1 if unknown).
     * @return Font size in pixels.
     */
    private int computeFontSize(float distance) {
        if (distance < 0) {
            return (MIN_FONT_SIZE + MAX_FONT_SIZE) / 2; // Default mid-size
        }
        // Map distance: 0.5m → MAX_FONT_SIZE, 10m → MIN_FONT_SIZE
        float t = (distance - 0.5f) / 9.5f;
        t = Math.max(0f, Math.min(1f, t));
        return Math.round(MAX_FONT_SIZE - t * (MAX_FONT_SIZE - MIN_FONT_SIZE));
    }

    /**
     * Clears the label texture (makes it fully transparent).
     */
    private void clearLabelTexture() {
        ByteBuffer clearData = ByteBuffer.allocateDirect(4)
                .order(ByteOrder.nativeOrder());
        clearData.put((byte) 0).put((byte) 0).put((byte) 0).put((byte) 0);
        clearData.position(0);
        glRenderer.updateLabelData(clearData, 1, 1);
    }

    // ================================================================
    // Path Vertex Buffer Generation
    // ================================================================

    /**
     * Creates a vertex buffer for path line rendering from a PathResult.
     * Each segment between waypoints produces a line with metadata
     * for the path shader (segment type, path status, parametric distance).
     *
     * @param pathResult The path result with waypoints.
     * @return FloatBuffer with path vertex data, or null if no path.
     */
    @Nullable
    private FloatBuffer createPathVertexBuffer(@NonNull InferenceResult.PathResult pathResult) {
        List<float[]> waypoints = pathResult.waypoints;
        if (waypoints == null || waypoints.size() < 2) {
            return null;
        }

        // Each segment = 2 vertices × 6 floats (x, y, segmentType, pathStatus, paramDist, widthScale)
        int numSegments = waypoints.size() - 1;
        int floatsPerVertex = 6;
        int totalFloats = numSegments * 2 * floatsPerVertex;

        FloatBuffer buffer = ByteBuffer.allocateDirect(totalFloats * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();

        // Determine path status: 0=CLEAR, 1=PARTIAL, 2=BLOCKED
        float pathStatus = pathResult.isClear ? 0.0f : 1.0f;
        if (pathResult.cost > 100f) {
            pathStatus = 2.0f; // BLOCKED
        }

        for (int i = 0; i < numSegments; i++) {
            float[] from = waypoints.get(i);
            float[] to = waypoints.get(i + 1);

            float paramStart = (float) i / numSegments;
            float paramEnd = (float) (i + 1) / numSegments;

            // Determine segment type: 0=solid for CLEAR, 1=dashed for PARTIAL
            float segmentType = pathResult.isClear ? 0.0f : 1.0f;

            // Width scale (could vary based on confidence or other factors)
            float widthScale = 1.0f;

            // Vertex 1 (from)
            buffer.put(from[0]);            // x
            buffer.put(from[1]);            // y
            buffer.put(segmentType);        // segmentType
            buffer.put(pathStatus);         // pathStatus
            buffer.put(paramStart);         // parametric distance
            buffer.put(widthScale);         // width scale

            // Vertex 2 (to)
            buffer.put(to[0]);              // x
            buffer.put(to[1]);              // y
            buffer.put(segmentType);        // segmentType
            buffer.put(pathStatus);         // pathStatus
            buffer.put(paramEnd);           // parametric distance
            buffer.put(widthScale);         // width scale
        }

        buffer.position(0);
        return buffer;
    }

    // ================================================================
    // LabelInfo Creation Helpers
    // ================================================================

    /**
     * Creates a list of LabelInfo objects from a SceneResult.
     * Combines detection results with depth and segmentation data.
     *
     * @param result The scene result.
     * @return List of LabelInfo for rendering.
     */
    @NonNull
    private List<LabelInfo> createLabelInfoList(@NonNull InferenceResult.SceneResult result) {
        List<LabelInfo> labels = new ArrayList<>();

        if (result.detection == null || result.detection.objects == null) {
            return labels;
        }

        for (InferenceResult.DetectedObject obj : result.detection.objects) {
            float distance = obj.distance;

            // Try to get distance from depth map if not already set
            if (distance < 0 && result.depth != null) {
                float depthVal = result.depth.getDepthAt(
                        obj.bbox.centerX(), obj.bbox.centerY());
                if (depthVal > 0) {
                    distance = result.depth.estimateDistance(depthVal,
                            0.5f, 2.0f); // Rough calibration
                }
            }

            LabelInfo label = new LabelInfo(
                    obj.id,
                    obj.name,
                    obj.confidence,
                    obj.bbox.centerX(),
                    obj.bbox.centerY(),
                    obj.bbox.width(),
                    obj.bbox.height(),
                    distance,
                    obj.hazardLevel
            );

            labels.add(label);
        }

        return labels;
    }

    // ================================================================
    // Cleanup
    // ================================================================

    /**
     * Releases all resources held by the OverlayRenderer.
     * Call this when the rendering system is being destroyed.
     */
    public void cleanup() {
        if (labelBitmap != null && !labelBitmap.isRecycled()) {
            labelBitmap.recycle();
            labelBitmap = null;
        }
        labelCanvas = null;
        currentLabels = null;
        Log.i(TAG, "OverlayRenderer cleaned up");
    }
}
