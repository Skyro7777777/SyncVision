/**
 * SyncDiagramView.java
 *
 * Custom View that draws the sync diagram â€” a relationship graph showing
 * detected objects as nodes and their spatial relationships as edges.
 * The diagram uses terminal green (#00FF41) on a semi-transparent dark
 * background, matching the E.D.I.T.H-style HUD aesthetic.
 *
 * Features:
 *   - Force-directed layout (simplified, from C++ sync_diagram)
 *   - ALL CAPS labels on nodes
 *   - Relationship labels on edges (ON, NEAR, CONTAINS, BLOCKS, SUPPORTS)
 *   - Animated: nodes slowly settle into position
 *   - Semi-transparent dark background
 *   - Small default size: ~200x200dp
 *   - Touch to expand/collapse
 *
 * Sync Vision â€” Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.ui
 * Target SDK: 29+
 */

package com.syncvision.app.ui;

import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.DecelerateInterpolator;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.syncvision.app.ml.InferenceResult;
import com.syncvision.app.nativelib.NativeConstants;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Custom view that renders a force-directed relationship diagram
 * of detected objects and their spatial relationships.
 * <p>
 * The diagram shows objects as labeled circles (nodes) connected
 * by lines (edges) that represent spatial relationships like
 * ON, NEAR, CONTAINS, BLOCKS, SUPPORTS.
 * <p>
 * Touch to expand the diagram to full-screen overlay; touch again
 * to collapse back to the small floating view.
 */
public class SyncDiagramView extends View {

    private static final String TAG = "SV-SyncDiagramView";

    // ================================================================
    // Visual Constants
    // ================================================================

    /** Terminal green color (#00FF41). */
    private static final int TERMINAL_GREEN = Color.rgb(0, 255, 65);

    /** Dim terminal green for edges and secondary elements. */
    private static final int DIM_GREEN = Color.rgb(0, 160, 40);

    /** Dark background color (semi-transparent). */
    private static final int BG_COLOR = Color.argb(200, 5, 12, 8);

    /** Node background color (semi-transparent dark). */
    private static final int NODE_BG_COLOR = Color.argb(180, 10, 30, 15);

    /** Node border color. */
    private static final int NODE_BORDER_COLOR = Color.argb(200, 0, 255, 65);

    /** Text color for node labels. */
    private static final int TEXT_COLOR = TERMINAL_GREEN;

    /** Edge text color (dimmer). */
    private static final int EDGE_TEXT_COLOR = Color.argb(180, 0, 200, 50);

    /** Default node radius in pixels. */
    private static final float NODE_RADIUS = 22f;

    /** Minimum node radius. */
    private static final float MIN_NODE_RADIUS = 14f;

    /** Maximum node radius. */
    private static final float MAX_NODE_RADIUS = 32f;

    /** Node label text size in pixels. */
    private static final float NODE_TEXT_SIZE = 10f;

    /** Edge label text size in pixels. */
    private static final float EDGE_TEXT_SIZE = 8f;

    /** Edge line width. */
    private static final float EDGE_LINE_WIDTH = 1.5f;

    /** Node border width. */
    private static final float NODE_BORDER_WIDTH = 1.5f;

    // ================================================================
    // Force-Directed Layout Constants
    // ================================================================

    /** Coulomb repulsion constant. */
    private static final float REPULSION = 800f;

    /** Hooke attraction constant. */
    private static final float ATTRACTION = 0.005f;

    /** Center gravity constant. */
    private static final float GRAVITY = 0.02f;

    /** Damping factor for velocity decay. */
    private static final float DAMPING = 0.85f;

    /** Maximum displacement per iteration. */
    private static final float MAX_DISPLACEMENT = 10f;

    /** Minimum distance between nodes. */
    private static final float MIN_DISTANCE = 40f;

    /** Number of layout iterations per frame. */
    private static final int LAYOUT_ITERATIONS = 5;

    // ================================================================
    // Inner Classes â€” Data Model
    // ================================================================

    /**
     * Represents a single node in the sync diagram.
     * Each node corresponds to a detected object.
     */
    public static class DiagramNode {
        /** Object ID. */
        public int id;

        /** Object name (displayed in ALL CAPS). */
        @NonNull
        public String name;

        /** Current X position. */
        public float x;

        /** Current Y position. */
        public float y;

        /** Velocity X. */
        public float vx;

        /** Velocity Y. */
        public float vy;

        /** Node radius (varies by object type). */
        public float radius;

        /** Icon type (PERSON, VEHICLE, etc.). */
        public int iconType;

        public DiagramNode(int id, @NonNull String name, float x, float y, int iconType) {
            this.id = id;
            this.name = name;
            this.x = x;
            this.y = y;
            this.vx = 0f;
            this.vy = 0f;
            this.iconType = iconType;
            this.radius = NODE_RADIUS;
        }
    }

    /**
     * Represents an edge (relationship) between two nodes.
     */
    public static class DiagramEdge {
        /** Source node ID. */
        public int fromId;

        /** Destination node ID. */
        public int toId;

        /** Relationship type (ON, NEAR, CONTAINS, BLOCKS, SUPPORTS). */
        public int relationship;

        public DiagramEdge(int fromId, int toId, int relationship) {
            this.fromId = fromId;
            this.toId = toId;
            this.relationship = relationship;
        }
    }

    // ================================================================
    // State
    // ================================================================

    /** Current diagram nodes. */
    private final List<DiagramNode> nodes = new ArrayList<>();

    /** Current diagram edges. */
    private final List<DiagramEdge> edges = new ArrayList<>();

    /** Whether the diagram is expanded (full-screen overlay). */
    private boolean expanded = false;

    /** Whether the layout has been initialized. */
    private boolean layoutInitialized = false;

    /** Random for initial position seeding. */
    private final Random random = new Random();

    /** Animation runnable for force-directed layout. */
    private final Runnable layoutRunnable = this::runLayoutStep;

    /** Whether the layout animation is running. */
    private boolean layoutRunning = false;

    // ================================================================
    // Paints
    // ================================================================

    private final Paint bgPaint;
    private final Paint edgePaint;
    private final Paint edgeDashedPaint;
    private final Paint edgeTextPaint;
    private final Paint nodeBgPaint;
    private final Paint nodeBorderPaint;
    private final Paint nodeTextPaint;
    private final Paint nodeIconPaint;

    // ================================================================
    // Constructors
    // ================================================================

    public SyncDiagramView(@NonNull Context context) {
        this(context, null);
    }

    public SyncDiagramView(@NonNull Context context, @Nullable AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public SyncDiagramView(@NonNull Context context, @Nullable AttributeSet attrs,
                           int defStyleAttr) {
        super(context, attrs, defStyleAttr);

        // Initialize paints
        bgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        bgPaint.setColor(BG_COLOR);
        bgPaint.setStyle(Paint.Style.FILL);

        edgePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        edgePaint.setColor(DIM_GREEN);
        edgePaint.setStyle(Paint.Style.STROKE);
        edgePaint.setStrokeWidth(EDGE_LINE_WIDTH);

        edgeDashedPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        edgeDashedPaint.setColor(DIM_GREEN);
        edgeDashedPaint.setStyle(Paint.Style.STROKE);
        edgeDashedPaint.setStrokeWidth(EDGE_LINE_WIDTH);
        edgeDashedPaint.setPathEffect(new DashPathEffect(new float[]{6f, 4f}, 0f));

        edgeTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        edgeTextPaint.setColor(EDGE_TEXT_COLOR);
        edgeTextPaint.setTypeface(Typeface.MONOSPACE);
        edgeTextPaint.setTextSize(EDGE_TEXT_SIZE);
        edgeTextPaint.setTextAlign(Paint.Align.CENTER);

        nodeBgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        nodeBgPaint.setColor(NODE_BG_COLOR);
        nodeBgPaint.setStyle(Paint.Style.FILL);

        nodeBorderPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        nodeBorderPaint.setColor(NODE_BORDER_COLOR);
        nodeBorderPaint.setStyle(Paint.Style.STROKE);
        nodeBorderPaint.setStrokeWidth(NODE_BORDER_WIDTH);

        nodeTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        nodeTextPaint.setColor(TEXT_COLOR);
        nodeTextPaint.setTypeface(Typeface.MONOSPACE);
        nodeTextPaint.setTextSize(NODE_TEXT_SIZE);
        nodeTextPaint.setTextAlign(Paint.Align.CENTER);

        nodeIconPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        nodeIconPaint.setColor(TERMINAL_GREEN);
        nodeIconPaint.setTypeface(Typeface.MONOSPACE);
        nodeIconPaint.setTextSize(14f);
        nodeIconPaint.setTextAlign(Paint.Align.CENTER);

        // Enable click for expand/collapse
        setClickable(true);
        setFocusable(true);

        Log.d(TAG, "SyncDiagramView initialized");
    }

    // ================================================================
    // Data Updates
    // ================================================================

    /**
     * Updates the diagram from a SceneResult.
     * Extracts detected objects and creates nodes/edges.
     *
     * @param result The latest scene result from ML pipelines.
     */
    public void updateFromScene(@NonNull InferenceResult.SceneResult result) {
        boolean needsLayout = false;

        // Add nodes from detected objects
        if (result.detection != null && result.detection.objects != null) {
            // Check for new objects
            for (InferenceResult.DetectedObject obj : result.detection.objects) {
                if (findNode(obj.id) == null) {
                    // New object â€” add a node at a random position
                    float cx = getWidth() > 0 ? getWidth() / 2f : 100f;
                    float cy = getHeight() > 0 ? getHeight() / 2f : 100f;
                    float rx = (random.nextFloat() - 0.5f) * 60f;
                    float ry = (random.nextFloat() - 0.5f) * 60f;

                    int iconType = classifyIconType(obj.name);
                    DiagramNode node = new DiagramNode(
                            obj.id,
                            obj.name.toUpperCase(Locale.US),
                            cx + rx, cy + ry,
                            iconType
                    );

                    // Adjust radius based on detection confidence and size
                    node.radius = Math.max(MIN_NODE_RADIUS,
                            Math.min(MAX_NODE_RADIUS,
                                    NODE_RADIUS + obj.bbox.area() * 50f));

                    nodes.add(node);
                    needsLayout = true;
                }
            }

            // Remove nodes for objects that are no longer detected
            nodes.removeIf(node -> {
                boolean found = false;
                for (InferenceResult.DetectedObject obj : result.detection.objects) {
                    if (obj.id == node.id) {
                        found = true;
                        break;
                    }
                }
                return !found;
            });
        }

        // Update node names (in case classification changed)
        if (result.detection != null && result.detection.objects != null) {
            for (InferenceResult.DetectedObject obj : result.detection.objects) {
                DiagramNode node = findNode(obj.id);
                if (node != null) {
                    node.name = obj.name.toUpperCase(Locale.US);
                }
            }
        }

        // Generate edges from relationships
        // For simplicity, we generate NEAR edges between nearby objects
        edges.clear();
        if (nodes.size() > 1) {
            for (int i = 0; i < nodes.size(); i++) {
                for (int j = i + 1; j < nodes.size(); j++) {
                    DiagramNode a = nodes.get(i);
                    DiagramNode b = nodes.get(j);
                    float dx = a.x - b.x;
                    float dy = a.y - b.y;
                    float dist = (float) Math.sqrt(dx * dx + dy * dy);

                    // Only connect nearby nodes (in diagram space)
                    if (dist < Math.max(getWidth(), getHeight()) * 0.6f
                            || nodes.size() <= 4) {
                        // Determine relationship based on spatial analysis
                        int rel = determineRelationship(a, b);
                        edges.add(new DiagramEdge(a.id, b.id, rel));
                    }
                }
            }
        }

        if (needsLayout && !layoutRunning) {
            startLayoutAnimation();
        }

        invalidate();
    }

    /**
     * Sets the diagram data directly from pre-computed nodes and edges.
     *
     * @param newNodes The diagram nodes.
     * @param newEdges The diagram edges.
     */
    public void setDiagramData(@NonNull List<DiagramNode> newNodes,
                               @NonNull List<DiagramEdge> newEdges) {
        nodes.clear();
        nodes.addAll(newNodes);
        edges.clear();
        edges.addAll(newEdges);
        layoutInitialized = false;
        startLayoutAnimation();
        invalidate();
    }

    // ================================================================
    // Force-Directed Layout
    // ================================================================

    /**
     * Initializes the layout by placing nodes in a circle around
     * the center of the view.
     */
    private void initializeLayout() {
        if (nodes.isEmpty() || getWidth() == 0 || getHeight() == 0) return;

        float cx = getWidth() / 2f;
        float cy = getHeight() / 2f;
        float radius = Math.min(cx, cy) * 0.6f;

        for (int i = 0; i < nodes.size(); i++) {
            double angle = (2.0 * Math.PI * i) / nodes.size();
            nodes.get(i).x = cx + (float) (radius * Math.cos(angle));
            nodes.get(i).y = cy + (float) (radius * Math.sin(angle));
            nodes.get(i).vx = 0f;
            nodes.get(i).vy = 0f;
        }

        layoutInitialized = true;
    }

    /**
     * Runs one step of the force-directed layout algorithm.
     * Computes repulsion, attraction, and gravity forces,
     * then updates node positions with damping.
     */
    private void runLayoutStep() {
        if (nodes.isEmpty() || getWidth() == 0) {
            layoutRunning = false;
            return;
        }

        if (!layoutInitialized) {
            initializeLayout();
        }

        for (int iter = 0; iter < LAYOUT_ITERATIONS; iter++) {
            float cx = getWidth() / 2f;
            float cy = getHeight() / 2f;

            // Compute forces on each node
            for (DiagramNode node : nodes) {
                float fx = 0f;
                float fy = 0f;

                // Repulsion from all other nodes (Coulomb)
                for (DiagramNode other : nodes) {
                    if (other.id == node.id) continue;

                    float dx = node.x - other.x;
                    float dy = node.y - other.y;
                    float distSq = dx * dx + dy * dy;
                    if (distSq < 1f) distSq = 1f;
                    float dist = (float) Math.sqrt(distSq);

                    float force = REPULSION / distSq;
                    fx += (dx / dist) * force;
                    fy += (dy / dist) * force;
                }

                // Attraction along edges (Hooke)
                for (DiagramEdge edge : edges) {
                    DiagramNode other = null;
                    if (edge.fromId == node.id) {
                        other = findNode(edge.toId);
                    } else if (edge.toId == node.id) {
                        other = findNode(edge.fromId);
                    }
                    if (other == null) continue;

                    float dx = other.x - node.x;
                    float dy = other.y - node.y;
                    float dist = (float) Math.sqrt(dx * dx + dy * dy);

                    float force = ATTRACTION * dist;
                    fx += (dx / Math.max(dist, 1f)) * force;
                    fy += (dy / Math.max(dist, 1f)) * force;
                }

                // Gravity toward center
                float gdx = cx - node.x;
                float gdy = cy - node.y;
                fx += gdx * GRAVITY;
                fy += gdy * GRAVITY;

                // Update velocity with damping
                node.vx = (node.vx + fx) * DAMPING;
                node.vy = (node.vy + fy) * DAMPING;

                // Clamp displacement
                float displacement = (float) Math.sqrt(node.vx * node.vx + node.vy * node.vy);
                if (displacement > MAX_DISPLACEMENT) {
                    node.vx = (node.vx / displacement) * MAX_DISPLACEMENT;
                    node.vy = (node.vy / displacement) * MAX_DISPLACEMENT;
                }
            }

            // Apply velocities
            float padding = NODE_RADIUS * 2;
            for (DiagramNode node : nodes) {
                node.x += node.vx;
                node.y += node.vy;

                // Keep nodes within bounds
                node.x = Math.max(padding, Math.min(getWidth() - padding, node.x));
                node.y = Math.max(padding, Math.min(getHeight() - padding, node.y));
            }
        }

        invalidate();

        // Continue animation if nodes are still moving
        boolean stillMoving = false;
        for (DiagramNode node : nodes) {
            if (Math.abs(node.vx) > 0.1f || Math.abs(node.vy) > 0.1f) {
                stillMoving = true;
                break;
            }
        }

        if (stillMoving) {
            postDelayed(layoutRunnable, 16); // ~60 FPS
        } else {
            layoutRunning = false;
        }
    }

    /**
     * Starts the force-directed layout animation.
     */
    private void startLayoutAnimation() {
        if (layoutRunning) return;
        layoutRunning = true;
        post(layoutRunnable);
    }

    // ================================================================
    // Drawing
    // ================================================================

    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);

        int w = getWidth();
        int h = getHeight();
        if (w == 0 || h == 0) return;

        // Draw dark background
        canvas.drawRect(0, 0, w, h, bgPaint);

        // Draw border
        Paint borderPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        borderPaint.setColor(Color.argb(100, 0, 255, 65));
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(1f);
        canvas.drawRect(0, 0, w, h, borderPaint);

        // Draw title
        Paint titlePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        titlePaint.setColor(TERMINAL_GREEN);
        titlePaint.setTypeface(Typeface.MONOSPACE);
        titlePaint.setTextSize(9f);
        titlePaint.setTextAlign(Paint.Align.LEFT);
        canvas.drawText("SYNC DIAGRAM", 6f, 12f, titlePaint);

        if (nodes.isEmpty()) {
            // Draw "NO DATA" placeholder
            Paint noDataPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            noDataPaint.setColor(DIM_GREEN);
            noDataPaint.setTypeface(Typeface.MONOSPACE);
            noDataPaint.setTextSize(10f);
            noDataPaint.setTextAlign(Paint.Align.CENTER);
            canvas.drawText("NO DATA", w / 2f, h / 2f, noDataPaint);
            return;
        }

        // Draw edges
        for (DiagramEdge edge : edges) {
            DiagramNode from = findNode(edge.fromId);
            DiagramNode to = findNode(edge.toId);
            if (from == null || to == null) continue;

            // Choose paint style based on relationship
            Paint paint = (edge.relationship == NativeConstants.RELATIONSHIP_NEAR)
                    ? edgeDashedPaint : edgePaint;
            canvas.drawLine(from.x, from.y, to.x, to.y, paint);

            // Draw relationship label at midpoint
            float midX = (from.x + to.x) / 2f;
            float midY = (from.y + to.y) / 2f;
            String relLabel = NativeConstants.relationshipToString(edge.relationship);
            canvas.drawText(relLabel, midX, midY - 4f, edgeTextPaint);
        }

        // Draw nodes
        for (DiagramNode node : nodes) {
            // Node background circle
            canvas.drawCircle(node.x, node.y, node.radius, nodeBgPaint);

            // Node border
            canvas.drawCircle(node.x, node.y, node.radius, nodeBorderPaint);

            // Icon glyph (simple single-char icon)
            String icon = getIconChar(node.iconType);
            canvas.drawText(icon, node.x, node.y + 4f, nodeIconPaint);

            // Node label (ALL CAPS, below the circle)
            float labelY = node.y + node.radius + NODE_TEXT_SIZE + 2f;
            canvas.drawText(node.name, node.x, labelY, nodeTextPaint);
        }

        // Draw expand/collapse indicator
        Paint indicatorPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        indicatorPaint.setColor(DIM_GREEN);
        indicatorPaint.setTypeface(Typeface.MONOSPACE);
        indicatorPaint.setTextSize(8f);
        indicatorPaint.setTextAlign(Paint.Align.RIGHT);
        canvas.drawText(expanded ? "[COLLAPSE]" : "[EXPAND]", w - 6f, 12f, indicatorPaint);
    }

    // ================================================================
    // Touch Handling (Expand/Collapse)
    // ================================================================

    @Override
    public boolean onTouchEvent(@NonNull MotionEvent event) {
        if (event.getAction() == MotionEvent.ACTION_UP) {
            toggleExpand();
            return true;
        }
        return super.onTouchEvent(event);
    }

    /**
     * Toggles the diagram between expanded and collapsed states.
     * Expanded: fills most of the screen for detailed view.
     * Collapsed: small 200x200dp floating view in the corner.
     */
    private void toggleExpand() {
        expanded = !expanded;

        int currentWidth = getWidth();
        int currentHeight = getHeight();

        if (expanded) {
            // Expand to fill most of the screen
            int targetWidth = (int) (getResources().getDisplayMetrics().widthPixels * 0.85);
            int targetHeight = (int) (getResources().getDisplayMetrics().heightPixels * 0.7);

            animateSizeChange(currentWidth, currentHeight, targetWidth, targetHeight);

            // Bring to front
            bringToFront();
        } else {
            // Collapse to default size
            int targetSize = (int) (200 * getResources().getDisplayMetrics().density);
            animateSizeChange(currentWidth, currentHeight, targetSize, targetSize);
        }

        // Re-run layout after size change
        layoutInitialized = false;
        startLayoutAnimation();

        Log.d(TAG, "Diagram " + (expanded ? "expanded" : "collapsed"));
    }

    /**
     * Animates a size change using ValueAnimator.
     */
    private void animateSizeChange(int fromW, int fromH, int toW, int toH) {
        ValueAnimator animator = ValueAnimator.ofFloat(0f, 1f);
        animator.setDuration(300);
        animator.setInterpolator(new DecelerateInterpolator());
        animator.addUpdateListener(animation -> {
            float t = (float) animation.getAnimatedValue();
            int w = Math.round(fromW + (toW - fromW) * t);
            int h = Math.round(fromH + (toH - fromH) * t);

            ViewGroup.LayoutParams params = getLayoutParams();
            if (params != null) {
                params.width = w;
                params.height = h;
                setLayoutParams(params);
            }
        });
        animator.start();
    }

    // ================================================================
    // Helpers
    // ================================================================

    /**
     * Finds a node by its ID.
     *
     * @param id The node ID to search for.
     * @return The matching node, or null.
     */
    @Nullable
    private DiagramNode findNode(int id) {
        for (DiagramNode node : nodes) {
            if (node.id == id) return node;
        }
        return null;
    }

    /**
     * Classifies an object name into an icon type.
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
        // Nature
        if (lower.contains("plant") || lower.contains("tree") || lower.contains("flower")
                || lower.contains("grass") || lower.contains("potted")) {
            return NativeConstants.ICON_NATURE;
        }

        // Default: nature (catch-all for animals, etc.)
        if (lower.contains("dog") || lower.contains("cat") || lower.contains("bird")
                || lower.contains("horse") || lower.contains("sheep")
                || lower.contains("cow") || lower.contains("bear")) {
            return NativeConstants.ICON_NATURE;
        }

        return NativeConstants.ICON_FURNITURE; // Default
    }

    /**
     * Returns a single-character icon glyph for an icon type.
     *
     * @param iconType The icon type constant.
     * @return A character representing the icon.
     */
    @NonNull
    private String getIconChar(int iconType) {
        switch (iconType) {
            case NativeConstants.ICON_PERSON:      return "P";
            case NativeConstants.ICON_VEHICLE:     return "V";
            case NativeConstants.ICON_FURNITURE:   return "F";
            case NativeConstants.ICON_ELECTRONICS: return "E";
            case NativeConstants.ICON_NATURE:      return "N";
            default:                               return "?";
        }
    }

    /**
     * Determines the relationship between two nodes based on
     * their positions and types.
     *
     * @param a First node.
     * @param b Second node.
     * @return Relationship type constant.
     */
    private int determineRelationship(@NonNull DiagramNode a, @NonNull DiagramNode b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;

        // Vertical adjacency: A is above B â†’ ON or SUPPORTS
        if (Math.abs(dx) < Math.abs(dy) * 0.5f && a.y < b.y) {
            // Person above furniture â†’ ON
            if (a.iconType == NativeConstants.ICON_PERSON
                    && (b.iconType == NativeConstants.ICON_FURNITURE
                    || b.iconType == NativeConstants.ICON_VEHICLE)) {
                return NativeConstants.RELATIONSHIP_ON;
            }
            return NativeConstants.RELATIONSHIP_SUPPORTS;
        }

        // Default: NEAR (proximity)
        return NativeConstants.RELATIONSHIP_NEAR;
    }

    /**
     * Returns whether the diagram is currently expanded.
     */
    public boolean isExpanded() {
        return expanded;
    }

    /**
     * Clears all diagram data.
     */
    public void clear() {
        nodes.clear();
        edges.clear();
        layoutInitialized = false;
        invalidate();
    }
}
