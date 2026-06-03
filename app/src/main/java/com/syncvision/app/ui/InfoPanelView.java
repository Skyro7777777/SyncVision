/**
 * InfoPanelView.java
 *
 * Shows detailed information about a tapped/selected detected object.
 * The panel slides in from the right side of the screen and displays
 * the object name (large ALL CAPS), description, distance, hazard
 * level, and confidence — all in terminal green on a dark background.
 *
 * The panel has a CLI/terminal aesthetic that matches the E.D.I.T.H
 * HUD style, with monospace ALL CAPS text and green borders.
 *
 * Features:
 *   - Slides in from the right when shown
 *   - Slides out when dismissed
 *   - Close button (X) in top-right corner
 *   - Terminal/CLI output panel appearance
 *   - Shows: object name, description, distance, hazard, confidence
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.ui
 * Target SDK: 29+
 */

package com.syncvision.app.ui;

import android.animation.ObjectAnimator;
import android.content.Context;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.DecelerateInterpolator;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.syncvision.app.ml.InferenceResult;
import com.syncvision.app.nativelib.NativeConstants;

import java.util.Locale;

/**
 * Detail panel view for a detected object. Displays all available
 * information about the object in a terminal-style layout.
 * <p>
 * The panel slides in from the right edge of the screen and slides
 * back out when dismissed. The content is styled with monospace
 * ALL CAPS text in terminal green (#00FF41) on a dark background.
 */
public class InfoPanelView extends LinearLayout {

    private static final String TAG = "SV-InfoPanelView";

    // ================================================================
    // Visual Constants
    // ================================================================

    /** Terminal green color (#00FF41). */
    private static final int TERMINAL_GREEN = Color.rgb(0, 255, 65);

    /** Dim terminal green for secondary text. */
    private static final int DIM_GREEN = Color.argb(200, 0, 200, 50);

    /** Dark background color. */
    private static final int BG_COLOR = Color.argb(230, 8, 16, 10);

    /** Border color. */
    private static final int BORDER_COLOR = Color.argb(180, 0, 255, 65);

    /** Panel border width in pixels. */
    private static final int BORDER_WIDTH_DP = 1;

    /** Slide animation duration in milliseconds. */
    private static final long SLIDE_DURATION_MS = 300L;

    /** Padding in dp. */
    private static final int PADDING_DP = 16;

    /** Title text size in sp. */
    private static final float TITLE_TEXT_SIZE = 20f;

    /** Body text size in sp. */
    private static final float BODY_TEXT_SIZE = 13f;

    /** Label text size in sp. */
    private static final float LABEL_TEXT_SIZE = 10f;

    // ================================================================
    // UI Components
    // ================================================================

    /** Close button. */
    private TextView closeButton;

    /** Title (object name in ALL CAPS). */
    private TextView titleView;

    /** Description text. */
    private TextView descriptionView;

    /** Distance label. */
    private TextView distanceView;

    /** Hazard level label. */
    private TextView hazardView;

    /** Confidence label. */
    private TextView confidenceView;

    /** Additional info section. */
    private TextView additionalView;

    /** Scroll container for content. */
    private ScrollView scrollContainer;

    // ================================================================
    // State
    // ================================================================

    /** Whether the panel is currently visible. */
    private boolean isShowing = false;

    /** Close listener callback. */
    @Nullable
    private Runnable onCloseListener;

    /** Currently displayed object. */
    @Nullable
    private InferenceResult.DetectedObject currentObject;

    // ================================================================
    // Constructors
    // ================================================================

    public InfoPanelView(@NonNull Context context) {
        this(context, null);
    }

    public InfoPanelView(@NonNull Context context, @Nullable AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public InfoPanelView(@NonNull Context context, @Nullable AttributeSet attrs,
                         int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    // ================================================================
    // Initialization
    // ================================================================

    /**
     * Initializes the panel layout and creates all child views.
     */
    private void init() {
        Context context = getContext();
        int paddingPx = dpToPx(PADDING_DP);
        int borderPx = dpToPx(BORDER_WIDTH_DP);

        // Panel orientation and styling
        setOrientation(VERTICAL);
        setPadding(paddingPx, paddingPx, paddingPx, paddingPx);
        setBackgroundColor(BG_COLOR);

        // ---- Header bar (title + close button) ----
        LinearLayout headerBar = new LinearLayout(context);
        headerBar.setOrientation(HORIZONTAL);
        headerBar.setGravity(Gravity.CENTER_VERTICAL);
        headerBar.setPadding(0, 0, 0, dpToPx(8));

        // Title label
        titleView = new TextView(context);
        titleView.setTypeface(Typeface.MONOSPACE);
        titleView.setTextColor(TERMINAL_GREEN);
        titleView.setTextSize(TITLE_TEXT_SIZE);
        titleView.setAllCaps(true);
        titleView.setSingleLine(false);
        titleView.setMaxLines(2);
        LinearLayout.LayoutParams titleParams = new LinearLayout.LayoutParams(
                0, LayoutParams.WRAP_CONTENT, 1f);
        titleView.setLayoutParams(titleParams);

        // Close button
        closeButton = new TextView(context);
        closeButton.setText("[X]");
        closeButton.setTypeface(Typeface.MONOSPACE);
        closeButton.setTextColor(TERMINAL_GREEN);
        closeButton.setTextSize(16f);
        closeButton.setPadding(dpToPx(8), 0, 0, 0);
        closeButton.setClickable(true);
        closeButton.setOnClickListener(v -> hide());

        headerBar.addView(titleView);
        headerBar.addView(closeButton);
        addView(headerBar);

        // ---- Separator line ----
        View separator = new View(context);
        separator.setBackgroundColor(BORDER_COLOR);
        separator.setLayoutParams(new LayoutParams(
                LayoutParams.MATCH_PARENT, borderPx));
        addView(separator);

        // ---- Scrollable content area ----
        scrollContainer = new ScrollView(context);
        scrollContainer.setLayoutParams(new LayoutParams(
                LayoutParams.MATCH_PARENT, 0, 1f));
        scrollContainer.setVerticalScrollBarEnabled(false);

        LinearLayout contentLayout = new LinearLayout(context);
        contentLayout.setOrientation(VERTICAL);
        contentLayout.setPadding(0, dpToPx(8), 0, 0);

        // Description
        descriptionView = createLabelValuePair("DESCRIPTION", context);
        contentLayout.addView(descriptionView);

        // Distance
        distanceView = createLabelValuePair("DISTANCE", context);
        contentLayout.addView(distanceView);

        // Hazard level
        hazardView = createLabelValuePair("HAZARD", context);
        contentLayout.addView(hazardView);

        // Confidence
        confidenceView = createLabelValuePair("CONFIDENCE", context);
        contentLayout.addView(confidenceView);

        // Additional info
        additionalView = createLabelValuePair("NOTES", context);
        contentLayout.addView(additionalView);

        scrollContainer.addView(contentLayout);
        addView(scrollContainer);

        // ---- Bottom separator ----
        View bottomSep = new View(context);
        bottomSep.setBackgroundColor(BORDER_COLOR);
        bottomSep.setLayoutParams(new LayoutParams(
                LayoutParams.MATCH_PARENT, borderPx));
        addView(bottomSep);

        // ---- Footer ----
        TextView footer = new TextView(context);
        footer.setTypeface(Typeface.MONOSPACE);
        footer.setTextColor(DIM_GREEN);
        footer.setTextSize(LABEL_TEXT_SIZE);
        footer.setText("SYNC VISION — TAP [X] OR SWIPE TO CLOSE");
        footer.setAllCaps(true);
        footer.setPadding(0, dpToPx(4), 0, 0);
        addView(footer);

        // Initially hidden
        setVisibility(GONE);
        setTranslationX(getResources().getDisplayMetrics().widthPixels);

        Log.d(TAG, "InfoPanelView initialized");
    }

    /**
     * Creates a styled TextView for a label-value pair in the panel.
     *
     * @param label The label prefix (e.g., "DISTANCE").
     * @param context The context.
     * @return A configured TextView.
     */
    @NonNull
    private TextView createLabelValuePair(@NonNull String label, @NonNull Context context) {
        TextView tv = new TextView(context);
        tv.setTypeface(Typeface.MONOSPACE);
        tv.setTextColor(TERMINAL_GREEN);
        tv.setTextSize(BODY_TEXT_SIZE);
        tv.setAllCaps(true);
        tv.setPadding(0, dpToPx(4), 0, dpToPx(4));
        tv.setSingleLine(false);
        return tv;
    }

    // ================================================================
    // Show / Hide
    // ================================================================

    /**
     * Shows the info panel with data from the given detected object.
     * The panel slides in from the right.
     *
     * @param object The detected object to display.
     */
    public void show(@NonNull InferenceResult.DetectedObject object) {
        currentObject = object;
        updateContent(object);

        setVisibility(VISIBLE);

        // Slide in from the right
        float parentWidth = getWidth() > 0 ? getWidth()
                : getResources().getDisplayMetrics().widthPixels * 0.7f;
        setTranslationX(parentWidth);

        ObjectAnimator slideIn = ObjectAnimator.ofFloat(
                this, "translationX", parentWidth, 0f);
        slideIn.setDuration(SLIDE_DURATION_MS);
        slideIn.setInterpolator(new DecelerateInterpolator());
        slideIn.start();

        isShowing = true;
        Log.d(TAG, "InfoPanel shown for: " + object.name);
    }

    /**
     * Hides the info panel by sliding it out to the right.
     */
    public void hide() {
        float parentWidth = getWidth() > 0 ? getWidth()
                : getResources().getDisplayMetrics().widthPixels * 0.7f;

        ObjectAnimator slideOut = ObjectAnimator.ofFloat(
                this, "translationX", 0f, parentWidth);
        slideOut.setDuration(SLIDE_DURATION_MS);
        slideOut.setInterpolator(new DecelerateInterpolator());
        slideOut.addListener(new android.animation.AnimatorListenerAdapter() {
            @Override
            public void onAnimationEnd(android.animation.Animator animation) {
                setVisibility(GONE);
                currentObject = null;
                isShowing = false;
            }
        });
        slideOut.start();

        if (onCloseListener != null) {
            onCloseListener.run();
        }

        Log.d(TAG, "InfoPanel hiding");
    }

    /**
     * Returns whether the panel is currently visible.
     */
    public boolean isShowing() {
        return isShowing;
    }

    // ================================================================
    // Content Updates
    // ================================================================

    /**
     * Updates the panel content from a DetectedObject.
     *
     * @param obj The detected object.
     */
    private void updateContent(@NonNull InferenceResult.DetectedObject obj) {
        // Title: Object name in ALL CAPS
        titleView.setText(obj.name.toUpperCase(Locale.US));

        // Description
        String desc = (obj.description != null && !obj.description.isEmpty())
                ? obj.description.toUpperCase(Locale.US)
                : "NO ADDITIONAL DESCRIPTION AVAILABLE";
        descriptionView.setText(String.format(Locale.US, "DESCRIPTION:\n%s", desc));

        // Distance
        String distanceStr;
        if (obj.distance > 0) {
            distanceStr = String.format(Locale.US, "~%.1f METERS", obj.distance);
        } else {
            distanceStr = "UNKNOWN";
        }
        distanceView.setText(String.format(Locale.US, "DISTANCE: %s", distanceStr));

        // Hazard level
        String hazardStr = NativeConstants.hazardLevelToString(obj.hazardLevel);
        hazardView.setText(String.format(Locale.US, "HAZARD: %s", hazardStr));

        // Color-code hazard level
        switch (obj.hazardLevel) {
            case NativeConstants.HAZARD_HIGH:
                hazardView.setTextColor(Color.rgb(255, 50, 50));
                break;
            case NativeConstants.HAZARD_MEDIUM:
                hazardView.setTextColor(Color.rgb(255, 180, 0));
                break;
            case NativeConstants.HAZARD_LOW:
                hazardView.setTextColor(Color.rgb(200, 200, 0));
                break;
            default:
                hazardView.setTextColor(TERMINAL_GREEN);
                break;
        }

        // Confidence
        String confStr = String.format(Locale.US, "CONFIDENCE: %.0f%%",
                obj.confidence * 100f);
        confidenceView.setText(confStr);

        // Additional info
        StringBuilder additional = new StringBuilder();
        additional.append("ID: ").append(obj.id);

        // Bounding box info
        additional.append(String.format(Locale.US,
                "\nBBOX: [%.2f, %.2f, %.2f, %.2f]",
                obj.bbox.left, obj.bbox.top, obj.bbox.right, obj.bbox.bottom));

        // Segmentation mask availability
        if (obj.segmentationMask != null) {
            additional.append("\nSEGMENTATION: AVAILABLE");
        }

        additionalView.setText(String.format(Locale.US, "NOTES:\n%s",
                additional.toString().toUpperCase(Locale.US)));
    }

    // ================================================================
    // Swipe-to-dismiss
    // ================================================================

    private float touchStartX = 0f;
    private boolean isSwiping = false;

    @Override
    public boolean onTouchEvent(@NonNull MotionEvent event) {
        float x = event.getX();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                touchStartX = x;
                isSwiping = false;
                return true;

            case MotionEvent.ACTION_MOVE:
                float dx = x - touchStartX;
                if (dx < -50f) {
                    isSwiping = true;
                    // Translate the panel with the finger
                    setTranslationX(Math.max(dx, -getWidth()));
                }
                return true;

            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                if (isSwiping && (touchStartX - x) > getWidth() * 0.3f) {
                    // Swipe was far enough — dismiss
                    hide();
                } else {
                    // Snap back
                    setTranslationX(0f);
                }
                isSwiping = false;
                return true;
        }

        return super.onTouchEvent(event);
    }

    // ================================================================
    // Listener
    // ================================================================

    /**
     * Sets a callback to be invoked when the panel is closed.
     *
     * @param listener The close listener, or null to remove.
     */
    public void setOnCloseListener(@Nullable Runnable listener) {
        this.onCloseListener = listener;
    }

    // ================================================================
    // Utility
    // ================================================================

    /**
     * Converts dp to pixels.
     */
    private int dpToPx(int dp) {
        float density = getResources().getDisplayMetrics().density;
        return Math.round(dp * density);
    }
}
