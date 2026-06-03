/**
 * OverlayView.java
 *
 * Canvas-based overlay for HUD (Heads-Up Display) elements in the
 * Sync Vision app. Draws all 2D UI elements directly onto the camera
 * view using the Android Canvas API, overlaying the OpenGL-rendered
 * camera feed and ML effects.
 *
 * HUD elements drawn:
 *   - Weather status text (top-left)
 *   - Face count (top-right)
 *   - FPS counter (bottom-left)
 *   - Threat indicator (pulsing icon, bottom-left)
 *   - Mode indicators (bottom-center)
 *   - Scanline effect overlay (full-screen CRT effect)
 *   - Corner brackets (HUD aesthetic)
 *
 * All text is rendered in monospace, ALL CAPS, terminal green (#00FF41)
 * on semi-transparent dark backgrounds for readability.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.ui
 * Target SDK: 29+
 */

package com.syncvision.app.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.Locale;

/**
 * Canvas-based HUD overlay view. Draws all heads-up display elements
 * on top of the camera feed using direct Canvas rendering.
 * <p>
 * This view is transparent except for the drawn HUD elements, allowing
 * the OpenGL-rendered camera feed and ML overlays to show through.
 * <p>
 * The HUD aesthetic follows the E.D.I.T.H-style terminal green theme:
 *   - Monospace font, ALL CAPS
 *   - Terminal green #00FF41
 *   - Semi-transparent dark panel backgrounds
 *   - Corner bracket decorations
 *   - CRT scanline effect
 */
public class OverlayView extends View {

    private static final String TAG = "SV-OverlayView";

    // ================================================================
    // Visual Constants
    // ================================================================

    /** Terminal green (#00FF41). */
    private static final int TERMINAL_GREEN = Color.rgb(0, 255, 65);

    /** Dim terminal green for secondary elements. */
    private static final int DIM_GREEN = Color.argb(160, 0, 180, 45);

    /** Bright terminal green for emphasis. */
    private static final int BRIGHT_GREEN = Color.argb(255, 50, 255, 100);

    /** Red for hazard/threat indicators. */
    private static final int HAZARD_RED = Color.rgb(255, 50, 50);

    /** Orange for medium threat. */
    private static final int HAZARD_ORANGE = Color.rgb(255, 180, 0);

    /** Yellow for low threat. */
    private static final int HAZARD_YELLOW = Color.rgb(200, 200, 0);

    /** Semi-transparent dark background for text panels. */
    private static final int PANEL_BG = Color.argb(140, 0, 8, 4);

    /** Scanline darkening color. */
    private static final int SCANLINE_COLOR = Color.argb(12, 0, 0, 0);

    /** Corner bracket color. */
    private static final int BRACKET_COLOR = Color.argb(120, 0, 255, 65);

    // ================================================================
    // Layout Constants (in dp, converted at draw time)
    // ================================================================

    /** HUD text size in sp. */
    private static final float HUD_TEXT_SIZE = 12f;

    /** Small HUD text size in sp. */
    private static final float HUD_TEXT_SIZE_SMALL = 9f;

    /** Title text size in sp. */
    private static final float HUD_TITLE_SIZE = 11f;

    /** Corner bracket length in dp. */
    private static final float BRACKET_LENGTH = 30f;

    /** Corner bracket thickness in dp. */
    private static final float BRACKET_THICKNESS = 2f;

    /** Panel corner radius in dp. */
    private static final float PANEL_RADIUS = 4f;

    /** Panel padding in dp. */
    private static final float PANEL_PAD = 6f;

    /** Margin from screen edge in dp. */
    private static final float MARGIN = 12f;

    /** Scanline spacing in pixels. */
    private static final int SCANLINE_SPACING = 3;

    // ================================================================
    // State
    // ================================================================

    /** Current FPS value. */
    private float fps = 0f;

    /** Current weather status string. */
    @NonNull
    private String weatherStatus = "WEATHER: INITIALIZING...";

    /** Current face count. */
    private int faceCount = 0;

    /** Current threat level (0-3). */
    private int threatLevel = 0;

    /** Current operating mode. */
    @NonNull
    private MainActivity.OperatingMode operatingMode = MainActivity.OperatingMode.SCAN;

    /** Scene summary text. */
    @NonNull
    private String sceneSummary = "";

    /** Whether flash is enabled. */
    private boolean flashEnabled = false;

    /** Animation time for pulsing effects. */
    private long animTimeMs = 0;

    /** Scanline intensity [0, 1]. */
    private float scanlineIntensity = 0.3f;

    // ================================================================
    // Paints (lazily initialized)
    // ================================================================

    private Paint hudTextPaint;
    private Paint hudTextSmallPaint;
    private Paint hudTitlePaint;
    private Paint panelBgPaint;
    private Paint bracketPaint;
    private Paint scanlinePaint;
    private Paint threatPaint;
    private Paint modeButtonPaint;
    private Paint modeActivePaint;
    private Paint modeTextPaint;
    private Paint modeActiveTextPaint;

    // ================================================================
    // Constructors
    // ================================================================

    public OverlayView(@NonNull Context context) {
        this(context, null);
    }

    public OverlayView(@NonNull Context context, @Nullable AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public OverlayView(@NonNull Context context, @Nullable AttributeSet attrs,
                       int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        initPaints();

        // Enable continuous invalidation for animation effects
        setWillNotDraw(false);

        Log.d(TAG, "OverlayView initialized");
    }

    // ================================================================
    // Paint Initialization
    // ================================================================

    /**
     * Initializes all Paint objects used for drawing HUD elements.
     */
    private void initPaints() {
        // Main HUD text paint (terminal green, monospace, ALL CAPS)
        hudTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        hudTextPaint.setColor(TERMINAL_GREEN);
        hudTextPaint.setTypeface(Typeface.MONOSPACE);
        hudTextPaint.setTextSize(HUD_TEXT_SIZE
                * getResources().getDisplayMetrics().scaledDensity);

        // Small HUD text paint
        hudTextSmallPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        hudTextSmallPaint.setColor(DIM_GREEN);
        hudTextSmallPaint.setTypeface(Typeface.MONOSPACE);
        hudTextSmallPaint.setTextSize(HUD_TEXT_SIZE_SMALL
                * getResources().getDisplayMetrics().scaledDensity);

        // Title/label text paint
        hudTitlePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        hudTitlePaint.setColor(TERMINAL_GREEN);
        hudTitlePaint.setTypeface(Typeface.MONOSPACE);
        hudTitlePaint.setTextSize(HUD_TITLE_SIZE
                * getResources().getDisplayMetrics().scaledDensity);

        // Panel background paint
        panelBgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        panelBgPaint.setColor(PANEL_BG);
        panelBgPaint.setStyle(Paint.Style.FILL);

        // Corner bracket paint
        bracketPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        bracketPaint.setColor(BRACKET_COLOR);
        bracketPaint.setStyle(Paint.Style.STROKE);
        bracketPaint.setStrokeWidth(BRACKET_THICKNESS
                * getResources().getDisplayMetrics().density);

        // Scanline paint
        scanlinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        scanlinePaint.setColor(SCANLINE_COLOR);
        scanlinePaint.setStyle(Paint.Style.FILL);

        // Threat indicator paint (changes color based on level)
        threatPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        threatPaint.setTypeface(Typeface.MONOSPACE);
        threatPaint.setTextSize(HUD_TEXT_SIZE
                * getResources().getDisplayMetrics().scaledDensity);

        // Mode button background paint
        modeButtonPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        modeButtonPaint.setColor(Color.argb(100, 0, 40, 15));
        modeButtonPaint.setStyle(Paint.Style.FILL);

        // Active mode button background
        modeActivePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        modeActivePaint.setColor(Color.argb(180, 0, 80, 25));
        modeActivePaint.setStyle(Paint.Style.FILL);

        // Mode button text
        modeTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        modeTextPaint.setColor(DIM_GREEN);
        modeTextPaint.setTypeface(Typeface.MONOSPACE);
        modeTextPaint.setTextSize(HUD_TEXT_SIZE_SMALL
                * getResources().getDisplayMetrics().scaledDensity);
        modeTextPaint.setTextAlign(Paint.Align.CENTER);

        // Active mode button text
        modeActiveTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        modeActiveTextPaint.setColor(TERMINAL_GREEN);
        modeActiveTextPaint.setTypeface(Typeface.MONOSPACE);
        modeActiveTextPaint.setTextSize(HUD_TEXT_SIZE_SMALL
                * getResources().getDisplayMetrics().scaledDensity);
        modeActiveTextPaint.setTextAlign(Paint.Align.CENTER);
    }

    // ================================================================
    // Public Setters
    // ================================================================

    /** Sets the current FPS value. */
    public void setFps(float fps) {
        this.fps = fps;
    }

    /** Sets the weather status string. */
    public void setWeatherStatus(@NonNull String status) {
        this.weatherStatus = status.toUpperCase(Locale.US);
    }

    /** Sets the detected face count. */
    public void setFaceCount(int count) {
        this.faceCount = count;
    }

    /** Sets the threat level (0-3). */
    public void setThreatLevel(int level) {
        this.threatLevel = level;
    }

    /** Sets the current operating mode. */
    public void setOperatingMode(@NonNull MainActivity.OperatingMode mode) {
        this.operatingMode = mode;
    }

    /** Sets the scene summary string. */
    public void setSceneSummary(@NonNull String summary) {
        this.sceneSummary = summary.toUpperCase(Locale.US);
    }

    /** Sets whether flash is enabled. */
    public void setFlashEnabled(boolean enabled) {
        this.flashEnabled = enabled;
    }

    /** Sets the scanline effect intensity [0, 1]. */
    public void setScanlineIntensity(float intensity) {
        this.scanlineIntensity = Math.max(0f, Math.min(1f, intensity));
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

        animTimeMs = System.currentTimeMillis();
        float density = getResources().getDisplayMetrics().density;
        float margin = MARGIN * density;
        float pad = PANEL_PAD * density;
        float bracketLen = BRACKET_LENGTH * density;

        // ---- 1. Draw corner brackets (HUD aesthetic) ----
        drawCornerBrackets(canvas, w, h, bracketLen);

        // ---- 2. Draw weather status (top-left) ----
        drawWeatherStatus(canvas, w, h, margin, pad, density);

        // ---- 3. Draw face count (top-right) ----
        drawFaceCount(canvas, w, h, margin, pad, density);

        // ---- 4. Draw FPS counter (bottom-left) ----
        drawFpsCounter(canvas, w, h, margin, pad, density);

        // ---- 5. Draw threat indicator (bottom-left, below FPS) ----
        drawThreatIndicator(canvas, w, h, margin, pad, density);

        // ---- 6. Draw mode toggle buttons (bottom-center) ----
        drawModeButtons(canvas, w, h, density);

        // ---- 7. Draw flash indicator (top-right, below face count) ----
        if (flashEnabled) {
            drawFlashIndicator(canvas, w, h, margin, pad, density);
        }

        // ---- 8. Draw scanline effect overlay ----
        drawScanlines(canvas, w, h);

        // Schedule next frame for animation
        invalidate();
    }

    // ================================================================
    // Individual HUD Element Drawing Methods
    // ================================================================

    /**
     * Draws decorative corner brackets around the screen edges.
     */
    private void drawCornerBrackets(@NonNull Canvas canvas, int w, int h, float len) {
        float inset = 6f * getResources().getDisplayMetrics().density;

        // Top-left bracket
        canvas.drawLine(inset, inset, inset + len, inset, bracketPaint);
        canvas.drawLine(inset, inset, inset, inset + len, bracketPaint);

        // Top-right bracket
        canvas.drawLine(w - inset - len, inset, w - inset, inset, bracketPaint);
        canvas.drawLine(w - inset, inset, w - inset, inset + len, bracketPaint);

        // Bottom-left bracket
        canvas.drawLine(inset, h - inset, inset + len, h - inset, bracketPaint);
        canvas.drawLine(inset, h - inset - len, inset, h - inset, bracketPaint);

        // Bottom-right bracket
        canvas.drawLine(w - inset - len, h - inset, w - inset, h - inset, bracketPaint);
        canvas.drawLine(w - inset, h - inset - len, w - inset, h - inset, bracketPaint);
    }

    /**
     * Draws the weather status text in the top-left corner.
     */
    private void drawWeatherStatus(@NonNull Canvas canvas, int w, int h,
                                   float margin, float pad, float density) {
        String text = weatherStatus;
        float textWidth = hudTitlePaint.measureText(text);
        float textHeight = hudTitlePaint.getTextSize();

        RectF bgRect = new RectF(margin, margin,
                margin + textWidth + pad * 2, margin + textHeight + pad * 2);
        canvas.drawRoundRect(bgRect, PANEL_RADIUS * density, PANEL_RADIUS * density, panelBgPaint);
        canvas.drawText(text, margin + pad, margin + pad + textHeight, hudTitlePaint);
    }

    /**
     * Draws the face count in the top-right corner.
     */
    private void drawFaceCount(@NonNull Canvas canvas, int w, int h,
                               float margin, float pad, float density) {
        String text = String.format(Locale.US, "%d FACES DETECTED", faceCount);
        float textWidth = hudTitlePaint.measureText(text);
        float textHeight = hudTitlePaint.getTextSize();

        float left = w - margin - textWidth - pad * 2;
        RectF bgRect = new RectF(left, margin,
                w - margin, margin + textHeight + pad * 2);
        canvas.drawRoundRect(bgRect, PANEL_RADIUS * density, PANEL_RADIUS * density, panelBgPaint);
        canvas.drawText(text, left + pad, margin + pad + textHeight, hudTitlePaint);
    }

    /**
     * Draws the FPS counter in the bottom-left corner.
     */
    private void drawFpsCounter(@NonNull Canvas canvas, int w, int h,
                                float margin, float pad, float density) {
        String text = String.format(Locale.US, "FPS: %.0f", fps);
        float textWidth = hudTextPaint.measureText(text);
        float textHeight = hudTextPaint.getTextSize();

        float top = h - margin - textHeight - pad * 2 - 30 * density;
        RectF bgRect = new RectF(margin, top,
                margin + textWidth + pad * 2, top + textHeight + pad * 2);
        canvas.drawRoundRect(bgRect, PANEL_RADIUS * density, PANEL_RADIUS * density, panelBgPaint);
        canvas.drawText(text, margin + pad, top + pad + textHeight, hudTextPaint);
    }

    /**
     * Draws the threat level indicator with a pulsing animation.
     */
    private void drawThreatIndicator(@NonNull Canvas canvas, int w, int h,
                                     float margin, float pad, float density) {
        if (threatLevel == 0) return;

        // Pulsing alpha animation
        float pulsePhase = (float) (Math.sin(animTimeMs / 300.0) + 1.0) / 2.0f;
        int alpha = (int) (120 + 135 * pulsePhase);

        String threatText;
        int threatColor;
        switch (threatLevel) {
            case 3:
                threatText = "!!! THREAT: HIGH !!!";
                threatColor = Color.argb(alpha, 255, 50, 50);
                break;
            case 2:
                threatText = "! THREAT: MEDIUM !";
                threatColor = Color.argb(alpha, 255, 180, 0);
                break;
            case 1:
                threatText = "THREAT: LOW";
                threatColor = Color.argb(Math.min(255, alpha + 40), 200, 200, 0);
                break;
            default:
                return;
        }

        threatPaint.setColor(threatColor);
        float textHeight = threatPaint.getTextSize();
        float textWidth = threatPaint.measureText(threatText);

        float top = h - margin - textHeight - pad * 2;
        RectF bgRect = new RectF(margin, top,
                margin + textWidth + pad * 2, top + textHeight + pad * 2);

        // Pulsing background
        Paint threatBg = new Paint(Paint.ANTI_ALIAS_FLAG);
        int bgAlpha = (int) (80 + 80 * pulsePhase);
        if (threatLevel >= 3) {
            threatBg.setColor(Color.argb(bgAlpha, 60, 0, 0));
        } else {
            threatBg.setColor(Color.argb(bgAlpha, 40, 30, 0));
        }
        canvas.drawRoundRect(bgRect, PANEL_RADIUS * density, PANEL_RADIUS * density, threatBg);
        canvas.drawText(threatText, margin + pad, top + pad + textHeight, threatPaint);
    }

    /**
     * Draws the mode toggle buttons at the bottom-center of the screen.
     */
    private void drawModeButtons(@NonNull Canvas canvas, int w, int h, float density) {
        float btnWidth = 70f * density;
        float btnHeight = 28f * density;
        float btnMargin = 4f * density;
        float totalWidth = btnWidth * 3 + btnMargin * 2;
        float startX = (w - totalWidth) / 2f;
        float startY = h - btnHeight - 12f * density;

        String[] modes = {"SCAN", "PATH", "ID"};
        MainActivity.OperatingMode[] modeValues = {
                MainActivity.OperatingMode.SCAN,
                MainActivity.OperatingMode.PATH,
                MainActivity.OperatingMode.IDENTIFY
        };

        for (int i = 0; i < 3; i++) {
            float left = startX + i * (btnWidth + btnMargin);
            float top = startY;
            float right = left + btnWidth;
            float bottom = top + btnHeight;

            boolean isActive = (operatingMode == modeValues[i]);

            // Button background
            RectF btnRect = new RectF(left, top, right, bottom);
            canvas.drawRoundRect(btnRect, 4f * density, 4f * density,
                    isActive ? modeActivePaint : modeButtonPaint);

            // Border
            Paint borderPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            borderPaint.setColor(isActive ? TERMINAL_GREEN : DIM_GREEN);
            borderPaint.setStyle(Paint.Style.STROKE);
            borderPaint.setStrokeWidth(1f * density);
            canvas.drawRoundRect(btnRect, 4f * density, 4f * density, borderPaint);

            // Button text
            Paint textPaint = isActive ? modeActiveTextPaint : modeTextPaint;
            float textX = left + btnWidth / 2f;
            float textY = top + btnHeight / 2f
                    + modeTextPaint.getTextSize() / 3f;
            canvas.drawText(modes[i], textX, textY, textPaint);
        }
    }

    /**
     * Draws a flash indicator when the torch is enabled.
     */
    private void drawFlashIndicator(@NonNull Canvas canvas, int w, int h,
                                    float margin, float pad, float density) {
        String text = "FLASH: ON";
        float textWidth = hudTextSmallPaint.measureText(text);
        float textHeight = hudTextSmallPaint.getTextSize();

        float top = margin + hudTitlePaint.getTextSize() + pad * 3;
        float left = w - margin - textWidth - pad * 2;

        RectF bgRect = new RectF(left, top,
                left + textWidth + pad * 2, top + textHeight + pad * 2);
        canvas.drawRoundRect(bgRect, PANEL_RADIUS * density, PANEL_RADIUS * density, panelBgPaint);
        canvas.drawText(text, left + pad, top + pad + textHeight, hudTextSmallPaint);
    }

    /**
     * Draws the CRT scanline effect overlay across the entire view.
     * The scanlines are thin horizontal darkening lines that simulate
     * a CRT monitor aesthetic.
     */
    private void drawScanlines(@NonNull Canvas canvas, int w, int h) {
        if (scanlineIntensity <= 0.01f) return;

        // Adjust scanline alpha based on intensity
        int alpha = (int) (12 * scanlineIntensity);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setColor(Color.argb(alpha, 0, 0, 0));
        paint.setStyle(Paint.Style.FILL);

        // Draw horizontal lines with spacing
        for (int y = 0; y < h; y += SCANLINE_SPACING) {
            canvas.drawRect(0, y, w, y + 1, paint);
        }
    }

    // ================================================================
    // Touch Mode Button Handling
    // ================================================================

    /**
     * Checks if a touch event falls within a mode button area and
     * returns the corresponding operating mode, or null.
     *
     * @param x Touch X coordinate.
     * @param y Touch Y coordinate.
     * @return The operating mode at the touch location, or null.
     */
    @Nullable
    public MainActivity.OperatingMode getModeAtTouch(float x, float y) {
        float density = getResources().getDisplayMetrics().density;
        float btnWidth = 70f * density;
        float btnHeight = 28f * density;
        float btnMargin = 4f * density;
        float totalWidth = btnWidth * 3 + btnMargin * 2;
        float startX = (getWidth() - totalWidth) / 2f;
        float startY = getHeight() - btnHeight - 12f * density;

        MainActivity.OperatingMode[] modes = {
                MainActivity.OperatingMode.SCAN,
                MainActivity.OperatingMode.PATH,
                MainActivity.OperatingMode.IDENTIFY
        };

        for (int i = 0; i < 3; i++) {
            float left = startX + i * (btnWidth + btnMargin);
            float right = left + btnWidth;

            if (x >= left && x <= right && y >= startY && y <= startY + btnHeight) {
                return modes[i];
            }
        }

        return null;
    }
}
