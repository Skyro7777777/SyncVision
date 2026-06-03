/**
 * GreenTheme.java
 *
 * Constants and helper methods for the terminal green visual theme
 * used throughout the Sync Vision app. This theme emulates the
 * E.D.I.T.H-style HUD with a classic CRT terminal aesthetic:
 * dark backgrounds, bright green outlines, scanline effects, and
 * monospace typography.
 *
 * Color Palette:
 *   Primary:   #00FF41 (terminal green — the signature outline color)
 *   Dim:       #00CC33 (secondary text, borders, dim elements)
 *   Bright:    #33FF66 (highlights, labels, interactive elements)
 *   Background:#0A0A0A (almost black — camera feed base)
 *   Panel:     #111111 (semi-transparent dark panels for text)
 *
 * Hazard Colors:
 *   HIGH:   #FF3333 (red — immediate danger)
 *   MEDIUM: #FF9900 (orange — moderate threat)
 *   LOW:    #FFFF00 (yellow — minor caution)
 *
 * Rendering Parameters:
 *   Scanline alpha, outline thickness, font size range — all tuned
 *   for readability on mobile screens in various lighting conditions.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.util
 * Target SDK: 29+
 */

package com.syncvision.app.util;

import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;

import androidx.annotation.ColorInt;
import androidx.annotation.NonNull;

/**
 * Constants and Paint factory methods for the terminal green theme.
 * This class centralizes all visual theme parameters so that the
 * rendering pipeline, overlay views, and UI components share a
 * consistent visual language.
 * <p>
 * This is a constants-only class — do not instantiate.
 */
public final class GreenTheme {

    // Private constructor — constants only
    private GreenTheme() {
        throw new AssertionError("GreenTheme is a constants class; do not instantiate.");
    }

    // ================================================================
    // Primary Colors
    // ================================================================

    /**
     * Terminal green — the signature outline and text color.
     * Classic CRT phosphor green (#00FF41).
     * Used for: object outlines, primary labels, HUD text.
     */
    @ColorInt
    public static final int COLOR_GREEN = 0xFF00FF41;

    /**
     * Dimmer green for secondary text and subtle elements.
     * Slightly darker than terminal green (#00CC33).
     * Used for: secondary info, border glow, inactive elements.
     */
    @ColorInt
    public static final int COLOR_GREEN_DIM = 0xFF00CC33;

    /**
     * Brighter green for highlights and emphasis.
     * Lighter than terminal green (#33FF66).
     * Used for: active labels, interactive highlights, scan markers.
     */
    @ColorInt
    public static final int COLOR_GREEN_BRIGHT = 0xFF33FF66;

    // ================================================================
    // Background Colors
    // ================================================================

    /**
     * Almost black background for the camera feed base (#0A0A0A).
     * Not pure black to avoid harsh contrast with camera imagery.
     */
    @ColorInt
    public static final int COLOR_DARK_BG = 0xFF0A0A0A;

    /**
     * Semi-transparent dark panel for text backgrounds (#CC111111).
     * Alpha = 0xCC (80%) — dark enough for readability but preserves
     * context of the camera feed beneath.
     * Used for: info panels, label backgrounds, HUD sections.
     */
    @ColorInt
    public static final int COLOR_DARK_PANEL = 0xCC111111;

    // ================================================================
    // Hazard Level Colors
    // ================================================================

    /**
     * Red for high threat / danger (#FF3333).
     * Used for: HIGH hazard outlines, danger indicators, alerts.
     */
    @ColorInt
    public static final int COLOR_HAZARD_HIGH = 0xFFFF3333;

    /**
     * Orange for medium threat / warning (#FF9900).
     * Used for: MEDIUM hazard outlines, caution indicators.
     */
    @ColorInt
    public static final int COLOR_HAZARD_MEDIUM = 0xFFFF9900;

    /**
     * Yellow for low threat / caution (#FFFF00).
     * Used for: LOW hazard outlines, informational warnings.
     */
    @ColorInt
    public static final int COLOR_HAZARD_LOW = 0xFFFFFF00;

    // ================================================================
    // Typography
    // ================================================================

    /**
     * Monospace font family name for the terminal aesthetic.
     * All labels, HUD text, and info panels use monospace for
     * the CRT terminal look and consistent character widths.
     */
    public static final String FONT_MONOSPACE = "monospace";

    // ================================================================
    // Visual Effect Parameters
    // ================================================================

    /**
     * Alpha value for the CRT scanline overlay effect.
     * At 0.03 (3%), scanlines are subtly visible without
     * significantly reducing camera feed brightness.
     */
    public static final float SCANLINE_ALPHA = 0.03f;

    /**
     * Outline thickness in pixels for the green object outlines.
     * 2.0px provides visibility without overwhelming the object.
     * Scales with display density in the rendering pipeline.
     */
    public static final float OUTLINE_THICKNESS = 2.0f;

    /**
     * Minimum label font size in scaled pixels (sp).
     * Used for distant (far) objects where labels should be small.
     */
    public static final int LABEL_FONT_SIZE_MIN = 12;

    /**
     * Maximum label font size in scaled pixels (sp).
     * Used for nearby (close) objects where labels should be large.
     */
    public static final int LABEL_FONT_SIZE_MAX = 24;

    // ================================================================
    // Paint Factory Methods
    // ================================================================

    /**
     * Creates a Paint configured for primary green text/outlines.
     * Settings: ANTI_ALIAS, monospace, terminal green color,
     * stroke style with OUTLINE_THICKNESS.
     *
     * @return Configured Paint for primary green rendering.
     */
    @NonNull
    public static Paint getGreenPaint() {
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setColor(COLOR_GREEN);
        paint.setTypeface(Typeface.MONOSPACE);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(OUTLINE_THICKNESS);
        return paint;
    }

    /**
     * Creates a Paint configured for dim green text/outlines.
     * Settings: ANTI_ALIAS, monospace, dim green color,
     * stroke style with OUTLINE_THICKNESS.
     *
     * @return Configured Paint for secondary/dim green rendering.
     */
    @NonNull
    public static Paint getDimGreenPaint() {
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setColor(COLOR_GREEN_DIM);
        paint.setTypeface(Typeface.MONOSPACE);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(OUTLINE_THICKNESS);
        return paint;
    }

    /**
     * Creates a Paint configured for bright green text/outlines.
     * Settings: ANTI_ALIAS, monospace, bright green color,
     * stroke style with OUTLINE_THICKNESS.
     *
     * @return Configured Paint for highlight/bright green rendering.
     */
    @NonNull
    public static Paint getBrightGreenPaint() {
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setColor(COLOR_GREEN_BRIGHT);
        paint.setTypeface(Typeface.MONOSPACE);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(OUTLINE_THICKNESS);
        return paint;
    }

    // ================================================================
    // Utility Methods
    // ================================================================

    /**
     * Returns the terminal green color with a custom alpha value.
     *
     * @param alpha Alpha from 0.0 (transparent) to 1.0 (opaque).
     * @return ARGB color int.
     */
    @ColorInt
    public static int getGreenWithAlpha(float alpha) {
        int a = Math.max(0, Math.min(255, Math.round(alpha * 255)));
        return (a << 24) | (COLOR_GREEN & 0x00FFFFFF);
    }

    /**
     * Returns the hazard color for a given integer hazard level.
     *
     * @param hazardLevel Hazard level (0-3).
     * @return ARGB color int matching the hazard level.
     */
    @ColorInt
    public static int getHazardColor(int hazardLevel) {
        switch (hazardLevel) {
            case 3:  return COLOR_HAZARD_HIGH;
            case 2:  return COLOR_HAZARD_MEDIUM;
            case 1:  return COLOR_HAZARD_LOW;
            default: return COLOR_GREEN;
        }
    }

    /**
     * Returns a human-readable hex string for a color int.
     * Useful for logging and debugging.
     *
     * @param color ARGB color int.
     * @return Hex string like "#00FF41".
     */
    @NonNull
    public static String colorToHex(@ColorInt int color) {
        return String.format("#%06X", color & 0x00FFFFFF);
    }
}
