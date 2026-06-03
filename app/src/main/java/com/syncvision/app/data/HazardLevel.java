/**
 * HazardLevel.java
 *
 * Enum representing the four hazard severity levels used throughout the
 * Sync Vision app for threat assessment and UI color coding. Each level
 * maps to an integer value (matching NativeConstants), a display string
 * for the HUD, and a color from the GreenTheme palette.
 *
 * Levels:
 *   NONE   (0): Safe — no threat detected, green indicator
 *   LOW    (1): Caution — minor risk, yellow indicator
 *   MEDIUM (2): Warning — moderate risk, orange indicator
 *   HIGH   (3): Danger — significant risk, red indicator
 *
 * This enum provides type-safe hazard level handling across the data
 * layer, scene engine, and UI components, replacing raw int comparisons.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.data
 * Target SDK: 29+
 */

package com.syncvision.app.data;

import androidx.annotation.ColorInt;
import androidx.annotation.NonNull;

import com.syncvision.app.util.GreenTheme;

/**
 * Enum representing hazard severity levels with associated colors
 * and display strings. Provides type-safe hazard handling and
 * bidirectional conversion between integer and enum representations.
 */
public enum HazardLevel {

    /** No hazard — safe to traverse/approach. */
    NONE(0, "SAFE", GreenTheme.COLOR_HAZARD_LOW),

    /** Low hazard — minor risk, proceed with caution. */
    LOW(1, "CAUTION", GreenTheme.COLOR_HAZARD_LOW),

    /** Medium hazard — moderate risk, consider alternative route. */
    MEDIUM(2, "WARNING", GreenTheme.COLOR_HAZARD_MEDIUM),

    /** High hazard — significant risk, avoid if possible. */
    HIGH(3, "DANGER", GreenTheme.COLOR_HAZARD_HIGH);

    // ================================================================
    // Instance Fields
    // ================================================================

    /** Integer value matching NativeConstants hazard level indices. */
    private final int value;

    /** Display string for HUD and info panel rendering. */
    private final String displayString;

    /** Color from GreenTheme palette for this hazard level. */
    @ColorInt
    private final int color;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a HazardLevel enum constant.
     *
     * @param value         Integer value (0-3).
     * @param displayString Short label for UI display.
     * @param color         ARGB color for rendering.
     */
    HazardLevel(int value, @NonNull String displayString, @ColorInt int color) {
        this.value = value;
        this.displayString = displayString;
        this.color = color;
    }

    // ================================================================
    // Accessors
    // ================================================================

    /**
     * Returns the integer value of this hazard level.
     * Matches NativeConstants: HAZARD_NONE=0, HAZARD_LOW=1,
     * HAZARD_MEDIUM=2, HAZARD_HIGH=3.
     *
     * @return Integer value (0-3).
     */
    public int getValue() {
        return value;
    }

    /**
     * Returns the display string for this hazard level.
     * Suitable for HUD rendering: "SAFE", "CAUTION", "WARNING", "DANGER".
     *
     * @return Short display string (all caps).
     */
    @NonNull
    public String getDisplayString() {
        return displayString;
    }

    /**
     * Returns the ARGB color associated with this hazard level.
     * Colors are from the GreenTheme palette:
     *   NONE  → yellow (#FFFF00)
     *   LOW   → yellow (#FFFF00)
     *   MEDIUM → orange (#FF9900)
     *   HIGH  → red (#FF3333)
     *
     * @return Color as ARGB int.
     */
    @ColorInt
    public int getColor() {
        return color;
    }

    // ================================================================
    // Conversion Methods
    // ================================================================

    /**
     * Converts an integer hazard level to the corresponding enum value.
     * This is the primary method for converting from ML pipeline results,
     * NativeConstants, or database storage to the type-safe enum.
     *
     * @param value Integer hazard level (0-3).
     * @return Corresponding HazardLevel enum, or NONE for invalid values.
     */
    @NonNull
    public static HazardLevel fromInt(int value) {
        switch (value) {
            case 0:  return NONE;
            case 1:  return LOW;
            case 2:  return MEDIUM;
            case 3:  return HIGH;
            default:
                // Invalid values default to NONE (safe fallback)
                return NONE;
        }
    }

    /**
     * Returns true if this hazard level represents a threat
     * (LOW, MEDIUM, or HIGH — anything above NONE).
     *
     * @return True if hazard level > NONE.
     */
    public boolean isThreat() {
        return this != NONE;
    }

    /**
     * Returns true if this hazard level is at or above the given level.
     * Useful for threshold checks, e.g., hazard.atLeast(MEDIUM).
     *
     * @param minimum The minimum level to check against.
     * @return True if this level >= minimum.
     */
    public boolean atLeast(@NonNull HazardLevel minimum) {
        return this.value >= minimum.value;
    }

    /**
     * Returns the higher of two hazard levels.
     * Used for scene-level threat aggregation.
     *
     * @param a First hazard level.
     * @param b Second hazard level.
     * @return The higher of the two levels.
     */
    @NonNull
    public static HazardLevel max(@NonNull HazardLevel a, @NonNull HazardLevel b) {
        return a.value >= b.value ? a : b;
    }

    @NonNull
    @Override
    public String toString() {
        return displayString + "(" + value + ")";
    }
}
