/**
 * ObjectDescription.java
 *
 * Room Entity representing a pre-populated description entry for a known
 * object class. These descriptions provide the Sync Vision app with rich
 * contextual information about detected objects — including short
 * descriptions, common uses, hazard levels, and categories.
 *
 * The data is loaded from assets/data/object_descriptions.json on first
 * launch and cached in the Room database for fast offline access.
 *
 * Example entry:
 *   className="car", displayName="CAR",
 *   shortDescription="MOTOR VEHICLE WITH FOUR WHEELS\nUSED FOR TRANSPORTATION ON ROADS",
 *   commonUses="PERSONAL TRANSPORT, CARGO DELIVERY, EMERGENCY SERVICES",
 *   hazardLevel=3, category="VEHICLE"
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.data
 * Target SDK: 29+
 */

package com.syncvision.app.data;

import androidx.annotation.NonNull;
import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.PrimaryKey;

/**
 * Room Entity for the object_descriptions table. Stores contextual
 * descriptions for known object classes, enabling the app to display
 * rich information when an object is detected by the ML pipeline.
 * <p>
 * The primary key is the auto-generated id, while className serves
 * as the natural lookup key for queries from the detection pipeline.
 */
@Entity(tableName = "object_descriptions")
public class ObjectDescription {

    // ================================================================
    // Fields
    // ================================================================

    /** Auto-generated primary key. */
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    private int id;

    /**
     * Machine-readable class name matching COCO/ML model output.
     * Example: "car", "person", "potted_plant"
     * Used as the primary lookup key via DAO queries.
     */
    @ColumnInfo(name = "className")
    @NonNull
    private String className;

    /**
     * Human-readable display name in ALL CAPS for terminal aesthetic.
     * Example: "CAR", "PERSON", "POTTED PLANT"
     */
    @ColumnInfo(name = "displayName")
    @NonNull
    private String displayName;

    /**
     * Short 2-3 line description in ALL CAPS for HUD display.
     * Newline characters (\n) separate lines.
     * Example: "MOTOR VEHICLE WITH FOUR WHEELS\nUSED FOR TRANSPORTATION ON ROADS"
     */
    @ColumnInfo(name = "shortDescription")
    @NonNull
    private String shortDescription;

    /**
     * Common uses in ALL CAPS, comma-separated.
     * Example: "PERSONAL TRANSPORT, CARGO DELIVERY, EMERGENCY SERVICES"
     */
    @ColumnInfo(name = "commonUses")
    @NonNull
    private String commonUses;

    /**
     * Hazard level (0-3) matching HazardLevel enum values:
     *   0 = NONE (safe), 1 = LOW (caution),
     *   2 = MEDIUM (warning), 3 = HIGH (danger)
     */
    @ColumnInfo(name = "hazardLevel")
    private int hazardLevel;

    /**
     * Object category for grouping and filtering.
     * Example: "VEHICLE", "ANIMAL", "FURNITURE", "ELECTRONICS"
     */
    @ColumnInfo(name = "category")
    @NonNull
    private String category;

    // ================================================================
    // Constructor
    // ================================================================

    /**
     * Creates a new ObjectDescription with all fields.
     * Used by Room and by the JSON pre-population loader.
     *
     * @param className        Machine-readable class name.
     * @param displayName      Human-readable display name (ALL CAPS).
     * @param shortDescription 2-3 line short description (ALL CAPS).
     * @param commonUses       Common uses (ALL CAPS, comma-separated).
     * @param hazardLevel      Hazard level (0-3).
     * @param category         Object category.
     */
    public ObjectDescription(@NonNull String className,
                             @NonNull String displayName,
                             @NonNull String shortDescription,
                             @NonNull String commonUses,
                             int hazardLevel,
                             @NonNull String category) {
        this.className = className;
        this.displayName = displayName;
        this.shortDescription = shortDescription;
        this.commonUses = commonUses;
        this.hazardLevel = hazardLevel;
        this.category = category;
    }

    // ================================================================
    // Getters and Setters
    // ================================================================

    /** Returns the auto-generated row ID. */
    public int getId() {
        return id;
    }

    /** Sets the auto-generated row ID (used by Room). */
    public void setId(int id) {
        this.id = id;
    }

    /** Returns the machine-readable class name (e.g., "car"). */
    @NonNull
    public String getClassName() {
        return className;
    }

    /** Sets the machine-readable class name. */
    public void setClassName(@NonNull String className) {
        this.className = className;
    }

    /** Returns the display name in ALL CAPS (e.g., "CAR"). */
    @NonNull
    public String getDisplayName() {
        return displayName;
    }

    /** Sets the display name. */
    public void setDisplayName(@NonNull String displayName) {
        this.displayName = displayName;
    }

    /** Returns the short description (2-3 lines, ALL CAPS). */
    @NonNull
    public String getShortDescription() {
        return shortDescription;
    }

    /** Sets the short description. */
    public void setShortDescription(@NonNull String shortDescription) {
        this.shortDescription = shortDescription;
    }

    /** Returns the common uses string (comma-separated, ALL CAPS). */
    @NonNull
    public String getCommonUses() {
        return commonUses;
    }

    /** Sets the common uses string. */
    public void setCommonUses(@NonNull String commonUses) {
        this.commonUses = commonUses;
    }

    /** Returns the hazard level (0-3). */
    public int getHazardLevel() {
        return hazardLevel;
    }

    /** Sets the hazard level (0-3). */
    public void setHazardLevel(int hazardLevel) {
        this.hazardLevel = hazardLevel;
    }

    /** Returns the object category (e.g., "VEHICLE"). */
    @NonNull
    public String getCategory() {
        return category;
    }

    /** Sets the object category. */
    public void setCategory(@NonNull String category) {
        this.category = category;
    }

    // ================================================================
    // Utility Methods
    // ================================================================

    /**
     * Returns the hazard level as a type-safe HazardLevel enum.
     *
     * @return HazardLevel enum corresponding to the integer hazardLevel.
     */
    @NonNull
    public HazardLevel getHazardLevelEnum() {
        return HazardLevel.fromInt(hazardLevel);
    }

    /**
     * Returns a multi-line info string suitable for the InfoPanelView.
     * Format: DISPLAYNAME\nSHORT_DESCRIPTION\nUSES: COMMON_USES
     *
     * @return Formatted info string.
     */
    @NonNull
    public String toInfoString() {
        StringBuilder sb = new StringBuilder();
        sb.append(displayName).append("\n");
        sb.append(shortDescription).append("\n");
        sb.append("USES: ").append(commonUses);
        return sb.toString();
    }

    @NonNull
    @Override
    public String toString() {
        return "ObjectDescription{" +
                "id=" + id +
                ", className='" + className + '\'' +
                ", displayName='" + displayName + '\'' +
                ", hazardLevel=" + hazardLevel +
                ", category='" + category + '\'' +
                '}';
    }
}
