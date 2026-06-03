/**
 * ObjectDao.java
 *
 * Room Data Access Object for the object_descriptions table.
 * Provides query methods for looking up object descriptions by
 * class name, retrieving all descriptions, inserting new entries,
 * and filtering by hazard level.
 *
 * Primary usage patterns:
 *   - ML pipeline queries by className after detection
 *   - DescriptionLoader bulk-inserts from JSON on first run
 *   - HazardScorer queries high-hazard objects for scene assessment
 *   - UI components query by className for InfoPanelView display
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.data
 * Target SDK: 29+
 */

package com.syncvision.app.data;

import androidx.annotation.NonNull;
import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;

import java.util.List;

/**
 * Room DAO for the object_descriptions table.
 * All query methods are synchronous and should be called on a
 * background thread (Room enforces this for non-LiveData queries).
 */
@Dao
public interface ObjectDao {

    // ================================================================
    // Lookup Queries
    // ================================================================

    /**
     * Looks up an object description by its machine-readable class name.
     * This is the primary query used by the ML pipeline after detection.
     * <p>
     * Example: query("car") → ObjectDescription with displayName="CAR"
     *
     * @param className The class name from the ML model output.
     * @return The matching ObjectDescription, or null if not found.
     */
    @Query("SELECT * FROM object_descriptions WHERE className = :className")
    ObjectDescription getByClassName(@NonNull String className);

    /**
     * Returns all object descriptions in the database.
     * Used by DescriptionLoader for validation and by settings
     * for browsing available descriptions.
     *
     * @return List of all ObjectDescription entries.
     */
    @Query("SELECT * FROM object_descriptions")
    @NonNull
    List<ObjectDescription> getAllDescriptions();

    /**
     * Returns all object descriptions with hazard level at or above
     * the specified minimum. Used by the HazardScorer for threat
     * assessment and scene-level hazard queries.
     * <p>
     * Example: query(3) returns all HIGH-hazard objects.
     *
     * @param minLevel Minimum hazard level (0-3).
     * @return List of matching ObjectDescription entries.
     */
    @Query("SELECT * FROM object_descriptions WHERE hazardLevel >= :minLevel")
    @NonNull
    List<ObjectDescription> getByMinHazardLevel(int minLevel);

    // ================================================================
    // Insert Operations
    // ================================================================

    /**
     * Inserts a single object description. If a row with the same
     * primary key already exists, it will be replaced.
     * Used for both initial bulk loading and runtime updates.
     *
     * @param description The ObjectDescription to insert or replace.
     */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    void insert(@NonNull ObjectDescription description);

    /**
     * Inserts multiple object descriptions in a single transaction.
     * If any row conflicts, it will be replaced. This is the preferred
     * method for bulk loading from JSON on first run.
     *
     * @param descriptions List of ObjectDescription entries to insert.
     */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    void insertAll(@NonNull List<ObjectDescription> descriptions);

    // ================================================================
    // Count Queries
    // ================================================================

    /**
     * Returns the total number of object descriptions in the database.
     * Used by DescriptionLoader to determine if pre-population is needed
     * (count == 0 means first run).
     *
     * @return Number of rows in the object_descriptions table.
     */
    @Query("SELECT COUNT(*) FROM object_descriptions")
    int getCount();
}
