/**
 * ObjectDatabase.java
 *
 * Room Database for the Sync Vision app's object description store.
 * Provides the ObjectDao for querying and inserting ObjectDescription
 * entities, and pre-populates the database from a JSON asset file
 * on first run using a RoomDatabase.Callback.
 *
 * The pre-population reads assets/data/object_descriptions.json,
 * parses it with Gson, and inserts all entries in a single
 * transaction during database creation. Subsequent opens skip
 * pre-population if data already exists.
 *
 * Database: sync_vision_objects.db (internal storage)
 * Version: 1
 * Entities: ObjectDescription
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.data
 * Target SDK: 29+
 */

package com.syncvision.app.data;

import android.content.Context;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;
import androidx.sqlite.db.SupportSQLiteDatabase;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Room Database providing access to the object_descriptions table.
 * <p>
 * Uses a singleton pattern to ensure only one database instance exists.
 * Pre-populates from assets/data/object_descriptions.json on first
 * creation via a RoomDatabase.Callback running on a dedicated IO executor.
 */
@Database(
        entities = {ObjectDescription.class},
        version = 1,
        exportSchema = false
)
public abstract class ObjectDatabase extends RoomDatabase {

    private static final String TAG = "SV-ObjectDatabase";

    /** Database file name in internal storage. */
    private static final String DATABASE_NAME = "sync_vision_objects.db";

    /** Path to the JSON pre-population file in assets. */
    private static final String ASSET_PATH = "data/object_descriptions.json";

    /** Single-threaded executor for database IO operations. */
    private static final ExecutorService DATABASE_EXECUTOR =
            Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "SV-ObjectDB");
                t.setPriority(Thread.NORM_PRIORITY - 1);
                t.setDaemon(true);
                return t;
            });

    /** Singleton instance — volatile for double-checked locking. */
    private static volatile ObjectDatabase sInstance;

    // ================================================================
    // Abstract DAO Provider
    // ================================================================

    /**
     * Returns the ObjectDao for querying and inserting ObjectDescription
     * entries. Room generates the implementation at compile time.
     *
     * @return ObjectDao instance.
     */
    @NonNull
    public abstract ObjectDao objectDao();

    // ================================================================
    // Singleton Access
    // ================================================================

    /**
     * Returns the singleton ObjectDatabase instance, creating it if needed.
     * Thread-safe via double-checked locking pattern.
     *
     * @param context Application or activity context.
     * @return The ObjectDatabase singleton.
     */
    @NonNull
    public static ObjectDatabase getInstance(@NonNull Context context) {
        if (sInstance == null) {
            synchronized (ObjectDatabase.class) {
                if (sInstance == null) {
                    sInstance = buildDatabase(context.getApplicationContext());
                    Log.i(TAG, "ObjectDatabase singleton created");
                }
            }
        }
        return sInstance;
    }

    /**
     * Builds the Room database with the pre-population callback.
     *
     * @param context Application context.
     * @return Built ObjectDatabase instance.
     */
    @NonNull
    private static ObjectDatabase buildDatabase(@NonNull Context context) {
        return Room.databaseBuilder(
                        context.getApplicationContext(),
                        ObjectDatabase.class,
                        DATABASE_NAME
                )
                .addCallback(new PrePopulateCallback(context.getApplicationContext()))
                .fallbackToDestructiveMigration()
                .build();
    }

    // ================================================================
    // Pre-Population Callback
    // ================================================================

    /**
     * RoomDatabase.Callback that pre-populates the database from
     * assets/data/object_descriptions.json when the database is
     * first created. Runs asynchronously on DATABASE_EXECUTOR to
     * avoid blocking the calling thread.
     * <p>
     * The JSON file format is expected to be an array of objects:
     * <pre>
     * [
     *   {
     *     "className": "car",
     *     "displayName": "CAR",
     *     "shortDescription": "MOTOR VEHICLE WITH FOUR WHEELS\nUSED FOR TRANSPORTATION ON ROADS",
     *     "commonUses": "PERSONAL TRANSPORT, CARGO DELIVERY",
     *     "hazardLevel": 3,
     *     "category": "VEHICLE"
     *   },
     *   ...
     * ]
     * </pre>
     */
    private static class PrePopulateCallback extends RoomDatabase.Callback {

        private final Context mContext;

        PrePopulateCallback(@NonNull Context context) {
            super();
            mContext = context;
        }

        @Override
        public void onCreate(@NonNull SupportSQLiteDatabase db) {
            super.onCreate(db);
            Log.i(TAG, "Database created — scheduling pre-population from JSON");
            DATABASE_EXECUTOR.execute(() -> populateFromJson(mContext));
        }

        @Override
        public void onOpen(@NonNull SupportSQLiteDatabase db) {
            super.onOpen(db);
            // Check if database is empty and needs population
            // This handles cases where onCreate was already called but
            // the population failed or the JSON file was added later
            DATABASE_EXECUTOR.execute(() -> {
                try {
                    ObjectDao dao = sInstance.objectDao();
                    int count = dao.getCount();
                    if (count == 0) {
                        Log.i(TAG, "Database is empty on open — populating from JSON");
                        populateFromJson(mContext);
                    } else {
                        Log.d(TAG, "Database has " + count + " entries — skipping population");
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Error checking database state on open", e);
                }
            });
        }
    }

    // ================================================================
    // JSON Parsing and Population
    // ================================================================

    /**
     * Reads the JSON file from assets, parses it, and inserts all
     * ObjectDescription entries into the database. This method is
     * called on DATABASE_EXECUTOR (background thread).
     *
     * @param context Application context for asset access.
     */
    private static void populateFromJson(@NonNull Context context) {
        try {
            String json = readAssetFile(context, ASSET_PATH);
            if (json == null || json.isEmpty()) {
                Log.w(TAG, "JSON asset is empty or missing: " + ASSET_PATH);
                return;
            }

            List<ObjectDescription> descriptions = parseJson(json);
            if (descriptions.isEmpty()) {
                Log.w(TAG, "No descriptions parsed from JSON");
                return;
            }

            // Insert all entries in a single transaction
            ObjectDao dao = sInstance.objectDao();
            dao.insertAll(descriptions);
            Log.i(TAG, "Pre-populated database with " + descriptions.size() + " object descriptions");

        } catch (JsonSyntaxException e) {
            Log.e(TAG, "JSON syntax error in " + ASSET_PATH, e);
        } catch (IllegalStateException e) {
            Log.e(TAG, "Database not available for population", e);
        } catch (Exception e) {
            Log.e(TAG, "Error populating database from JSON", e);
        }
    }

    /**
     * Reads a text file from the assets directory.
     *
     * @param context  Application context.
     * @param filePath Path relative to the assets folder.
     * @return File contents as a String, or null on error.
     */
    private static String readAssetFile(@NonNull Context context, @NonNull String filePath) {
        StringBuilder sb = new StringBuilder();
        try (InputStream is = context.getAssets().open(filePath);
             BufferedReader reader = new BufferedReader(
                     new InputStreamReader(is, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
            return sb.toString();
        } catch (IOException e) {
            Log.w(TAG, "Could not read asset file: " + filePath, e);
            return null;
        }
    }

    /**
     * Parses the JSON string into a list of ObjectDescription objects.
     * Uses Gson for robust JSON parsing with error handling per entry.
     *
     * @param json JSON string containing the object descriptions array.
     * @return List of parsed ObjectDescription entries (may be empty).
     */
    @NonNull
    private static List<ObjectDescription> parseJson(@NonNull String json) {
        List<ObjectDescription> descriptions = new ArrayList<>();

        try {
            JsonArray array = JsonParser.parseString(json).getAsJsonArray();
            Gson gson = new Gson();

            for (int i = 0; i < array.size(); i++) {
                try {
                    JsonObject obj = array.get(i).getAsJsonObject();

                    String className = obj.has("className")
                            ? obj.get("className").getAsString() : "";
                    String displayName = obj.has("displayName")
                            ? obj.get("displayName").getAsString() : className.toUpperCase();
                    String shortDescription = obj.has("shortDescription")
                            ? obj.get("shortDescription").getAsString() : "";
                    String commonUses = obj.has("commonUses")
                            ? obj.get("commonUses").getAsString() : "";
                    int hazardLevel = obj.has("hazardLevel")
                            ? obj.get("hazardLevel").getAsInt() : 0;
                    String category = obj.has("category")
                            ? obj.get("category").getAsString() : "UNKNOWN";

                    // Skip entries with empty className (invalid)
                    if (className.isEmpty()) {
                        Log.w(TAG, "Skipping entry at index " + i + ": empty className");
                        continue;
                    }

                    // Clamp hazard level to valid range
                    hazardLevel = Math.max(0, Math.min(3, hazardLevel));

                    ObjectDescription desc = new ObjectDescription(
                            className,
                            displayName,
                            shortDescription,
                            commonUses,
                            hazardLevel,
                            category
                    );
                    descriptions.add(desc);

                } catch (Exception e) {
                    Log.w(TAG, "Error parsing entry at index " + i + ", skipping", e);
                }
            }
        } catch (IllegalStateException e) {
            Log.e(TAG, "JSON root is not an array", e);
        }

        return descriptions;
    }

    // ================================================================
    // Lifecycle
    // ================================================================

    /**
     * Shuts down the database executor service. Should be called
     * when the application is terminating to release resources.
     */
    public static void shutdownExecutor() {
        if (!DATABASE_EXECUTOR.isShutdown()) {
            DATABASE_EXECUTOR.shutdownNow();
            Log.i(TAG, "Database executor shut down");
        }
    }
}
