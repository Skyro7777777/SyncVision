/**
 * DescriptionLoader.java
 *
 * High-level loader for object descriptions that bridges the JSON asset
 * file and the Room database. Provides a simple singleton API for the
 * rest of the app to query object descriptions by class name without
 * worrying about database initialization or thread management.
 *
 * Initialization flow:
 *   1. init(context) triggers asynchronous database creation + JSON pre-population
 *   2. getDescription(className) blocks briefly on first call if still loading
 *   3. Subsequent calls use the in-memory cache for zero-latency lookups
 *
 * Thread safety: all public methods are thread-safe. The singleton uses
 * double-checked locking, and the internal cache uses volatile + synchronized.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.data
 * Target SDK: 29+
 */

package com.syncvision.app.data;

import android.content.Context;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

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
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Singleton loader for object descriptions. Reads from the Room database
 * (which is pre-populated from JSON on first run) and maintains an
 * in-memory cache for fast lookups during real-time ML inference.
 * <p>
 * Usage:
 * <pre>
 *   // In Application.onCreate():
 *   DescriptionLoader.getInstance().init(getApplicationContext());
 *
 *   // In ML pipeline:
 *   ObjectDescription desc = DescriptionLoader.getInstance().getDescription("car");
 *   if (desc != null) {
 *       String info = desc.getShortDescription();
 *   }
 * </pre>
 */
public class DescriptionLoader {

    private static final String TAG = "SV-DescriptionLoader";

    /** Path to the JSON asset file for fallback loading. */
    private static final String ASSET_PATH = "data/object_descriptions.json";

    /** Timeout for waiting on database initialization (seconds). */
    private static final long INIT_TIMEOUT_SECONDS = 10L;

    // ================================================================
    // Singleton
    // ================================================================

    /** Singleton instance — volatile for double-checked locking. */
    private static volatile DescriptionLoader sInstance;

    /**
     * Returns the singleton DescriptionLoader instance.
     *
     * @return The DescriptionLoader singleton.
     */
    @NonNull
    public static DescriptionLoader getInstance() {
        if (sInstance == null) {
            synchronized (DescriptionLoader.class) {
                if (sInstance == null) {
                    sInstance = new DescriptionLoader();
                }
            }
        }
        return sInstance;
    }

    /** Private constructor for singleton pattern. */
    private DescriptionLoader() {
        // Singleton — use getInstance()
    }

    // ================================================================
    // State
    // ================================================================

    /** Application context for asset and database access. */
    private volatile Context mAppContext;

    /** Whether initialization has been triggered. */
    private final AtomicBoolean mInitStarted = new AtomicBoolean(false);

    /** Whether loading is complete (database ready or fallback loaded). */
    private final AtomicBoolean mLoaded = new AtomicBoolean(false);

    /** Latch that signals when initialization is complete. */
    private volatile CountDownLatch mInitLatch;

    /** In-memory cache: className → ObjectDescription for fast lookups. */
    private volatile Map<String, ObjectDescription> mCache = Collections.emptyMap();

    /** Executor for background loading operations. */
    private final ExecutorService mLoaderExecutor =
            Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "SV-DescLoader");
                t.setPriority(Thread.NORM_PRIORITY - 1);
                t.setDaemon(true);
                return t;
            });

    /** Room database reference (may be null before init). */
    private volatile ObjectDatabase mDatabase;

    // ================================================================
    // Initialization
    // ================================================================

    /**
     * Initializes the DescriptionLoader. Must be called once, ideally
     * in Application.onCreate(). Triggers asynchronous loading from the
     * Room database (which auto-populates from JSON on first run).
     *
     * @param context Application context (will use getApplicationContext()).
     */
    public void init(@NonNull Context context) {
        if (mInitStarted.getAndSet(true)) {
            Log.d(TAG, "Already initialized — skipping");
            return;
        }

        mAppContext = context.getApplicationContext();
        mInitLatch = new CountDownLatch(1);

        Log.i(TAG, "Starting description loader initialization");

        mLoaderExecutor.execute(() -> {
            try {
                loadFromDatabase();
            } catch (Exception e) {
                Log.e(TAG, "Database loading failed, trying JSON fallback", e);
                try {
                    loadFromAssets();
                } catch (Exception e2) {
                    Log.e(TAG, "JSON fallback also failed", e2);
                }
            } finally {
                mLoaded.set(true);
                if (mInitLatch != null) {
                    mInitLatch.countDown();
                }
                Log.i(TAG, "Description loader initialized — "
                        + mCache.size() + " descriptions cached");
            }
        });
    }

    /**
     * Loads descriptions from the Room database. The database's
     * PrePopulateCallback handles JSON→Room on first creation.
     */
    private void loadFromDatabase() {
        mDatabase = ObjectDatabase.getInstance(mAppContext);

        // Wait briefly for the database callback to complete population.
        // The callback runs on a separate executor, so we retry with
        // a small sleep loop.
        ObjectDao dao = mDatabase.objectDao();
        int attempts = 0;
        final int MAX_ATTEMPTS = 20;
        final int SLEEP_MS = 100;

        while (attempts < MAX_ATTEMPTS) {
            try {
                List<ObjectDescription> descriptions = dao.getAllDescriptions();
                if (!descriptions.isEmpty()) {
                    buildCache(descriptions);
                    return;
                }
            } catch (Exception e) {
                Log.w(TAG, "Database query attempt " + attempts + " failed", e);
            }

            attempts++;
            try {
                Thread.sleep(SLEEP_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        Log.w(TAG, "Database still empty after " + MAX_ATTEMPTS + " attempts");
    }

    /**
     * Fallback: loads descriptions directly from the JSON asset file.
     * Used when the Room database is unavailable or empty after
     * repeated attempts.
     */
    private void loadFromAssets() {
        try {
            String json = readAssetFile(ASSET_PATH);
            if (json == null || json.isEmpty()) {
                Log.w(TAG, "JSON asset not available for fallback loading");
                return;
            }

            List<ObjectDescription> descriptions = parseJson(json);
            if (!descriptions.isEmpty()) {
                buildCache(descriptions);
                Log.i(TAG, "Fallback: loaded " + descriptions.size()
                        + " descriptions from JSON assets");
            }
        } catch (Exception e) {
            Log.e(TAG, "Error in JSON fallback loading", e);
        }
    }

    /**
     * Builds the in-memory cache from a list of descriptions.
     *
     * @param descriptions List of ObjectDescription entries.
     */
    private void buildCache(@NonNull List<ObjectDescription> descriptions) {
        Map<String, ObjectDescription> cache = new HashMap<>(descriptions.size() * 4 / 3);
        for (ObjectDescription desc : descriptions) {
            cache.put(desc.getClassName(), desc);
        }
        mCache = Collections.unmodifiableMap(cache);
    }

    // ================================================================
    // Public Query API
    // ================================================================

    /**
     * Returns the ObjectDescription for a given class name, or null
     * if not found. This is the primary lookup method used by the
     * ML pipeline after object detection.
     * <p>
     * If initialization is still in progress, this method will block
     * for up to INIT_TIMEOUT_SECONDS waiting for the cache to be
     * populated. After that, it returns null.
     *
     * @param className The machine-readable class name (e.g., "car").
     * @return ObjectDescription, or null if not found / not loaded.
     */
    @Nullable
    public ObjectDescription getDescription(@NonNull String className) {
        awaitInitialization();

        // Direct cache lookup — zero allocation
        ObjectDescription desc = mCache.get(className);
        if (desc != null) {
            return desc;
        }

        // Try normalized lookup (lowercase, spaces → underscores)
        String normalized = className.toLowerCase()
                .replace(' ', '_')
                .replace('-', '_');
        desc = mCache.get(normalized);
        if (desc != null) {
            return desc;
        }

        // Try database directly if cache miss
        if (mDatabase != null) {
            try {
                desc = mDatabase.objectDao().getByClassName(className);
                if (desc != null) {
                    // Add to cache for future lookups
                    synchronized (this) {
                        Map<String, ObjectDescription> newCache =
                                new HashMap<>(mCache);
                        newCache.put(className, desc);
                        mCache = Collections.unmodifiableMap(newCache);
                    }
                    return desc;
                }
            } catch (Exception e) {
                Log.w(TAG, "Database lookup failed for: " + className, e);
            }
        }

        return null;
    }

    /**
     * Returns all loaded ObjectDescription entries.
     * If initialization is still in progress, blocks briefly.
     *
     * @return Unmodifiable list of all ObjectDescription entries.
     */
    @NonNull
    public List<ObjectDescription> getAllDescriptions() {
        awaitInitialization();
        return new ArrayList<>(mCache.values());
    }

    /**
     * Returns true if the description data has been loaded and the
     * cache is ready for queries. Non-blocking check.
     *
     * @return True if descriptions are loaded.
     */
    public boolean isLoaded() {
        return mLoaded.get();
    }

    /**
     * Returns the number of descriptions currently cached.
     * Non-blocking.
     *
     * @return Cache size.
     */
    public int getCacheSize() {
        return mCache.size();
    }

    // ================================================================
    // Internal Helpers
    // ================================================================

    /**
     * Blocks the calling thread until initialization is complete
     * or the timeout expires. Safe to call from any thread.
     */
    private void awaitInitialization() {
        if (mLoaded.get()) {
            return;
        }

        if (mInitLatch != null) {
            try {
                if (!mInitLatch.await(INIT_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                    Log.w(TAG, "Initialization timeout — returning partial data");
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                Log.w(TAG, "Interrupted while waiting for initialization");
            }
        }
    }

    /**
     * Reads a text file from the assets directory.
     *
     * @param filePath Path relative to assets folder.
     * @return File contents, or null on error.
     */
    @Nullable
    private String readAssetFile(@NonNull String filePath) {
        if (mAppContext == null) {
            Log.w(TAG, "App context not available for asset reading");
            return null;
        }

        StringBuilder sb = new StringBuilder();
        try (InputStream is = mAppContext.getAssets().open(filePath);
             BufferedReader reader = new BufferedReader(
                     new InputStreamReader(is, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
            return sb.toString();
        } catch (IOException e) {
            Log.w(TAG, "Could not read asset: " + filePath, e);
            return null;
        }
    }

    /**
     * Parses the JSON string into a list of ObjectDescription entries.
     *
     * @param json JSON string (array of object description objects).
     * @return Parsed list (may be empty, never null).
     */
    @NonNull
    private List<ObjectDescription> parseJson(@NonNull String json) {
        List<ObjectDescription> descriptions = new ArrayList<>();

        try {
            JsonArray array = JsonParser.parseString(json).getAsJsonArray();

            for (int i = 0; i < array.size(); i++) {
                try {
                    JsonObject obj = array.get(i).getAsJsonObject();

                    String className = obj.has("className")
                            ? obj.get("className").getAsString() : "";
                    String displayName = obj.has("displayName")
                            ? obj.get("displayName").getAsString()
                            : className.toUpperCase();
                    String shortDescription = obj.has("shortDescription")
                            ? obj.get("shortDescription").getAsString() : "";
                    String commonUses = obj.has("commonUses")
                            ? obj.get("commonUses").getAsString() : "";
                    int hazardLevel = obj.has("hazardLevel")
                            ? obj.get("hazardLevel").getAsInt() : 0;
                    String category = obj.has("category")
                            ? obj.get("category").getAsString() : "UNKNOWN";

                    if (className.isEmpty()) {
                        continue;
                    }

                    hazardLevel = Math.max(0, Math.min(3, hazardLevel));

                    descriptions.add(new ObjectDescription(
                            className, displayName, shortDescription,
                            commonUses, hazardLevel, category));

                } catch (Exception e) {
                    Log.w(TAG, "Parse error at index " + i, e);
                }
            }
        } catch (JsonSyntaxException | IllegalStateException e) {
            Log.e(TAG, "Invalid JSON format", e);
        }

        return descriptions;
    }

    // ================================================================
    // Lifecycle
    // ================================================================

    /**
     * Releases resources. Should be called when the application is
     * terminating. After shutdown, the loader cannot be reused.
     */
    public void shutdown() {
        if (!mLoaderExecutor.isShutdown()) {
            mLoaderExecutor.shutdownNow();
        }
        ObjectDatabase.shutdownExecutor();
        Log.i(TAG, "DescriptionLoader shut down");
    }
}
