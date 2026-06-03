/**
 * ModelManager.java
 *
 * Singleton that manages TensorFlow Lite model loading, caching, and
 * interpreter lifecycle for the Sync Vision app. Supports GPU delegate
 * (with automatic CPU fallback) and NNAPI delegate for hardware
 * acceleration on compatible devices.
 *
 * Models are loaded from the assets/models/ directory and optionally
 * extracted to internal storage for memory-mapped access.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.ml
 * Target SDK: 29+
 */

package com.syncvision.app.ml;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Singleton manager for TFLite model interpreters.
 * <p>
 * Responsibilities:
 * <ul>
 *   <li>Load model files from assets/models/ and cache them</li>
 *   <li>Create Interpreter instances with optional GPU/NNAPI delegates</li>
 *   <li>Cache interpreters by model name for reuse across frames</li>
 *   <li>Release interpreters when no longer needed</li>
 * </ul>
 * <p>
 * Usage:
 * <pre>
 *   ModelManager.getInstance().initialize(context);
 *   ModelManager.getInstance().setUseGpu(true);
 *   Interpreter interp = ModelManager.getInstance().getInterpreter("deeplab_v3.tflite", 4);
 *   // ... use interpreter for inference ...
 *   ModelManager.getInstance().releaseInterpreter("deeplab_v3.tflite");
 * </pre>
 */
public class ModelManager {

    private static final String TAG = "SV-ModelManager";

    /** Subdirectory in assets where models are stored. */
    private static final String MODELS_DIR = "models";

    // ================================================================
    // Singleton
    // ================================================================

    private static volatile ModelManager instance;

    /**
     * Returns the singleton ModelManager instance.
     *
     * @return The ModelManager instance.
     */
    @NonNull
    public static ModelManager getInstance() {
        if (instance == null) {
            synchronized (ModelManager.class) {
                if (instance == null) {
                    instance = new ModelManager();
                }
            }
        }
        return instance;
    }

    /** Private constructor for singleton. */
    private ModelManager() {}

    // ================================================================
    // State
    // ================================================================

    /** Application context for asset and file access. */
    @Nullable
    private Context appContext;

    /** Whether the manager has been initialized. */
    private boolean initialized = false;

    /** Whether to attempt GPU delegate for new interpreters. */
    private volatile boolean useGpu = false;

    /** Whether to attempt NNAPI delegate for new interpreters. */
    private volatile boolean useNnapi = false;

    /** Cache of active interpreters keyed by model name. */
    private final ConcurrentHashMap<String, Interpreter> interpreterCache = new ConcurrentHashMap<>();

    /** Cache of GPU delegates keyed by model name (for cleanup). */
    private final ConcurrentHashMap<String, GpuDelegate> gpuDelegateCache = new ConcurrentHashMap<>();

    /** Cache of NNAPI delegates keyed by model name (for cleanup). */
    private final ConcurrentHashMap<String, NnApiDelegate> nnapiDelegateCache = new ConcurrentHashMap<>();

    /** Cache of extracted model file paths. */
    private final Map<String, String> modelFilePathCache = new HashMap<>();

    // ================================================================
    // Initialization
    // ================================================================

    /**
     * Initializes the ModelManager with the application context.
     * Must be called once before any other methods.
     *
     * @param context Application context.
     */
    public void initialize(@NonNull Context context) {
        if (initialized) {
            Log.w(TAG, "ModelManager already initialized");
            return;
        }
        this.appContext = context.getApplicationContext();
        this.initialized = true;
        Log.i(TAG, "ModelManager initialized");
    }

    /**
     * Checks whether the manager has been initialized.
     *
     * @return true if initialized.
     */
    public boolean isInitialized() {
        return initialized;
    }

    // ================================================================
    // GPU / NNAPI Configuration
    // ================================================================

    /**
     * Sets whether to use the GPU delegate for new interpreters.
     * This only affects interpreters created after this call.
     * If GPU initialization fails, it will silently fall back to CPU.
     *
     * @param useGpu true to enable GPU delegate.
     */
    public void setUseGpu(boolean useGpu) {
        this.useGpu = useGpu;
        Log.i(TAG, "GPU delegate " + (useGpu ? "enabled" : "disabled"));
    }

    /**
     * Sets whether to use the NNAPI delegate for new interpreters.
     * This only affects interpreters created after this call.
     * If NNAPI is not available, it will silently fall back to CPU.
     *
     * @param useNnapi true to enable NNAPI delegate.
     */
    public void setUseNnapi(boolean useNnapi) {
        this.useNnapi = useNnapi;
        Log.i(TAG, "NNAPI delegate " + (useNnapi ? "enabled" : "disabled"));
    }

    /**
     * Returns whether GPU delegate is currently enabled.
     */
    public boolean isUseGpu() {
        return useGpu;
    }

    /**
     * Returns whether NNAPI delegate is currently enabled.
     */
    public boolean isUseNnapi() {
        return useNnapi;
    }

    // ================================================================
    // Interpreter Management
    // ================================================================

    /**
     * Gets or creates a TFLite Interpreter for the specified model.
     * If an interpreter for this model already exists in the cache,
     * it is returned directly. Otherwise, a new one is created.
     *
     * @param modelName  The model filename (e.g., "deeplab_v3.tflite").
     * @param numThreads Number of threads for inference.
     * @return A TFLite Interpreter instance.
     * @throws IllegalStateException if the manager has not been initialized.
     * @throws IOException           if the model file cannot be loaded.
     */
    @NonNull
    public Interpreter getInterpreter(@NonNull String modelName, int numThreads)
            throws IOException {
        if (!initialized || appContext == null) {
            throw new IllegalStateException("ModelManager not initialized. Call initialize() first.");
        }

        // Return cached interpreter if available
        Interpreter cached = interpreterCache.get(modelName);
        if (cached != null) {
            return cached;
        }

        // Load model and create interpreter
        synchronized (this) {
            // Double-check after acquiring lock
            cached = interpreterCache.get(modelName);
            if (cached != null) {
                return cached;
            }

            Log.i(TAG, "Loading model: " + modelName);

            // Extract model from assets to internal storage
            String modelPath = extractModel(modelName);
            MappedByteBuffer modelBuffer = loadModelFile(modelPath);

            // Build interpreter options
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(numThreads);

            // Try GPU delegate
            if (useGpu) {
                try {
                    GpuDelegate gpuDelegate = new GpuDelegate();
                    options.addDelegate(gpuDelegate);
                    gpuDelegateCache.put(modelName, gpuDelegate);
                    Log.i(TAG, "GPU delegate added for " + modelName);
                } catch (Exception e) {
                    Log.w(TAG, "GPU delegate not available for " + modelName
                            + ", falling back to CPU", e);
                }
            }

            // Try NNAPI delegate (can coexist with GPU in some configs,
            // but typically only one hardware accelerator is used)
            if (useNnapi && !useGpu) {
                try {
                    NnApiDelegate nnapiDelegate = new NnApiDelegate();
                    options.addDelegate(nnapiDelegate);
                    nnapiDelegateCache.put(modelName, nnapiDelegate);
                    Log.i(TAG, "NNAPI delegate added for " + modelName);
                } catch (Exception e) {
                    Log.w(TAG, "NNAPI not available for " + modelName
                            + ", falling back to CPU", e);
                }
            }

            // Create the interpreter
            Interpreter interpreter = new Interpreter(modelBuffer, options);
            interpreterCache.put(modelName, interpreter);

            Log.i(TAG, "Model loaded successfully: " + modelName);
            return interpreter;
        }
    }

    /**
     * Releases the interpreter for the specified model.
     * Also cleans up any associated delegates.
     *
     * @param modelName The model filename to release.
     */
    public void releaseInterpreter(@NonNull String modelName) {
        synchronized (this) {
            // Close interpreter
            Interpreter interpreter = interpreterCache.remove(modelName);
            if (interpreter != null) {
                interpreter.close();
                Log.i(TAG, "Released interpreter for: " + modelName);
            }

            // Close GPU delegate
            GpuDelegate gpuDelegate = gpuDelegateCache.remove(modelName);
            if (gpuDelegate != null) {
                gpuDelegate.close();
                Log.i(TAG, "Released GPU delegate for: " + modelName);
            }

            // Close NNAPI delegate
            NnApiDelegate nnapiDelegate = nnapiDelegateCache.remove(modelName);
            if (nnapiDelegate != null) {
                nnapiDelegate.close();
                Log.i(TAG, "Released NNAPI delegate for: " + modelName);
            }
        }
    }

    /**
     * Checks if an interpreter for the given model is currently loaded.
     *
     * @param modelName The model filename.
     * @return true if the interpreter is loaded and cached.
     */
    public boolean isModelLoaded(@NonNull String modelName) {
        return interpreterCache.containsKey(modelName);
    }

    /**
     * Releases all loaded interpreters and delegates.
     * Call this when the app is going to the background or being destroyed.
     */
    public void releaseAll() {
        synchronized (this) {
            Log.i(TAG, "Releasing all interpreters (" + interpreterCache.size() + ")");

            // Close all interpreters
            for (Map.Entry<String, Interpreter> entry : interpreterCache.entrySet()) {
                try {
                    entry.getValue().close();
                } catch (Exception e) {
                    Log.w(TAG, "Error closing interpreter for " + entry.getKey(), e);
                }
            }
            interpreterCache.clear();

            // Close all GPU delegates
            for (Map.Entry<String, GpuDelegate> entry : gpuDelegateCache.entrySet()) {
                try {
                    entry.getValue().close();
                } catch (Exception e) {
                    Log.w(TAG, "Error closing GPU delegate for " + entry.getKey(), e);
                }
            }
            gpuDelegateCache.clear();

            // Close all NNAPI delegates
            for (Map.Entry<String, NnApiDelegate> entry : nnapiDelegateCache.entrySet()) {
                try {
                    entry.getValue().close();
                } catch (Exception e) {
                    Log.w(TAG, "Error closing NNAPI delegate for " + entry.getKey(), e);
                }
            }
            nnapiDelegateCache.clear();

            Log.i(TAG, "All interpreters released");
        }
    }

    // ================================================================
    // Model File Management
    // ================================================================

    /**
     * Extracts a model file from assets to internal storage.
     * Caches the extracted file path for subsequent access.
     *
     * @param modelName The model filename in assets/models/.
     * @return The absolute path to the extracted model file.
     * @throws IOException if extraction fails.
     */
    @NonNull
    private String extractModel(@NonNull String modelName) throws IOException {
        // Check cache first
        String cachedPath = modelFilePathCache.get(modelName);
        if (cachedPath != null && new File(cachedPath).exists()) {
            return cachedPath;
        }

        File outputFile = new File(appContext.getFilesDir(), modelName);

        // Check if file already exists and matches asset size
        String assetPath = MODELS_DIR + "/" + modelName;
        InputStream is = null;
        FileOutputStream fos = null;

        try {
            AssetManager assets = appContext.getAssets();
            is = assets.open(assetPath);

            fos = new FileOutputStream(outputFile);

            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                fos.write(buffer, 0, bytesRead);
            }

            fos.flush();
            String path = outputFile.getAbsolutePath();
            modelFilePathCache.put(modelName, path);

            Log.i(TAG, "Extracted model: " + modelName + " → " + path);
            return path;

        } finally {
            if (is != null) {
                try { is.close(); } catch (IOException ignored) {}
            }
            if (fos != null) {
                try { fos.close(); } catch (IOException ignored) {}
            }
        }
    }

    /**
     * Loads a model file into a MappedByteBuffer for TFLite.
     * Memory-mapped files allow the OS to page in the model
     * on demand, reducing peak memory usage.
     *
     * @param path Absolute path to the model file.
     * @return A MappedByteBuffer containing the model data.
     * @throws IOException if the file cannot be read.
     */
    @NonNull
    private MappedByteBuffer loadModelFile(@NonNull String path) throws IOException {
        java.io.RandomAccessFile file = new java.io.RandomAccessFile(path, "r");
        try {
            FileChannel channel = file.getChannel();
            long fileSize = channel.size();
            return channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
        } finally {
            // The file can be closed after mapping; the MappedByteBuffer
            // retains its mapping independently.
            file.close();
        }
    }

    // ================================================================
    // Model Availability
    // ================================================================

    /**
     * Checks if a model file exists in the assets directory.
     *
     * @param modelName The model filename.
     * @return true if the model exists in assets/models/.
     */
    public boolean isModelAvailable(@NonNull String modelName) {
        if (appContext == null) {
            return false;
        }
        try {
            String[] files = appContext.getAssets().list(MODELS_DIR);
            if (files != null) {
                for (String file : files) {
                    if (modelName.equals(file)) {
                        return true;
                    }
                }
            }
        } catch (IOException e) {
            Log.w(TAG, "Error listing model assets", e);
        }
        return false;
    }

    /**
     * Returns the number of currently loaded interpreters.
     */
    public int getLoadedModelCount() {
        return interpreterCache.size();
    }
}
