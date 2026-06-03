/**
 * SyncVisionApp.kt
 *
 * Application class for the Sync Vision camera app.
 * Initializes the ModelManager singleton, configures strict mode
 * in debug builds, and provides app-wide context access.
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app
 * Target SDK: 29+
 */

package com.syncvision.app

import android.app.Application
import android.os.Build
import android.os.StrictMode
import android.util.Log
import com.syncvision.app.ml.ModelManager

/**
 * Custom Application class for the Sync Vision app.
 *
 * Responsibilities:
 *   - Initialize the ModelManager singleton early in the app lifecycle
 *   - Enable StrictMode in debug builds to catch accidental I/O on the main thread
 *   - Provide a static app context accessor for components that lack context
 */
class SyncVisionApp : Application() {

    companion object {
        private const val TAG = "SV-SyncVisionApp"

        /**
         * Static reference to the application context.
         * Accessible from anywhere via SyncVisionApp.appContext.
         * This is safe because the Application instance is created
         * before any Activity and lives for the entire app lifecycle.
         */
        lateinit var appContext: Application
            private set
    }

    // ================================================================
    // Application Lifecycle
    // ================================================================

    override fun onCreate() {
        super.onCreate()

        // Store the app-wide context
        appContext = this

        // Enable StrictMode in debug builds only
        if (BuildConfig.DEBUG) {
            enableStrictMode()
        }

        // Initialize the ModelManager singleton
        try {
            ModelManager.getInstance().initialize(this)
            Log.i(TAG, "ModelManager initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ModelManager", e)
        }

        Log.i(TAG, "SyncVisionApp created — version: ${BuildConfig.VERSION_NAME}")
    }

    // ================================================================
    // Strict Mode Configuration
    // ================================================================

    /**
     * Enables StrictMode for debug builds to catch:
     *   - Accidental disk I/O on the main thread
     *   - Network operations on the main thread
     *   - Unbuffered output streams
     *   - Custom slow calls
     *
     * Only enabled in DEBUG builds — never in release.
     * Penalty: Log to crashlytics / logcat (no death penalty
     * to avoid crashing during development for minor violations).
     */
    private fun enableStrictMode() {
        // Thread policy: detect disk reads/writes, network ops, custom slow calls
        val threadPolicy = StrictMode.ThreadPolicy.Builder()
            .detectAll()
            .penaltyLog()    // Log violations to logcat
            .build()

        // VM policy: detect activity leaks, closable objects, SQL objects, etc.
        val vmPolicyBuilder = StrictMode.VmPolicy.Builder()
            .detectActivityLeaks()
            .detectLeakedClosableObjects()
            .detectLeakedSqlLiteObjects()

        // detectUntaggedSockets is available on API 26+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vmPolicyBuilder.detectUntaggedSockets()
        }

        // detectNonSdkApiUsage is available on API 28+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            vmPolicyBuilder.detectNonSdkApiUsage()
        }

        val vmPolicy = vmPolicyBuilder
            .penaltyLog()
            .build()

        StrictMode.setThreadPolicy(threadPolicy)
        StrictMode.setVmPolicy(vmPolicy)

        Log.d(TAG, "StrictMode enabled (debug build)")
    }

    // ================================================================
    // Termination
    // ================================================================

    override fun onTerminate() {
        super.onTerminate()
        try {
            ModelManager.getInstance().releaseAll()
            Log.i(TAG, "ModelManager released all interpreters")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing ModelManager", e)
        }
        Log.i(TAG, "SyncVisionApp terminated")
    }
}
