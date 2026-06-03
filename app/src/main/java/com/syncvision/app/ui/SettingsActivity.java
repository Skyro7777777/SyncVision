/**
 * SettingsActivity.java
 *
 * Settings screen for the Sync Vision camera app. Provides user-configurable
 * options for GPU acceleration, NNAPI, detection thresholds, visual effects,
 * camera resolution, and other preferences.
 *
 * Uses AndroidX PreferenceFragmentCompat for a standard settings UI
 * with SwitchPreference, SeekBarPreference, ListPreference, and
 * Preference categories.
 *
 * Settings categories:
 *   - Performance: GPU, NNAPI, confidence threshold
 *   - Display: scanline intensity, night mode, sync diagram, depth labels
 *   - Alerts: audio, vibration
 *   - Camera: resolution
 *   - Language: translation language
 *   - About: app info, privacy statement
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.ui
 * Target SDK: 29+
 */

package com.syncvision.app.ui;

import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatDelegate;
import androidx.preference.DropDownPreference;
import androidx.preference.Preference;
import androidx.preference.PreferenceCategory;
import androidx.preference.PreferenceFragmentCompat;
import androidx.preference.PreferenceManager;
import androidx.preference.SeekBarPreference;
import androidx.preference.SwitchPreferenceCompat;

import com.syncvision.app.R;

/**
 * Activity hosting the settings fragment for the Sync Vision app.
 * Provides a standard Android settings experience with back navigation.
 */
public class SettingsActivity extends AppCompatActivity {

    private static final String TAG = "SV-SettingsActivity";

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Enable back button in action bar
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle("SETTINGS");
        }

        // Load the settings fragment
        getSupportFragmentManager()
                .beginTransaction()
                .replace(android.R.id.content, new SettingsFragment())
                .commit();

        Log.d(TAG, "SettingsActivity created");
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            finish();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    /**
     * Preference fragment that creates and manages all settings UI elements.
     * Settings are organized into logical categories and persisted via
     * SharedPreferences.
     */
    public static class SettingsFragment extends PreferenceFragmentCompat {

        // ================================================================
        // Preference Keys
        // ================================================================

        /** GPU acceleration toggle key. */
        public static final String KEY_GPU_ACCELERATION = "gpu_acceleration";

        /** NNAPI toggle key. */
        public static final String KEY_NNAPI = "nnapi";

        /** Detection confidence threshold key. */
        public static final String KEY_CONFIDENCE_THRESHOLD = "confidence_threshold";

        /** Scanline intensity key. */
        public static final String KEY_SCANLINE_INTENSITY = "scanline_intensity";

        /** Night mode key. */
        public static final String KEY_NIGHT_MODE = "night_mode";

        /** Audio alerts toggle key. */
        public static final String KEY_AUDIO_ALERTS = "audio_alerts";

        /** Vibration alerts toggle key. */
        public static final String KEY_VIBRATION_ALERTS = "vibration_alerts";

        /** Show sync diagram toggle key. */
        public static final String KEY_SHOW_SYNC_DIAGRAM = "show_sync_diagram";

        /** Show depth labels toggle key. */
        public static final String KEY_SHOW_DEPTH_LABELS = "show_depth_labels";

        /** Camera resolution key. */
        public static final String KEY_CAMERA_RESOLUTION = "camera_resolution";

        /** Language key (for translation). */
        public static final String KEY_LANGUAGE = "language";

        /** About / app info key. */
        public static final String KEY_ABOUT = "about";

        /** Privacy statement key. */
        public static final String KEY_PRIVACY = "privacy";

        // ================================================================
        // Default Values
        // ================================================================

        /** Default GPU acceleration state. */
        public static final boolean DEFAULT_GPU_ACCELERATION = true;

        /** Default NNAPI state. */
        public static final boolean DEFAULT_NNAPI = false;

        /** Default detection confidence threshold. */
        public static final int DEFAULT_CONFIDENCE_THRESHOLD = 50; // maps to 0.5

        /** Default scanline intensity. */
        public static final int DEFAULT_SCANLINE_INTENSITY = 30; // maps to 0.3

        /** Default night mode (auto). */
        public static final String DEFAULT_NIGHT_MODE = "auto";

        /** Default audio alerts state. */
        public static final boolean DEFAULT_AUDIO_ALERTS = true;

        /** Default vibration alerts state. */
        public static final boolean DEFAULT_VIBRATION_ALERTS = true;

        /** Default show sync diagram state. */
        public static final boolean DEFAULT_SHOW_SYNC_DIAGRAM = true;

        /** Default show depth labels state. */
        public static final boolean DEFAULT_SHOW_DEPTH_LABELS = false;

        /** Default camera resolution. */
        public static final String DEFAULT_CAMERA_RESOLUTION = "1280x720";

        /** Default language. */
        public static final String DEFAULT_LANGUAGE = "en";

        @Override
        public void onCreatePreferences(@Nullable Bundle savedInstanceState,
                                        @Nullable String rootKey) {
            // Get the preference manager
            PreferenceManager pm = getPreferenceManager();
            pm.setSharedPreferencesName("syncvision_settings");

            setPreferencesFromResource(R.xml.settings, rootKey);

            // If the XML doesn't exist, build programmatically
            try {
                if (getPreferenceScreen().getPreferenceCount() == 0) {
                    buildPreferencesProgrammatically();
                }
            } catch (Exception e) {
                Log.w(TAG, "XML preferences not found, building programmatically", e);
                buildPreferencesProgrammatically();
            }

            Log.d(TAG, "Settings preferences created");
        }

        /**
         * Builds all preferences programmatically as a fallback
         * when the XML resource is not available.
         */
        private void buildPreferencesProgrammatically() {
            PreferenceScreen screen = getPreferenceManager().createPreferenceScreen(requireContext());
            setPreferenceScreen(screen);

            // ---- Performance Category ----
            PreferenceCategory perfCategory = new PreferenceCategory(requireContext());
            perfCategory.setTitle("PERFORMANCE");
            screen.addPreference(perfCategory);

            // GPU acceleration toggle
            SwitchPreferenceCompat gpuPref = new SwitchPreferenceCompat(requireContext());
            gpuPref.setKey(KEY_GPU_ACCELERATION);
            gpuPref.setTitle("GPU ACCELERATION");
            gpuPref.setSummary("USE GPU DELEGATE FOR TFLITE INFERENCE");
            gpuPref.setDefaultValue(DEFAULT_GPU_ACCELERATION);
            perfCategory.addPreference(gpuPref);

            // NNAPI toggle
            SwitchPreferenceCompat nnapiPref = new SwitchPreferenceCompat(requireContext());
            nnapiPref.setKey(KEY_NNAPI);
            nnapiPref.setTitle("NNAPI");
            nnapiPref.setSummary("USE ANDROID NNAPI DELEGATE (MAY IMPROVE PERFORMANCE ON SOME DEVICES)");
            nnapiPref.setDefaultValue(DEFAULT_NNAPI);
            perfCategory.addPreference(nnapiPref);

            // Detection confidence threshold
            SeekBarPreference confPref = new SeekBarPreference(requireContext());
            confPref.setKey(KEY_CONFIDENCE_THRESHOLD);
            confPref.setTitle("DETECTION CONFIDENCE");
            confPref.setSummary("MINIMUM CONFIDENCE FOR OBJECT DETECTION");
            confPref.setMin(30);
            confPref.setMax(90);
            confPref.setDefaultValue(DEFAULT_CONFIDENCE_THRESHOLD);
            confPref.setShowSeekBarValue(true);
            perfCategory.addPreference(confPref);

            // ---- Display Category ----
            PreferenceCategory displayCategory = new PreferenceCategory(requireContext());
            displayCategory.setTitle("DISPLAY");
            screen.addPreference(displayCategory);

            // Scanline intensity
            SeekBarPreference scanlinePref = new SeekBarPreference(requireContext());
            scanlinePref.setKey(KEY_SCANLINE_INTENSITY);
            scanlinePref.setTitle("SCANLINE INTENSITY");
            scanlinePref.setSummary("CRT SCANLINE EFFECT INTENSITY (0 = OFF)");
            scanlinePref.setMin(0);
            scanlinePref.setMax(100);
            scanlinePref.setDefaultValue(DEFAULT_SCANLINE_INTENSITY);
            scanlinePref.setShowSeekBarValue(true);
            displayCategory.addPreference(scanlinePref);

            // Night mode
            DropDownPreference nightModePref = new DropDownPreference(requireContext());
            nightModePref.setKey(KEY_NIGHT_MODE);
            nightModePref.setTitle("NIGHT MODE");
            nightModePref.setSummary("AUTOMATIC OR MANUAL NIGHT MODE");
            nightModePref.setEntries(new CharSequence[]{"AUTO", "ON", "OFF"});
            nightModePref.setEntryValues(new CharSequence[]{"auto", "on", "off"});
            nightModePref.setDefaultValue(DEFAULT_NIGHT_MODE);
            displayCategory.addPreference(nightModePref);

            // Show sync diagram
            SwitchPreferenceCompat syncPref = new SwitchPreferenceCompat(requireContext());
            syncPref.setKey(KEY_SHOW_SYNC_DIAGRAM);
            syncPref.setTitle("SHOW SYNC DIAGRAM");
            syncPref.setSummary("DISPLAY THE RELATIONSHIP GRAPH OVERLAY");
            syncPref.setDefaultValue(DEFAULT_SHOW_SYNC_DIAGRAM);
            displayCategory.addPreference(syncPref);

            // Show depth labels
            SwitchPreferenceCompat depthPref = new SwitchPreferenceCompat(requireContext());
            depthPref.setKey(KEY_SHOW_DEPTH_LABELS);
            depthPref.setTitle("SHOW DEPTH LABELS");
            depthPref.setSummary("DISPLAY DISTANCE ESTIMATES ON DETECTED OBJECTS");
            depthPref.setDefaultValue(DEFAULT_SHOW_DEPTH_LABELS);
            displayCategory.addPreference(depthPref);

            // ---- Alerts Category ----
            PreferenceCategory alertsCategory = new PreferenceCategory(requireContext());
            alertsCategory.setTitle("ALERTS");
            screen.addPreference(alertsCategory);

            // Audio alerts
            SwitchPreferenceCompat audioPref = new SwitchPreferenceCompat(requireContext());
            audioPref.setKey(KEY_AUDIO_ALERTS);
            audioPref.setTitle("AUDIO ALERTS");
            audioPref.setSummary("PLAY SOUND FOR HAZARD WARNINGS");
            audioPref.setDefaultValue(DEFAULT_AUDIO_ALERTS);
            alertsCategory.addPreference(audioPref);

            // Vibration alerts
            SwitchPreferenceCompat vibPref = new SwitchPreferenceCompat(requireContext());
            vibPref.setKey(KEY_VIBRATION_ALERTS);
            vibPref.setTitle("VIBRATION ALERTS");
            vibPref.setSummary("VIBRATE DEVICE FOR HAZARD WARNINGS");
            vibPref.setDefaultValue(DEFAULT_VIBRATION_ALERTS);
            alertsCategory.addPreference(vibPref);

            // ---- Camera Category ----
            PreferenceCategory cameraCategory = new PreferenceCategory(requireContext());
            cameraCategory.setTitle("CAMERA");
            screen.addPreference(cameraCategory);

            // Camera resolution
            DropDownPreference resPref = new DropDownPreference(requireContext());
            resPref.setKey(KEY_CAMERA_RESOLUTION);
            resPref.setTitle("CAMERA RESOLUTION");
            resPref.setSummary("SELECT CAMERA PREVIEW RESOLUTION");
            resPref.setEntries(new CharSequence[]{"640 X 480 (VGA)", "1280 X 720 (HD)"});
            resPref.setEntryValues(new CharSequence[]{"640x480", "1280x720"});
            resPref.setDefaultValue(DEFAULT_CAMERA_RESOLUTION);
            cameraCategory.addPreference(resPref);

            // ---- Language Category ----
            PreferenceCategory langCategory = new PreferenceCategory(requireContext());
            langCategory.setTitle("LANGUAGE");
            screen.addPreference(langCategory);

            // Language selection
            DropDownPreference langPref = new DropDownPreference(requireContext());
            langPref.setKey(KEY_LANGUAGE);
            langPref.setTitle("TRANSLATION LANGUAGE");
            langPref.setSummary("LANGUAGE FOR OCR TRANSLATION OUTPUT");
            langPref.setEntries(new CharSequence[]{
                    "ENGLISH", "SPANISH", "FRENCH", "GERMAN",
                    "CHINESE", "JAPANESE", "KOREAN", "ARABIC"});
            langPref.setEntryValues(new CharSequence[]{
                    "en", "es", "fr", "de", "zh", "ja", "ko", "ar"});
            langPref.setDefaultValue(DEFAULT_LANGUAGE);
            langCategory.addPreference(langPref);

            // ---- About Category ----
            PreferenceCategory aboutCategory = new PreferenceCategory(requireContext());
            aboutCategory.setTitle("ABOUT");
            screen.addPreference(aboutCategory);

            // App info
            Preference aboutPref = new Preference(requireContext());
            aboutPref.setKey(KEY_ABOUT);
            aboutPref.setTitle("SYNC VISION");
            aboutPref.setSummary("VERSION 1.0.0 — ON-DEVICE ML CAMERA APP\nNO INTERNET PERMISSION. ALL DATA STAYS ON YOUR DEVICE.");
            aboutPref.setSelectable(false);
            aboutCategory.addPreference(aboutPref);

            // Privacy statement
            Preference privacyPref = new Preference(requireContext());
            privacyPref.setKey(KEY_PRIVACY);
            privacyPref.setTitle("PRIVACY STATEMENT");
            privacyPref.setSummary("SYNC VISION COLLECTS NO DATA. NO INTERNET PERMISSION. "
                    + "ALL ML PROCESSING HAPPENS ON-DEVICE. NO FACIAL RECOGNITION. "
                    + "NO CLOUD SERVICES. YOUR PRIVACY IS BY DESIGN.");
            privacyPref.setSelectable(false);
            aboutCategory.addPreference(privacyPref);

            Log.d(TAG, "Preferences built programmatically");
        }
    }
}
