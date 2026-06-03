/**
 * PermissionHelper.java
 *
 * Helper class for managing Android runtime permissions required by
 * the Sync Vision app. Currently handles the CAMERA permission,
 * which is essential for the app's core functionality.
 *
 * Handles:
 *   - Checking if permissions are already granted
 *   - Requesting permissions with proper rationale
 *   - Processing permission results
 *   - Showing an educational rationale dialog before the system
 *     permission prompt (recommended by Android UX guidelines)
 *
 * Usage from an Activity:
 * <pre>
 *   if (!PermissionHelper.checkAndRequestPermissions(this)) {
 *       // Permission request launched — wait for callback
 *   } else {
 *       // Already have permission — start camera
 *   }
 *
 *   // In onRequestPermissionsResult:
 *   if (PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults)) {
 *       // Camera permission granted — start camera
 *   }
 * </pre>
 *
 * Sync Vision — Android Camera App with ML-powered Overlay
 * Package: com.syncvision.app.util
 * Target SDK: 29+
 */

package com.syncvision.app.util;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Context;
import android.content.pm.PackageManager;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

/**
 * Helper for requesting and checking Android runtime permissions.
 * Focuses on the CAMERA permission required for the app's core
 * camera feed functionality.
 * <p>
 * This class follows Android best practices:
 *   1. Check if permission is already granted
 *   2. Show rationale if user previously denied
 *   3. Request permission via ActivityCompat
 *   4. Process result in onRequestPermissionsResult
 */
public final class PermissionHelper {

    private static final String TAG = "SV-PermissionHelper";

    /** Request code for camera permission. */
    public static final int REQUEST_CODE_CAMERA = 1001;

    /** Request code for all permissions combined. */
    public static final int REQUEST_CODE_ALL = 1000;

    /** Camera permission string. */
    private static final String PERM_CAMERA = Manifest.permission.CAMERA;

    /**
     * All permissions required by the Sync Vision app.
     * Currently only CAMERA — no INTERNET, no storage, no contacts.
     * Privacy-first design: minimal permission footprint.
     */
    private static final String[] REQUIRED_PERMISSIONS = {
            PERM_CAMERA
    };

    // Private constructor — utility class
    private PermissionHelper() {
        throw new AssertionError("PermissionHelper is a utility class; do not instantiate.");
    }

    // ================================================================
    // Permission Checks
    // ================================================================

    /**
     * Checks whether the CAMERA permission has been granted.
     *
     * @param context Any context (application or activity).
     * @return True if CAMERA permission is granted.
     */
    public static boolean hasCameraPermission(@NonNull Context context) {
        return ContextCompat.checkSelfPermission(context, PERM_CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * Checks whether all required permissions have been granted.
     *
     * @param context Any context.
     * @return True if all required permissions are granted.
     */
    public static boolean hasAllPermissions(@NonNull Context context) {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(context, permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    // ================================================================
    // Permission Requests
    // ================================================================

    /**
     * Checks for required permissions and requests them if not yet
     * granted. If the user has previously denied a permission, shows
     * a rationale dialog explaining why the permission is needed
     * before re-requesting.
     * <p>
     * This method should be called from an Activity's onCreate() or
     * onStart() to ensure permissions are available before the camera
     * subsystem initializes.
     *
     * @param activity The requesting activity.
     * @return True if all permissions are already granted,
     *         False if a permission request was launched.
     */
    public static boolean checkAndRequestPermissions(@NonNull AppCompatActivity activity) {
        // Check if all permissions are already granted
        if (hasAllPermissions(activity)) {
            Log.d(TAG, "All permissions already granted");
            return true;
        }

        // Check if we should show rationale for CAMERA permission
        if (ActivityCompat.shouldShowRequestPermissionRationale(activity, PERM_CAMERA)) {
            Log.d(TAG, "Showing permission rationale for CAMERA");
            showCameraRationaleDialog(activity);
            return false;
        }

        // No rationale needed — request permissions directly
        Log.d(TAG, "Requesting permissions: CAMERA");
        ActivityCompat.requestPermissions(
                activity,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_ALL
        );
        return false;
    }

    // ================================================================
    // Result Handling
    // ================================================================

    /**
     * Processes the result of a permission request. Should be called
     * from the Activity's onRequestPermissionsResult() callback.
     *
     * @param requestCode  The request code passed to requestPermissions().
     * @param permissions  The requested permissions.
     * @param grantResults The grant results for the corresponding permissions.
     * @return True if all requested permissions were granted.
     */
    public static boolean onRequestPermissionsResult(int requestCode,
                                                     @NonNull String[] permissions,
                                                     @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CODE_ALL || requestCode == REQUEST_CODE_CAMERA) {
            // Check if all results are granted
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }

            if (allGranted) {
                Log.i(TAG, "All requested permissions granted");
            } else {
                Log.w(TAG, "Some permissions denied — camera features will be unavailable");
            }

            return allGranted;
        }

        // Not our request code
        return false;
    }

    // ================================================================
    // Rationale Dialog
    // ================================================================

    /**
     * Shows an educational rationale dialog explaining why the CAMERA
     * permission is required. After the user acknowledges, the
     * permission request is re-launched.
     * <p>
     * Android UX guidelines recommend showing a rationale when the
     * user has previously denied a permission, to explain the benefit
     * before asking again.
     *
     * @param activity The activity to show the dialog on.
     */
    private static void showCameraRationaleDialog(@NonNull AppCompatActivity activity) {
        new AlertDialog.Builder(activity)
                .setTitle("CAMERA ACCESS REQUIRED")
                .setMessage(
                        "SYNC VISION NEEDS CAMERA ACCESS TO PROVIDE\n"
                        + "REAL-TIME OBJECT DETECTION AND OVERLAY.\n\n"
                        + "WITHOUT CAMERA PERMISSION, THE APP CANNOT\n"
                        + "FUNCTION. ALL PROCESSING IS ON-DEVICE.\n"
                        + "NO VIDEO OR IMAGES ARE STORED OR TRANSMITTED."
                )
                .setPositiveButton("GRANT", (dialog, which) -> {
                    Log.d(TAG, "User acknowledged rationale — requesting permissions");
                    ActivityCompat.requestPermissions(
                            activity,
                            REQUIRED_PERMISSIONS,
                            REQUEST_CODE_ALL
                    );
                })
                .setNegativeButton("DENY", (dialog, which) -> {
                    Log.w(TAG, "User denied permission rationale");
                    dialog.dismiss();
                })
                .setCancelable(false)
                .show();
    }

    // ================================================================
    // Utility
    // ================================================================

    /**
     * Returns the list of all required permission strings.
     * Useful for displaying which permissions the app needs.
     *
     * @return Array of required permission strings.
     */
    @NonNull
    public static String[] getRequiredPermissions() {
        return REQUIRED_PERMISSIONS.clone();
    }

    /**
     * Returns a human-readable description of the app's permission
     * requirements. Suitable for about/privacy screens.
     *
     * @return Formatted permission description string.
     */
    @NonNull
    public static String getPermissionDescription() {
        return "CAMERA: REQUIRED FOR OBJECT DETECTION\n"
                + "ALL PROCESSING IS ON-DEVICE\n"
                + "NO DATA IS STORED OR TRANSMITTED\n"
                + "NO INTERNET PERMISSION REQUESTED";
    }
}
