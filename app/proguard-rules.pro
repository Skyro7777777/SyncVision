# Sync Vision ProGuard Rules
# These rules prevent R8 from stripping or obfuscating classes that are
# accessed via reflection (JNI, Room, Gson, ML Kit, etc.)

# ============================================================
# TFLite / LiteRT
# ============================================================
-keep class org.tensorflow.lite.** { *; }
-keep class com.google.ai.edge.litert.** { *; }

# ============================================================
# MediaPipe
# ============================================================
-keep class com.google.mediapipe.** { *; }

# ============================================================
# ML Kit (on-device, no API key)
# ============================================================
-keep class com.google.mlkit.** { *; }

# ============================================================
# JNI — Native method classes and inner classes accessed from C++
# ============================================================
-keep class com.syncvision.app.nativelib.** { *; }
-keepclassmembers class com.syncvision.app.nativelib.** {
    native <methods>;
}

# Keep all classes with native methods (generic catch-all)
-keepclasseswithmembernames class * {
    native <methods>;
}

# ============================================================
# Room Database — entities, DAOs, and database class
# ============================================================
-keep class com.syncvision.app.data.** { *; }

# ============================================================
# InferenceResult and all inner classes
# Accessed via reflection by Gson, JNI, and ML pipelines
# ============================================================
-keep class com.syncvision.app.ml.InferenceResult { *; }
-keep class com.syncvision.app.ml.InferenceResult$* { *; }

# ============================================================
# OverlayRenderer.LabelInfo — accessed in rendering pipeline
# ============================================================
-keep class com.syncvision.app.rendering.OverlayRenderer$LabelInfo { *; }

# ============================================================
# All ML pipeline classes — keep them from being stripped
# They are instantiated via reflection-like patterns and
# referenced across pipeline boundaries
# ============================================================
-keep class com.syncvision.app.ml.** { *; }

# ============================================================
# Application class — referenced from AndroidManifest
# ============================================================
-keep class com.syncvision.app.SyncVisionApp { *; }

# ============================================================
# Activity and Fragment classes — referenced from AndroidManifest
# and instantiated by the Android framework
# ============================================================
-keep class com.syncvision.app.ui.MainActivity { *; }
-keep class com.syncvision.app.ui.SettingsActivity { *; }
-keep class com.syncvision.app.ui.CameraFragment { *; }

# ============================================================
# Custom Views — instantiated programmatically and by framework
# ============================================================
-keep class com.syncvision.app.ui.OverlayView { *; }
-keep class com.syncvision.app.ui.InfoPanelView { *; }
-keep class com.syncvision.app.ui.SyncDiagramView { *; }

# ============================================================
# Camera and rendering classes
# ============================================================
-keep class com.syncvision.app.camera.** { *; }
-keep class com.syncvision.app.rendering.** { *; }
-keep class com.syncvision.app.scene.** { *; }
-keep class com.syncvision.app.util.** { *; }

# ============================================================
# Gson — prevent obfuscation of field names used in JSON
# ============================================================
-keepattributes Signature
-keepattributes *Annotation*
-keep class com.google.gson.** { *; }

# Keep any class with @SerializedName annotations
-keepclassmembers,allowobfuscation class * {
    @com.google.gson.annotations.SerializedName <fields>;
}

# Keep all data classes used with Gson (our model classes)
-keep class com.syncvision.app.data.ObjectDescription { *; }
-keep class com.syncvision.app.data.HazardLevel { *; }

# ============================================================
# Dontwarn for optional / missing libraries
# ============================================================
-dontwarn org.tensorflow.lite.gpu.**
-dontwarn com.google.mediapipe.tasks.**
-dontwarn com.google.mediapipe.proto.**
-dontwarn com.google.mediapipe.framework.**
-dontwarn androidx.room.**

# ============================================================
# Keep source file names and line numbers for crash logs
# ============================================================
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile
