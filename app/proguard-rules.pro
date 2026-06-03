# Sync Vision ProGuard Rules

# Keep TFLite model classes
-keep class org.tensorflow.lite.** { *; }
-keep class com.google.ai.edge.litert.** { *; }

# Keep MediaPipe classes
-keep class com.google.mediapipe.** { *; }

# Keep ML Kit classes (on-device)
-keep class com.google.mlkit.** { *; }

# Keep OpenCV native methods
-keep class com.syncvision.app.native.** { *; }
-keepclassmembers class com.syncvision.app.native.** {
    native <methods>;
}

# Keep JNI bridge methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep Room entities
-keep class com.syncvision.app.data.** { *; }

# Keep object description data classes
-keep class com.syncvision.app.ml.InferenceResult** { *; }

# Don't warn about optional / missing libraries
-dontwarn org.tensorflow.lite.gpu.**
-dontwarn com.google.mediapipe.tasks.**
-dontwarn com.google.mediapipe.proto.**
-dontwarn com.google.mediapipe.framework.**
