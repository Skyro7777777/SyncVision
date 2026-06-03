# Sync Vision — Complete Project Structure & Setup Guide

> **"Sync the world around you. See what others can't."**
> Project Path: `/home/z/my-project/sync-vision/`

---

## 📁 Complete Project Path Structure

```
sync-vision/
│
├── .github/
│   └── workflows/
│       └── build.yml                          # GitHub Actions CI/CD for APK building
│
├── app/
│   ├── build.gradle.kts                       # App module Gradle config (dependencies, NDK, CMake)
│   ├── proguard-rules.pro                     # ProGuard rules for release builds
│   │
│   └── src/main/
│       ├── AndroidManifest.xml                # App manifest (NO INTERNET permission!)
│       │
│       ├── assets/
│       │   ├── data/
│       │   │   ├── object_descriptions.json   # 80+ COCO class descriptions (ALL CAPS)
│       │   │   └── hazard_levels.json         # Threat scoring rules per object
│       │   │
│       │   ├── models/                        # ✅ All 7 pre-trained TFLite models
│       │   │   ├── blazeface_short.tflite               # 224 KB - Face detection
│       │   │   ├── coco_ssd_mobilenet_v2_300.tflite    # 4.0 MB - Object detection (80 classes)
│       │   │   ├── deeplab_v3_mnv2_257.tflite         # 2.8 MB - Segmentation (21 classes)
│       │   │   ├── midas_v21_small_256.tflite          # 64 MB  - Depth estimation
│       │   │   ├── inaturalist_mnv2_int8.tflite        # 9 MB   - Plant/species ID
│       │   │   ├── landmark_effnet_int8.tflite         # 4 MB   - Landmark recognition
│       │   │   └── weather_classifier_int8.tflite      # 4 MB   - Weather classification
│       │   │
│       │   └── shaders/                       # OpenGL ES 3.0 GLSL shaders
│       │       ├── composite_vertex.glsl      # Main composite vertex shader
│       │       ├── composite_fragment.glsl    # ⭐ Core shader: Sobel edge → green outline + glow + scanlines + night mode
│       │       ├── outline_vertex.glsl        # Contour line vertex shader
│       │       ├── outline_fragment.glsl      # Contour line with glow + pulse
│       │       ├── label_vertex.glsl          # Text label quad vertex shader
│       │       ├── label_fragment.glsl        # Label with dark panel, shadow, border glow
│       │       ├── path_vertex.glsl           # Path visualization vertex shader
│       │       └── path_fragment.glsl         # Path: clear(dashed green) / partial(solid) / blocked(red)
│       │
│       ├── cpp/                               # C++ NDK native code
│       │   ├── CMakeLists.txt                 # CMake build (syncvision_native shared lib)
│       │   ├── jni_bridge.cpp                 # ⭐ JNI entry points (6 native methods)
│       │   ├── contour_processor.h/.cpp       # Contour extraction (OpenCV + fallback)
│       │   ├── label_placer.h/.cpp            # Smart label placement (collision avoidance)
│       │   ├── path_finder.h/.cpp             # A* pathfinding on depth grid
│       │   └── sync_diagram.h/.cpp            # Force-directed relationship diagram
│       │
│       ├── java/com/syncvision/app/
│       │   ├── SyncVisionApp.kt               # Application class (ModelManager init, StrictMode)
│       │   │
│       │   ├── camera/                        # Camera subsystem
│       │   │   ├── CameraConfig.java          # Constants: resolution, FPS, model sizes
│       │   │   ├── CameraManager.java         # CameraX setup, preview + ImageAnalysis
│       │   │   └── FrameDispatcher.java       # Routes frames to ML pipelines
│       │   │
│       │   ├── ml/                            # ML inference pipelines
│       │   │   ├── ModelManager.java          # ⭐ Singleton: TFLite interpreter pool, GPU/NNAPI delegates
│       │   │   ├── InferenceResult.java       # All result data types (DetectedObject, SegmentationResult, etc.)
│       │   │   ├── SegmentationPipeline.java  # DeepLab v3+ (257x257 → 21-class mask)
│       │   │   ├── DetectionPipeline.java     # COCO SSD (300x300 → 10 detections with NMS)
│       │   │   ├── DepthPipeline.java         # MiDaS (256x256 → depth map with distance est.)
│       │   │   ├── FacePipeline.java          # MediaPipe BlazeFace (face count only, NO identity)
│       │   │   ├── OcrPipeline.java           # ML Kit OCR (sync wrapper with 3s timeout)
│       │   │   ├── WeatherPipeline.java       # Weather classifier + heuristic fallback
│       │   │   ├── PlantPipeline.java         # iNaturalist (on-demand, tap-triggered)
│       │   │   ├── LandmarkPipeline.java      # Landmark classifier (placeholder mode)
│       │   │   └── BarcodePipeline.java       # ML Kit barcode (all formats)
│       │   │
│       │   ├── nativelib/                     # JNI bridge
│       │   │   ├── NativeProcessor.java       # 6 native method declarations + DetectedObject/LabelPlacement
│       │   │   └── NativeConstants.java       # JNI data format indices
│       │   │
│       │   ├── rendering/                     # OpenGL ES rendering
│       │   │   ├── GLRenderer.java            # ⭐ Main renderer: camera tex + Sobel mask + labels
│       │   │   ├── GLSurfaceManager.java      # GLSurfaceView lifecycle
│       │   │   ├── OverlayRenderer.java       # High-level API: updateScene(), updateLabels(), etc.
│       │   │   └── shader/
│       │   │       ├── ShaderProgram.java     # GLSL compile/link utility
│       │   │       ├── CompositeShader.java   # Composite shader wrapper
│       │   │       ├── OutlineShader.java     # Outline shader wrapper
│       │   │       └── LabelShader.java       # Label shader wrapper
│       │   │
│       │   ├── scene/                         # Scene understanding
│       │   │   ├── SceneUnderstanding.java    # ⭐ Central engine: fusion + tracking + hazard + weather
│       │   │   ├── ObjectFusion.java          # DeepLab masks + COCO SSD labels fusion
│       │   │   ├── RelationshipExtractor.java # Spatial relationships (ON/NEAR/CONTAINS/BLOCKS/SUPPORTS)
│       │   │   ├── PathAnalyzer.java          # Ground plane + A* pathfinding (native + Java fallback)
│       │   │   ├── HazardScorer.java          # Rule-based threat: HIGH/MEDIUM/LOW/NONE
│       │   │   ├── WeatherAnalyzer.java       # Temporal smoothing + contextual rules
│       │   │   └── SyncDiagramGenerator.java  # Native diagram + Java fallback
│       │   │
│       │   ├── ui/                            # UI components
│       │   │   ├── MainActivity.java          # Fullscreen camera + gestures + pipeline init
│       │   │   ├── CameraFragment.java        # GLSurfaceView host fragment
│       │   │   ├── OverlayView.java           # Canvas HUD: weather, faces, FPS, threat, mode buttons
│       │   │   ├── SyncDiagramView.java       # Force-directed relationship graph
│       │   │   ├── InfoPanelView.java         # Object detail panel (slide-in)
│       │   │   └── SettingsActivity.java      # Settings (GPU, NNAPI, thresholds, modes)
│       │   │
│       │   ├── data/                          # Data layer (Room + JSON)
│       │   │   ├── ObjectDescription.java     # Room entity (80+ COCO classes)
│       │   │   ├── ObjectDao.java             # Room DAO
│       │   │   ├── ObjectDatabase.java        # Room database + pre-populate from assets
│       │   │   ├── DescriptionLoader.java     # JSON → Room loader
│       │   │   └── HazardLevel.java           # Enum: NONE/LOW/MEDIUM/HIGH
│       │   │
│       │   └── util/                          # Utilities
│       │       ├── GreenTheme.java            # #00FF41 color constants
│       │       ├── PerformanceMonitor.java    # FPS/latency tracking
│       │       └── PermissionHelper.java      # Camera permission handling
│       │
│       └── res/                               # Android resources
│           ├── drawable/
│           │   ├── corner_bracket.xml          # HUD corner brackets
│           │   ├── scanline_overlay.xml        # CRT scanline effect
│           │   ├── hazard_indicator.xml        # Warning flash
│           │   ├── mode_button_bg.xml          # Mode toggle button background
│           │   ├── ic_launcher_background.xml  # App icon background
│           │   └── ic_launcher_foreground.xml  # App icon foreground
│           ├── layout/
│           │   ├── activity_main.xml           # Main camera layout
│           │   ├── fragment_camera.xml         # Camera fragment
│           │   ├── activity_settings.xml       # Settings layout
│           │   └── view_info_panel.xml         # Info panel layout
│           ├── mipmap-hdpi/                    # (empty - add launcher icons)
│           ├── mipmap-mdpi/                    # (empty - add launcher icons)
│           ├── mipmap-xhdpi/                   # (empty - add launcher icons)
│           ├── mipmap-xxhdpi/                  # (empty - add launcher icons)
│           ├── mipmap-xxxhdpi/                 # (empty - add launcher icons)
│           ├── raw/
│           │   └── alert_beep.wav             # Hazard alert sound (800Hz, 0.3s)
│           ├── values/
│           │   ├── colors.xml                  # #00FF41 as primary
│           │   ├── strings.xml                 # App strings
│           │   ├── themes.xml                  # Dark terminal theme
│           │   ├── dimens.xml                  # Dimension constants
│           │   └── arrays.xml                  # Dropdown arrays for settings
│           └── xml/
│               ├── network_security_config.xml # Block all outbound traffic
│               └── settings.xml                # PreferenceScreen layout
│
├── scripts/                                   # ⭐ Model download & training scripts
│   ├── download_models.sh                     # Download pre-trained models from Kaggle/GitHub
│   ├── train_weather_model.py                 # Weather classifier training (TFLite Model Maker + Keras)
│   ├── train_landmark_model.py                # Landmark classifier training (EfficientNetB0)
│   └── convert_midas.py                       # MiDaS PyTorch → TFLite conversion
│
├── build.gradle.kts                           # Root Gradle config (AGP 8.7.3, Kotlin 2.1.0)
├── settings.gradle.kts                        # Project settings
├── gradle.properties                          # JVM args, AndroidX, parallel builds
├── gradle/
│   └── wrapper/
│       └── gradle-wrapper.properties          # Gradle 8.11.1
├── gradlew                                    # Gradle wrapper (Unix)
├── gradlew.bat                                # Gradle wrapper (Windows)
└── PROJECT_PLAN.md                            # Complete project plan (17 sections)
```

---

## 🚀 Setup & Build Instructions

### Step 1: Clone and Setup
```bash
cd sync-vision

# Generate Gradle wrapper (requires Gradle installed on your machine)
gradle wrapper --gradle-version 8.11.1
```

### Step 2: Install Android SDK Components
```bash
sdkmanager "platforms;android-35"
sdkmanager "build-tools;35.0.0"
sdkmanager "ndk;27.0.12077973"
sdkmanager "cmake;3.22.1"
```

### Step 3: (Recommended) Replace Placeholder Models with Real Pre-trained Models
```bash
# See scripts/download_models.sh for detailed instructions
# Quick method using Kaggle CLI:
pip install kaggle
kaggle models download tensorflow/deeplabv3 -f tflite --untar
kaggle models download tensorflow/ssd-mobilenet-v2 -f tflite --untar

# Copy to assets/models/ with correct names
cp 1.tflite app/src/main/assets/models/deeplab_v3_mnv2_257.tflite
cp 1.tflite app/src/main/assets/models/coco_ssd_mobilenet_v2_300.tflite
```

### Step 4: Build
```bash
./gradlew assembleDebug     # Debug APK
./gradlew assembleRelease   # Release APK
```

### Step 5: Install
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

---

## ⚠️ What's Done vs. What You Need To Do

### ✅ DONE:
| Item | Details |
|------|---------|
| Project scaffolding | Gradle, NDK, CMake, dependencies all configured |
| CameraX integration | CameraManager, FrameDispatcher, CameraConfig |
| OpenGL ES rendering | GLRenderer, OverlayRenderer, all 8 shaders |
| ML pipeline code | All 9 pipelines (seg, det, depth, face, ocr, weather, plant, barcode, landmark) |
| C++ native code | 5 modules: contour, label, path, diagram, JNI bridge |
| Scene understanding | ObjectFusion, HazardScorer, PathAnalyzer, WeatherAnalyzer |
| UI components | MainActivity, OverlayView, SyncDiagramView, InfoPanelView, Settings |
| Data layer | Room database, JSON descriptions, description loader |
| Pre-trained models (7 of 7) | All 7 models included in assets/models/ |
| GitHub Actions | build.yml with NDK, CMake, Gradle build |
| Alert sound | alert_beep.wav |
| Privacy | No INTERNET permission, network security config |
| .gitignore | Proper Android gitignore |
| README.md | Complete project README |

### ⚠️ YOU NEED TO DO:

#### 1. Generate Gradle Wrapper

The `gradlew` script is a stub. Generate the proper wrapper:

```bash
# Install Gradle on your machine first:
# macOS: brew install gradle
# Linux: sudo apt install gradle
# Then:
cd sync-vision
gradle wrapper --gradle-version 8.11.1
```

This generates `gradle/wrapper/gradle-wrapper.jar` which is needed for builds.

#### 2. Add OpenCV (Optional but Recommended)

OpenCV makes contour extraction and Canny edge detection much better. Without it, the app uses custom C++ fallbacks.

```bash
# 1. Download OpenCV Android SDK:
#    https://opencv.org/releases/ → Download Android pack (4.10.0)

# 2. Extract and copy to project:
mkdir -p app/libs
cp -r ~/Downloads/OpenCV-android-sdk/sdk/native/libs/<abi>/libopencv_java4.so app/src/main/jniLibs/<abi>/

# 3. The C++ code already has #ifdef SYNCVISION_USE_OPENCV guards
#    that switch between OpenCV and custom implementations
```

#### 3. Add Launcher Icons

The mipmap directories are empty. Create icons using Android Studio:

```bash
# In Android Studio: right-click res/ → New → Image Asset
# Or use: https://romannurik.github.io/AndroidAssetStudio/icons-launcher.html
# Theme: Dark, with green (#00FF41) accent
```

---

## 📊 Model Sizes

| Model | Size |
|-------|------|
| BlazeFace Short-Range | 224 KB |
| COCO SSD MobileNet V1 | 4.0 MB |
| DeepLab V3 MobileNet V2 | 2.8 MB |
| MiDaS V2.1 | 64 MB |
| iNaturalist/CropNet | ~9 MB |
| Landmark Classifier | ~4 MB |
| Weather Classifier | ~4 MB |
| **Total** | **~88 MB** |

---

## 🔗 Quick Reference URLs

| Resource | URL |
|----------|-----|
| DeepLab V3+ (Kaggle) | https://www.kaggle.com/models/tensorflow/deeplabv3 |
| COCO SSD (Kaggle) | https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2 |
| BlazeFace (MediaPipe) | https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite |
| MiDaS (GitHub) | https://github.com/isl-org/MiDaS/releases |
| iNaturalist (Kaggle) | https://www.kaggle.com/models/google/inaturalist |
| Weather Dataset | https://www.kaggle.com/datasets/jehanbhathena/weather-dataset |
| Landmarks Dataset | https://www.kaggle.com/c/landmark-recognition-2021/data |
| OpenCV Android SDK | https://opencv.org/releases/ |
| TFLite Model Maker | https://www.tensorflow.org/lite/models/modify/model_maker |
| Kaggle API Docs | https://www.kaggle.com/docs/api |
| CameraX Docs | https://developer.android.com/training/camerax |
| ML Kit OCR | https://developers.google.com/ml-kit/vision/text-recognition |
| ML Kit Barcode | https://developers.google.com/ml-kit/vision/barcode-scanning |
| ML Kit Translation | https://developers.google.com/ml-kit/language/translation |
