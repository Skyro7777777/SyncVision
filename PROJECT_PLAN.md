# Sync Vision — Complete Project Plan

> **"Sync the world around you. See what others can't."**
> Android 10+ | Java + C++ (NDK) | Privacy-First | On-Device AI | No API Keys

---

## Table of Contents
1. [Vision & Philosophy](#1-vision--philosophy)
2. [Feature Breakdown](#2-feature-breakdown)
3. [Technology Stack](#3-technology-stack)
4. [ML Models & Libraries](#4-ml-models--libraries)
5. [Architecture Design](#5-architecture-design)
6. [Project Structure](#6-project-structure)
7. [Core Pipeline — Frame Processing](#7-core-pipeline--frame-processing)
8. [Green Line Rendering System](#8-green-line-rendering-system)
9. [Sync Diagram System](#9-sync-diagram-system)
10. [Path & Wayfinding System](#10-path--wayfinding-system)
11. [Weather Awareness System](#11-weather-awareness-system)
12. [E.D.I.T.H-Inspired Features (Real)](#12-edith-inspired-features-real)
13. [Privacy Architecture](#13-privacy-architecture)
14. [Performance Budget](#14-performance-budget)
15. [Build System & GitHub Actions](#15-build-system--github-actions)
16. [Development Phases](#16-development-phases)
17. [Data Collection for Next Prompt](#17-data-collection-for-next-prompt)

---

## 1. Vision & Philosophy

**Sync Vision** is a real-time augmented reality camera app that:
- **Outlines every detected object** with thin terminal-green (#00FF41) lines
- **Names and describes** each object in ALL CAPS small text beside it
- **Syncs environmental context** — weather, paths, hazards, relationships
- **Creates relationship diagrams** of the scene using green lines
- **Suggests optimal paths** when viewing roads, corridors, doorways
- **Works entirely on-device** — no cloud, no API keys, no tracking
- **Is accurate, not fictional** — every feature must actually work

### Design Principles
1. **Real over impressive** — If it can't work reliably, it doesn't ship
2. **Privacy by default** — All processing on-device, no data leaves the phone
3. **Speed matters** — Green outlines must feel instant (< 100ms latency)
4. **Terminal green aesthetic** — #00FF41, thin lines, ALL CAPS labels, monospace feel
5. **Help, not spy** — Detection yes, recognition no (faces detected but never identified)

---

## 2. Feature Breakdown

### Tier 1 — Core (Must Have, Phase 1)

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| F1 | **Object Outline** | Thin green (#00FF41) outline around every detected object using segmentation | Planned |
| F2 | **Object Name Label** | ALL CAPS name label (e.g., "CAR", "PERSON", "DOG") beside each object | Planned |
| F3 | **Object Description** | 2-3 line description of what the object is and what it's used for | Planned |
| F4 | **Real-time Camera** | CameraX-based real-time feed with overlay rendering at 15-30 FPS | Planned |
| F5 | **Green Line Aesthetic** | Terminal green (#00FF41), thin outlines, monospace ALL CAPS text, scan-line effect | Planned |

### Tier 2 — Sync Intelligence (Phase 2)

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| F6 | **Sync Diagram** | Small floating diagram showing relationships between detected objects via green lines | Planned |
| F7 | **Path Analysis** | When viewing roads/corridors/doorways, highlights clear/better path with green diagram | Planned |
| F8 | **Weather Awareness** | Analyzes sky/clouds and shows weather condition in ALL CAPS overlay | Planned |
| F9 | **Hazard Detection** | Marks potential hazards (vehicles, obstacles, drops) with warning indicators | Planned |
| F10 | **Distance Estimation** | Shows approximate distance to detected objects using depth estimation | Planned |

### Tier 3 — E.D.I.T.H-Inspired (Phase 3)

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| F11 | **Face Detection + Count** | Detects faces (outlines in green), counts people — NO recognition/identification | Planned |
| F12 | **Text/OCR Overlay** | Reads text in the scene and overlays it with green text, translates on-tap | Planned |
| F13 | **QR/Barcode Scanner** | Detects and decodes QR codes and barcodes with green outline | Planned |
| F14 | **Plant/Species ID** | On-tap: identifies plants, animals, insects using iNaturalist model | Planned |
| F15 | **Landmark Recognition** | On-tap: identifies known landmarks and monuments | Planned |
| F16 | **Threat Assessment** | Priority-based threat scoring: moving vehicles > stationary obstacles > informational | Planned |
| F17 | **Audio Alerts** | Optional audio alerts for high-threat detections (approaching vehicles, etc.) | Planned |
| F18 | **Night Enhancement** | Brightness/contrast boost in low-light for better detection visibility | Planned |
| F19 | **Translation Overlay** | Translate detected text to user's language (on-device ML Kit translation) | Planned |

### NOT Included (Fictional/Unrealistic)
- ❌ Looking into people's messages / private data
- ❌ Accessing any database of personal information
- ❌ Facial recognition (only detection)
- ❌ Network hacking / "Stark network"
- ❌ Holographic projections
- ❌ Predicting exact future events

---

## 3. Technology Stack

### Languages
| Language | Use Case | Justification |
|----------|----------|---------------|
| **Java** | Android framework, UI, CameraX, Activity/Fragment lifecycle | Primary Android language, SDK 29+ |
| **C++ (NDK/JNI)** | Image processing, OpenCV operations, contour extraction, rendering | Performance-critical: 10-50x faster than Java for pixel ops |
| **GLSL** | OpenGL ES 3.0 shaders for green outline rendering, Sobel edge detection | GPU-accelerated overlay rendering |
| **Kotlin** | Optional: CameraX extensions, Coroutines for async | Only if needed for specific AndroidX libraries |

### Android SDK & Build
| Component | Version |
|-----------|---------|
| **minSdkVersion** | 29 (Android 10) |
| **targetSdkVersion** | 35 (Android 15) |
| **compileSdkVersion** | 35 |
| **NDK** | 27.0.12077973 |
| **CMake** | 3.22.1+ |
| **Gradle** | 8.11+ |
| **AGP** | 8.7+ |
| **Java** | 17 |

### Core Libraries
| Library | Version | Purpose | License |
|---------|---------|---------|---------|
| **CameraX** | 1.4.1 | Camera preview + frame analysis | Apache 2.0 |
| **TFLite (LiteRT)** | 1.2.0 | On-device ML inference | Apache 2.0 |
| **MediaPipe Tasks Vision** | 0.10.14 | Face detection (BlazeFace) | Apache 2.0 |
| **ML Kit Text Recognition** | 16.0.1 | On-device OCR | Free (on-device) |
| **ML Kit Translation** | 17.0.3 | On-device text translation | Free (on-device) |
| **ML Kit Barcode Scanning** | 17.3.0 | QR/Barcode detection | Free (on-device) |
| **OpenCV** | 4.10.0 | C++ image processing (core + imgproc only) | Apache 2.0 |
| **OpenGL ES 3.0** | System | Overlay rendering via shaders | — |

---

## 4. ML Models & Libraries

### Model Inventory

| Model | Purpose | Size (INT8) | FPS (SD660) | Source | License |
|-------|---------|-------------|-------------|--------|---------|
| **DeepLab v3+ MobileNet V2** | Multi-class segmentation (21 classes) | 6.7 MB | 8-14 | TF Hub | Apache 2.0 |
| **COCO SSD MobileNet V2** | Object detection (80 classes, bounding boxes) | 3 MB | 18-30 | TF Hub | Apache 2.0 |
| **MiDaS v2.1 Small** | Depth estimation | 8 MB | 5-10 | Intel/MIT | MIT |
| **BlazeFace (short-range)** | Face detection | 0.3 MB | 30-60+ | MediaPipe | Apache 2.0 |
| **Weather Classifier** | Sky/weather classification (custom trained) | 4 MB | 25-40 | Custom (TFLite Model Maker) | Ours |
| **iNaturalist MobileNet** | Plant/animal species ID | 9 MB | 15-25 | TF Hub | Apache 2.0 |
| **Landmark Classifier** | Landmark recognition (custom trained) | 4 MB | 25-40 | Custom (TFLite Model Maker) | Ours |

### Combined Model Size: ~35 MB (INT8 quantized)

### Segmentation Strategy (Dual-Model Approach)
For the best outline quality, we use TWO models in tandem:

1. **DeepLab v3+** → Provides pixel-level semantic masks for contour extraction
2. **COCO SSD** → Provides bounding boxes + class labels (80 classes vs DeepLab's 21)

**Fusion**: Use DeepLab masks for outline rendering, but use COCO SSD class labels for the 80-class naming. When DeepLab detects "car" in its 21 classes, we cross-reference with COCO SSD for more specific labels.

### Object Description Database
Local SQLite/Room database with:
- 80+ COCO class entries
- Each entry: class name, 2-3 line description, common uses, hazard level
- Pre-populated, bundled in APK assets
- Example:
  ```
  CAR: MOTOR VEHICLE WITH FOUR WHEELS
  USED FOR TRANSPORTATION ON ROADS
  HAZARD: MOVING VEHICLE — STAY ALERT
  ```

---

## 5. Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYNC VISION APP                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  CameraX     │───▶│  Frame Dispatcher │───▶│  OpenGL Renderer │  │
│  │  Preview +   │    │  (Java/Kotlin)    │    │  (GLSL Shaders)  │  │
│  │  ImageAnalysis│   │                   │    │  Green outlines  │  │
│  └──────────────┘    │  Routes frames to │    │  Labels overlay  │  │
│                      │  ML pipelines     │    │  Sync diagram    │  │
│                      └───────┬───────────┘    └──────────────────┘  │
│                              │                         ▲            │
│                  ┌───────────┼───────────┐             │            │
│                  ▼           ▼           ▼             │            │
│         ┌──────────┐ ┌──────────┐ ┌──────────┐        │            │
│         │Pipeline 1│ │Pipeline 2│ │Pipeline 3│        │            │
│         │SEGM+DET  │ │DEPTH    │ │ON-DEMAND │        │            │
│         │DeepLab   │ │MiDaS    │ │OCR/Face/ │        │            │
│         │COCO SSD  │ │         │ │Plant/QR  │        │            │
│         │(every frm)│ │(every 3rd│ │(tap trig)│        │            │
│         └────┬─────┘ └────┬─────┘ └────┬─────┘        │            │
│              │            │            │               │            │
│              ▼            ▼            ▼               │            │
│         ┌─────────────────────────────────────┐        │            │
│         │     C++ Native Processing (NDK)      │        │            │
│         │  ┌─────────────────────────────┐    │        │            │
│         │  │ OpenCV (core + imgproc)      │    │        │            │
│         │  │ - findContours()             │    │        │            │
│         │  │ - Canny() edge detection     │    │        │            │
│         │  │ - drawContours()             │    │        │            │
│         │  │ - morphological operations   │    │        │            │
│         │  └─────────────────────────────┘    │        │            │
│         │  ┌─────────────────────────────┐    │        │            │
│         │  │ Contour Simplification       │    │        │            │
│         │  │ - Douglas-Peucker algorithm  │    │        │            │
│         │  │ - Thin outline extraction    │    │        │            │
│         │  └─────────────────────────────┘    │        │            │
│         │  ┌─────────────────────────────┐    │        │            │
│         │  │ Label Placement Engine       │    │        │            │
│         │  │ - Smart positioning          │    │        │            │
│         │  │ - Collision avoidance         │    │        │            │
│         │  │ - Depth-based sizing          │    │        │            │
│         │  └─────────────────────────────┘    │        │            │
│         └─────────────────────────────────────┘        │            │
│              │            │            │               │            │
│              ▼            ▼            ▼               │            │
│         ┌─────────────────────────────────────┐        │            │
│         │     Scene Understanding Engine       │────────┘            │
│         │  - Object fusion (segm + det)        │                    │
│         │  - Relationship extraction           │                    │
│         │  - Path analysis                     │                    │
│         │  - Hazard scoring                    │                    │
│         │  - Weather state                     │                    │
│         │  - Sync diagram generation           │                    │
│         └─────────────────────────────────────┘                    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    UI / Overlay Layer                         │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │  │
│  │  │ Main Camera│  │ Sync       │  │ Info Panel (on-tap)    │ │  │
│  │  │ Overlay    │  │ Diagram    │  │ - Detailed description │ │  │
│  │  │ (OpenGL)   │  │ (Canvas)   │  │ - Distance/depth      │ │  │
│  │  └────────────┘  └────────────┘  │ - Threat level        │ │  │
│  │                                   └────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Thread Architecture (8-core: 4×A73 big + 4×A53 LITTLE)

| Thread | Core | Priority | Task |
|--------|------|----------|------|
| Main/UI | A53 | Normal | Camera preview, touch events, UI updates |
| ML-Segment | A73 big | High | DeepLab v3+ + COCO SSD (every frame) |
| ML-Depth | A73 big | Normal | MiDaS depth (every 3rd frame) |
| ML-OnDemand | A73 big | On-demand | OCR, plant ID, landmarks (tap-triggered) |
| GL-Render | GPU | High | OpenGL ES overlay rendering |
| Native-Proc | A73 big | High | C++ contour extraction, label placement |
| Sync-Engine | A53 | Low | Sync diagram, relationship analysis |

---

## 6. Project Structure

```
sync-vision/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/syncvision/app/
│   │   │   │   ├── SyncVisionApp.kt              # Application class
│   │   │   │   ├── camera/
│   │   │   │   │   ├── CameraManager.java         # CameraX setup & lifecycle
│   │   │   │   │   ├── FrameDispatcher.java       # Routes frames to pipelines
│   │   │   │   │   └── CameraConfig.java          # Resolution, FPS config
│   │   │   │   ├── ml/
│   │   │   │   │   ├── SegmentationPipeline.java  # DeepLab v3+ inference
│   │   │   │   │   ├── DetectionPipeline.java     # COCO SSD inference
│   │   │   │   │   ├── DepthPipeline.java         # MiDaS depth inference
│   │   │   │   │   ├── FacePipeline.java          # BlazeFace inference
│   │   │   │   │   ├── OcrPipeline.java           # ML Kit OCR
│   │   │   │   │   ├── WeatherPipeline.java       # Weather classification
│   │   │   │   │   ├── PlantPipeline.java         # iNaturalist inference
│   │   │   │   │   ├── LandmarkPipeline.java      # Landmark classification
│   │   │   │   │   ├── BarcodePipeline.java       # ML Kit barcode
│   │   │   │   │   ├── ModelManager.java          # Model loading & caching
│   │   │   │   │   └── InferenceResult.java       # Result data classes
│   │   │   │   ├── native/
│   │   │   │   │   ├── NativeProcessor.java       # JNI bridge to C++
│   │   │   │   │   └── NativeConstants.java       # JNI constants
│   │   │   │   ├── rendering/
│   │   │   │   │   ├── GLRenderer.java            # OpenGL ES renderer
│   │   │   │   │   ├── GLSurfaceManager.java      # Surface lifecycle
│   │   │   │   │   ├── OverlayRenderer.java       # High-level overlay API
│   │   │   │   │   └── shader/
│   │   │   │   │       ├── OutlineShader.java     # Green outline shader
│   │   │   │   │       ├── LabelShader.java       # Text label shader
│   │   │   │   │       └── CompositeShader.java   # Camera+overlay composite
│   │   │   │   ├── scene/
│   │   │   │   │   ├── SceneUnderstanding.java    # Scene analysis engine
│   │   │   │   │   ├── ObjectFusion.java          # Segm + Det fusion
│   │   │   │   │   ├── RelationshipExtractor.java # Object relationships
│   │   │   │   │   ├── PathAnalyzer.java          # Path/wayfinding
│   │   │   │   │   ├── HazardScorer.java          # Threat assessment
│   │   │   │   │   ├── WeatherAnalyzer.java       # Weather from sky
│   │   │   │   │   └── SyncDiagramGenerator.java  # Green line diagram
│   │   │   │   ├── ui/
│   │   │   │   │   ├── MainActivity.java          # Main camera activity
│   │   │   │   │   ├── CameraFragment.java        # Camera view fragment
│   │   │   │   │   ├── SyncDiagramView.java       # Custom sync diagram view
│   │   │   │   │   ├── InfoPanelView.java         # Object detail panel
│   │   │   │   │   ├── SettingsActivity.java      # App settings
│   │   │   │   │   └── OverlayView.java           # Canvas overlay for text
│   │   │   │   ├── data/
│   │   │   │   │   ├── ObjectDescription.java     # Room entity
│   │   │   │   │   ├── ObjectDatabase.java        # Room database
│   │   │   │   │   ├── ObjectDao.java             # Room DAO
│   │   │   │   │   └── DescriptionLoader.java     # Load from assets
│   │   │   │   └── util/
│   │   │   │       ├── GreenTheme.java            # #00FF41 constants
│   │   │   │       ├── PerformanceMonitor.java    # FPS/latency tracking
│   │   │   │       └── PermissionHelper.java      # Camera, storage perms
│   │   │   ├── cpp/
│   │   │   │   ├── CMakeLists.txt                 # CMake build config
│   │   │   │   ├── jni_bridge.cpp                 # JNI entry points
│   │   │   │   ├── contour_processor.cpp          # Contour extraction
│   │   │   │   ├── contour_processor.h
│   │   │   │   ├── label_placer.cpp               # Smart label positioning
│   │   │   │   ├── label_placer.h
│   │   │   │   ├── path_finder.cpp                # Path analysis (A* based)
│   │   │   │   ├── path_finder.h
│   │   │   │   ├── sync_diagram.cpp               # Diagram layout engine
│   │   │   │   ├── sync_diagram.h
│   │   │   │   └── opencv_provider.cpp            # OpenCV module loader
│   │   │   ├── assets/
│   │   │   │   ├── models/
│   │   │   │   │   ├── deeplab_v3_mnv2_int8.tflite
│   │   │   │   │   ├── coco_ssd_mnv2_int8.tflite
│   │   │   │   │   ├── midas_v21_small_int8.tflite
│   │   │   │   │   ├── blazeface_short.tflite
│   │   │   │   │   ├── weather_classifier_int8.tflite
│   │   │   │   │   ├── inaturalist_mnv2_int8.tflite
│   │   │   │   │   └── landmark_effnet_int8.tflite
│   │   │   │   ├── data/
│   │   │   │   │   ├── object_descriptions.json   # 80+ COCO class descriptions
│   │   │   │   │   └── hazard_levels.json         # Threat scoring rules
│   │   │   │   └── shaders/
│   │   │   │       ├── outline_vertex.glsl        # Green outline vertex shader
│   │   │   │       ├── outline_fragment.glsl      # Green outline fragment shader
│   │   │   │       ├── composite_vertex.glsl      # Camera+overlay composite
│   │   │   │       ├── composite_fragment.glsl    # Sobel edge + green blend
│   │   │   │       └── label_fragment.glsl        # Text label rendering
│   │   │   ├── res/
│   │   │   │   ├── layout/
│   │   │   │   │   ├── activity_main.xml
│   │   │   │   │   ├── fragment_camera.xml
│   │   │   │   │   ├── activity_settings.xml
│   │   │   │   │   └── view_info_panel.xml
│   │   │   │   ├── values/
│   │   │   │   │   ├── colors.xml                 # #00FF41 as primary
│   │   │   │   │   ├── strings.xml
│   │   │   │   │   ├── themes.xml                 # Dark theme (terminal)
│   │   │   │   │   └── dimens.xml
│   │   │   │   ├── drawable/
│   │   │   │   │   ├── scanline_overlay.xml       # CRT scanline effect
│   │   │   │   │   ├── corner_bracket.xml         # HUD corner brackets
│   │   │   │   │   └── hazard_indicator.xml       # Warning flash drawable
│   │   │   │   ├── raw/
│   │   │   │   │   └── alert_beep.mp3             # Hazard alert sound
│   │   │   │   └── mipmap-*/
│   │   │   │       └── ic_launcher.png            # App icon
│   │   │   └── AndroidManifest.xml
│   │   └── test/                                   # Unit tests
│   └── build.gradle.kts                            # App module build
├── gradle/
│   └── wrapper/
├── .github/
│   └── workflows/
│       └── build.yml                               # GitHub Actions CI/CD
├── build.gradle.kts                                # Root build
├── settings.gradle.kts                             # Project settings
├── gradle.properties                               # Gradle config
├── gradlew                                         # Gradle wrapper
└── README.md
```

---

## 7. Core Pipeline — Frame Processing

### Frame Flow (Every Frame)

```
CameraX ImageAnalysis (640x480 YUV)
        │
        ▼
┌─────────────────┐
│ Frame Dispatcher │  (Java, runs on A53)
│ - Timestamp      │
│ - Frame counter  │
│ - Priority queue │
└───────┬─────────┘
        │
        ├──▶ Pipeline 1: SEGM+DET (every frame, A73 big)
        │    ├─ DeepLab v3+ → segmentation mask (640x480)
        │    ├─ COCO SSD → bounding boxes + class labels (80 classes)
        │    └─ Output: List<DetectedObject> with mask + label + confidence
        │
        ├──▶ Pipeline 2: DEPTH (every 3rd frame, A73 big)
        │    ├─ MiDaS v2.1 → depth map (256x256, upscaled)
        │    └─ Output: float[][] depthMap
        │
        ├──▶ Pipeline 3: FACE (every 2nd frame, A53 LITTLE)
        │    ├─ BlazeFace → face bounding boxes
        │    └─ Output: List<FaceRect> (no identity data)
        │
        └──▶ Pipeline 4: ON-DEMAND (tap-triggered, A73 big)
             ├─ OCR: ML Kit → text blocks with positions
             ├─ Plant: iNaturalist → species + confidence
             ├─ Landmark: Custom → landmark name + confidence
             └─ QR/Barcode: ML Kit → decoded content
        
        │
        ▼
┌──────────────────┐
│ C++ Native Proc  │  (NDK, A73 big)
│ - OpenCV contour  │
│ - Douglas-Peucker │
│ - Label placement │
│ - Path finding    │
│ - Sync diagram    │
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│ Scene Engine     │  (Java, A53)
│ - Object fusion   │
│ - Hazard scoring  │
│ - Weather state   │
│ - Relationships   │
└───────┬──────────┘
        │
        ▼
┌──────────────────┐
│ OpenGL Renderer  │  (GPU, GLThread)
│ - Upload mask tex │
│ - Sobel shader    │
│ - Green outlines  │
│ - Label overlay   │
│ - Sync diagram    │
│ - Path overlay    │
└──────────────────┘
```

### Latency Budget per Frame (target: 33ms / 30 FPS)

| Stage | Time (ms) | Notes |
|-------|-----------|-------|
| CameraX frame capture | 1-2 | Hardware timestamp |
| Frame dispatch | < 1 | Queue push |
| DeepLab inference | 30-60 | Runs async, result available next frame |
| COCO SSD inference | 15-30 | Parallel with DeepLab |
| C++ contour processing | 3-5 | OpenCV findContours |
| Label placement | 1-2 | C++ algorithm |
| OpenGL rendering | 2-3 | GPU composite |
| **Total pipeline (pipelined)** | **~33ms effective** | Results displayed 1 frame behind |

---

## 8. Green Line Rendering System

### OpenGL ES 3.0 Shader Pipeline

The green outline is rendered entirely on the GPU for maximum performance:

**Step 1**: Upload segmentation mask as a texture (R8 format, 1 byte per pixel)
**Step 2**: Fragment shader applies Sobel edge detection on the mask
**Step 3**: Where edge > threshold AND mask > 0, blend green color onto camera feed
**Step 4**: Label text rendered via Canvas on a transparent texture, composited in shader

```glsl
// composite_fragment.glsl — The core green outline shader
precision mediump float;

uniform sampler2D uCameraTexture;   // Camera feed
uniform sampler2D uMaskTexture;     // Segmentation mask
uniform sampler2D uLabelTexture;    // Text labels (transparent bg)
uniform vec2 uTexelSize;            // 1.0/width, 1.0/height
uniform vec3 uGreenColor;           // #00FF41 = vec3(0.0, 1.0, 0.255)

varying vec2 vTexCoord;

float sobelEdge(sampler2D tex, vec2 uv) {
    float tl = texture2D(tex, uv + vec2(-uTexelSize.x, -uTexelSize.y)).r;
    float l  = texture2D(tex, uv + vec2(-uTexelSize.x, 0.0)).r;
    float bl = texture2D(tex, uv + vec2(-uTexelSize.x,  uTexelSize.y)).r;
    float t  = texture2D(tex, uv + vec2(0.0, -uTexelSize.y)).r;
    float b  = texture2D(tex, uv + vec2(0.0,  uTexelSize.y)).r;
    float tr = texture2D(tex, uv + vec2( uTexelSize.x, -uTexelSize.y)).r;
    float r  = texture2D(tex, uv + vec2( uTexelSize.x, 0.0)).r;
    float br = texture2D(tex, uv + vec2( uTexelSize.x,  uTexelSize.y)).r;
    
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;
    return sqrt(gx * gx + gy * gy);
}

void main() {
    vec4 camera = texture2D(uCameraTexture, vTexCoord);
    float mask = texture2D(uMaskTexture, vTexCoord).r;
    vec4 label = texture2D(uLabelTexture, vTexCoord);
    
    // Green outline via Sobel on mask
    float edge = sobelEdge(uMaskTexture, vTexCoord);
    float outline = step(0.15, edge) * step(0.1, mask);
    
    // Composite: camera + green outline + labels
    vec3 result = mix(camera.rgb, uGreenColor, outline * 0.85);
    result = mix(result, label.rgb, label.a);
    
    // Subtle scanline effect for terminal aesthetic
    float scanline = 0.95 + 0.05 * sin(vTexCoord.y * 800.0);
    result *= scanline;
    
    gl_FragColor = vec4(result, 1.0);
}
```

### Label Rendering
- Text labels rendered on Android Canvas (transparent bitmap)
- Uploaded as GL_TEXTURE_2D with alpha blending
- Monospace font, ALL CAPS, terminal green color
- Smart placement: positioned beside object, avoids overlap with other labels
- Depth-based sizing: closer objects get slightly larger text

---

## 9. Sync Diagram System

### What It Shows
A small floating diagram (bottom-right corner) that shows relationships between detected objects using green lines and nodes.

### Diagram Generation Algorithm (C++)
1. Each detected object → node (circle with icon)
2. Spatial relationships → edges (proximity, alignment, containment)
3. Relationship types:
   - "ON" (cup on table)
   - "NEAR" (person near car)
   - "CONTAINS" (building contains door)
   - "BLOCKS" (car blocks path)
   - "SUPPORTS" (pillar supports roof)
4. Diagram uses force-directed layout (simplified)
5. Rendered with green lines on transparent canvas overlay

### Diagram Rendering
```
┌──────────────────┐
│  PERSON ─── CAR  │  ← Green lines connect related objects
│    │         │   │
│  NEAR     BLOCKS │  ← Relationship labels in small ALL CAPS
│    │         │   │
│  DOOR ──── ROAD  │
└──────────────────┘
```

---

## 10. Path & Wayfinding System

### How It Works
1. When the scene contains **roads, corridors, doorways, pathways** (detected via COCO SSD class labels + depth map)
2. MiDaS depth map identifies walkable ground plane
3. C++ path_finder.cpp implements:
   - Ground plane extraction from depth map
   - Obstacle detection (objects blocking path)
   - A* pathfinding on a discretized 2D grid (from depth + segmentation)
   - Optimal path visualization with green arrows
4. Path clarity scoring:
   - CLEAR PATH → solid green line with arrow
   - PARTIAL OBSTRUCTION → dashed green line
   - BLOCKED → red-tinted line with X mark
5. Multiple paths can be shown with clarity ranking

### Visual Output
- Green arrows overlaid on the camera feed showing direction
- Path diagram in sync diagram view
- ALL CAPS label: "CLEAR PATH →" or "OBSTRUCTED — ALTERNATE ROUTE →"

---

## 11. Weather Awareness System

### Sky Analysis Pipeline
1. **Sky Region Detection**: Upper 30-40% of camera frame (configurable)
2. **Weather Classification**: Custom MobileNetV2 TFLite model
   - Classes: CLEAR, PARTLY_CLOUDY, CLOUDY, OVERCAST, RAINY, FOGGY, SUNSET_SUNRISE, STORMY
3. **Cloud Formation Analysis**: 
   - OpenCV contour analysis on sky region (after brightness/contrast normalization)
   - Large flat clouds → stable weather
   - Tall/dark clouds → potential rain
   - Wispy high clouds → fair weather
4. **On-screen Display**: 
   - Top-left corner: "WEATHER: PARTLY CLOUDY" in ALL CAPS green text
   - Confidence percentage shown
5. **Refresh Rate**: Every 2-3 seconds (weather doesn't change fast)

### Training the Weather Model
- Use TFLite Model Maker with Multi-class Weather Dataset (MWD)
- Fine-tune MobileNetV2 → ~4 MB INT8 TFLite
- Training script will be included in project for reproducibility

---

## 12. E.D.I.T.H-Inspired Features (Real)

### F11: Face Detection + Count
- **What**: Detects faces, draws green outline, counts total people in frame
- **How**: MediaPipe BlazeFace (320 KB model, 30-60 FPS)
- **Display**: "3 FACES DETECTED" in top-right, green outlines on each face
- **Privacy**: No facial recognition, no identity data, no storage

### F12: Text/OCR Overlay
- **What**: Reads text in scene, overlays green highlighted version
- **How**: ML Kit Text Recognition V2 (on-device, no API key)
- **Display**: Detected text shown in green with bounding box
- **On-tap**: Translates to user's language (ML Kit on-device translation)

### F13: QR/Barcode Scanner
- **What**: Auto-detects QR codes and barcodes, outlines in green
- **How**: ML Kit Barcode Scanning (on-device, no API key)
- **Display**: Green outline + decoded content shown in ALL CAPS

### F14: Plant/Species ID
- **What**: Tap on a plant/animal to identify species
- **How**: iNaturalist MobileNet TFLite (7,000+ species, 9 MB)
- **Display**: Species name in green, confidence %, brief description
- **On-tap only**: Not running continuously (saves battery)

### F15: Landmark Recognition
- **What**: Tap on a building/monument to identify it
- **How**: Custom EfficientNet-Lite0 trained on Google Landmarks V2
- **Display**: Landmark name in green, brief historical info

### F16: Threat Assessment
- **What**: Priority-based scoring of potential hazards
- **How**: Rule-based engine + object detection
- **Scoring**:
  - HIGH: Moving vehicle nearby, fire/smoke, weapon-like objects
  - MEDIUM: Obstacles in path, steep drops (from depth), broken glass
  - LOW: Uneven surface, low ceiling, wet floor
- **Display**: Pulsing green/red indicator + "CAUTION: VEHICLE APPROACHING"

### F17: Audio Alerts
- **What**: Optional beep/vibration for high-threat detections
- **How**: Android MediaPlayer + Vibrator
- **Trigger**: Only for HIGH threat level objects
- **Privacy**: Completely local, no network

### F18: Night Enhancement
- **What**: Brightness/contrast boost in low-light conditions
- **How**: OpenGL ES shader adjustment based on average frame luminance
- **Display**: Auto-activates below certain lux threshold, "NIGHT MODE ACTIVE" label

### F19: Translation Overlay
- **What**: Translate detected foreign text to user's language
- **How**: ML Kit on-device Translation (downloadable language packs)
- **Display**: Original text with green underline, translated text below in slightly dimmer green
- **Languages**: Supports 50+ languages via on-device models

---

## 13. Privacy Architecture

### Principles
1. **Zero Cloud**: No data ever leaves the device
2. **No Accounts**: No sign-up, no user accounts, no tracking
3. **No Analytics**: No Google Analytics, no Firebase, no telemetry
4. **No Storage**: Camera frames are processed and discarded immediately
5. **No Face Recognition**: Faces are detected (green outline + count) but NEVER identified
6. **No Network**: App works fully offline after model download (models bundled in APK)

### Implementation
- `android.permission.INTERNET` **NOT** in AndroidManifest.xml
- No Firebase, no Crashlytics, no Google Analytics
- Camera frames held in memory only during processing, then released
- Object descriptions database is local SQLite, never synced
- User settings stored in SharedPreferences only
- Optional: Add network security config to block all outbound traffic

```xml
<!-- res/xml/network_security_config.xml -->
<network-security-config>
    <base-config cleartextTrafficPermitted="false">
        <trust-anchors>
            <certificates src="system" />
        </trust-anchors>
    </base-config>
</network-security-config>
```

---

## 14. Performance Budget

### Target: Mid-range device (Snapdragon 660, 4GB RAM, Android 10)

| Metric | Target | Acceptable | Critical |
|--------|--------|------------|----------|
| FPS (outline mode) | 20-30 | 15-20 | < 10 |
| FPS (full analysis) | 12-18 | 8-12 | < 5 |
| Latency (detection → display) | < 80ms | < 120ms | < 200ms |
| RAM usage | < 500 MB | < 700 MB | < 1 GB |
| Battery drain (30 min use) | < 8% | < 12% | < 20% |
| APK size | < 80 MB | < 120 MB | < 200 MB |
| Cold start | < 2s | < 4s | < 6s |
| Model load time | < 500ms | < 1s | < 2s |

### Optimization Strategies
1. **TFLite INT8 quantization** on all models
2. **TFLite GPU delegate** for DeepLab + COCO SSD
3. **Frame skipping**: Depth runs every 3rd frame, face every 2nd
4. **Resolution reduction**: ML models process 320x320 or 257x257
5. **Neural Networks API (NNAPI)**: Use on supported devices for hardware acceleration
6. **Memory pooling**: Reuse byte buffers for frames, avoid GC pressure
7. **Lazy model loading**: Only load models when feature is activated

---

## 15. Build System & GitHub Actions

### GitHub Actions Workflow (`build.yml`)

```yaml
name: Build Sync Vision APK

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:  # Manual trigger

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up JDK 17
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'
      
      - name: Setup Android SDK
        uses: android-actions/setup-android@v3
      
      - name: Cache Gradle packages
        uses: actions/cache@v4
        with:
          path: |
            ~/.gradle/caches
            ~/.gradle/wrapper
          key: gradle-${{ hashFiles('**/*.gradle.kts', '**/gradle-wrapper.properties') }}
          restore-keys: gradle-
      
      - name: Cache NDK
        uses: actions/cache@v4
        with:
          path: /usr/local/lib/android/sdk/ndk/
          key: ndk-27.0.12077973
      
      - name: Grant execute permission for gradlew
        run: chmod +x gradlew
      
      - name: Build Debug APK
        run: ./gradlew assembleDebug
      
      - name: Build Release APK
        run: ./gradlew assembleRelease
      
      - name: Upload Debug APK
        uses: actions/upload-artifact@v4
        with:
          name: sync-vision-debug
          path: app/build/outputs/apk/debug/*.apk
      
      - name: Upload Release APK
        uses: actions/upload-artifact@v4
        with:
          name: sync-vision-release
          path: app/build/outputs/apk/release/*.apk
      
      - name: Run Lint
        run: ./gradlew lint
      
      - name: Upload Lint Report
        uses: actions/upload-artifact@v4
        with:
          name: lint-report
          path: app/build/reports/lint-results-debug.html
```

### Signing (for release builds)
- Use GitHub Secrets for keystore
- Keystore password, key alias, key password stored as secrets
- Keystore file encoded as base64 in secrets

---

## 16. Development Phases

### Phase 1: Foundation (Weeks 1-3)
**Goal**: Working camera with green outlines on objects

| Step | Task | Details |
|------|------|---------|
| 1.1 | Project scaffolding | Gradle, NDK, CMake, dependencies |
| 1.2 | CameraX integration | Preview + ImageAnalysis at 640x480 |
| 1.3 | OpenGL ES surface | Camera feed rendering via GLSurfaceView |
| 1.4 | DeepLab v3+ integration | Load model, run inference, get mask |
| 1.5 | COCO SSD integration | Load model, get bounding boxes + labels |
| 1.6 | Green outline shader | Sobel on mask → green outline on camera |
| 1.7 | Object name labels | ALL CAPS name beside each object |
| 1.8 | Description database | Load object_descriptions.json, show 2-3 line desc |
| 1.9 | Terminal green theme | #00FF41, scanline effect, HUD brackets |

**Deliverable**: Camera app that outlines objects in green with names and descriptions

### Phase 2: Sync Intelligence (Weeks 4-6)
**Goal**: Scene understanding, paths, weather, hazards

| Step | Task | Details |
|------|------|---------|
| 2.1 | MiDaS depth integration | Depth map for distance estimation |
| 2.2 | Path analysis engine | Ground plane extraction, A* pathfinding |
| 2.3 | Path visualization | Green arrows for clear/obstructed paths |
| 2.4 | Weather classification | Sky region analysis, weather model |
| 2.5 | Hazard scoring | Rule-based threat assessment |
| 2.6 | Sync diagram | Relationship extraction, force-directed graph |
| 2.7 | Distance labels | Approximate distance on each object |

**Deliverable**: App that shows paths, weather, hazards, and object relationships

### Phase 3: E.D.I.T.H Features (Weeks 7-9)
**Goal**: Full feature set with all E.D.I.T.H-inspired capabilities

| Step | Task | Details |
|------|------|---------|
| 3.1 | Face detection | BlazeFace integration, face count |
| 3.2 | OCR overlay | ML Kit text recognition, text highlighting |
| 3.3 | QR/Barcode scanner | ML Kit barcode detection |
| 3.4 | Plant/Species ID | iNaturalist model, on-tap identification |
| 3.5 | Landmark recognition | Custom model, on-tap identification |
| 3.6 | Translation overlay | ML Kit on-device translation |
| 3.7 | Audio alerts | Threat-level audio feedback |
| 3.8 | Night enhancement | Shader-based brightness boost |

**Deliverable**: Full-featured app with all E.D.I.T.H-inspired capabilities

### Phase 4: Polish & Performance (Weeks 10-12)
**Goal**: Production-quality app

| Step | Task | Details |
|------|------|---------|
| 4.1 | Performance optimization | TFLite GPU delegate, NNAPI, frame skipping |
| 4.2 | UI polish | Settings screen, feature toggles, tutorials |
| 4.3 | Battery optimization | Efficient scheduling, model warm-up |
| 4.4 | Edge case handling | Low light, fast motion, occlusion |
| 4.5 | Accessibility | TalkBack support, font scaling |
| 4.6 | Testing | Device testing on multiple Android versions |
| 4.7 | GitHub Actions CI | Automated build pipeline |

**Deliverable**: Production-ready APK with automated build

---

## 17. Data Collection for Next Prompt

When the user says **"Start building"**, I will use the following data:

### Key Decisions Made
- **Language**: Java (primary) + C++ (NDK) + GLSL (shaders)
- **minSdk**: 29 (Android 10)
- **Camera**: CameraX 1.4.1
- **ML Runtime**: TFLite (LiteRT) 1.2.0 with GPU delegate
- **Rendering**: OpenGL ES 3.0 with custom fragment shaders
- **Segmentation**: DeepLab v3+ MobileNet V2 (INT8, 6.7 MB)
- **Detection**: COCO SSD MobileNet V2 (INT8, 3 MB)
- **Depth**: MiDaS v2.1 Small (INT8, 8 MB)
- **Face**: MediaPipe BlazeFace (320 KB)
- **OCR**: ML Kit Text V2 (on-device)
- **Weather**: Custom MobileNetV2 (TFLite Model Maker, INT8, ~4 MB)
- **Plant ID**: iNaturalist MobileNet (INT8, 9 MB)
- **Landmark**: Custom EfficientNet-Lite0 (INT8, ~4 MB)
- **Image Processing**: OpenCV 4.10 (core + imgproc modules only)
- **Color**: #00FF41 (terminal green)
- **Privacy**: No INTERNET permission, no cloud, no tracking

### Build Order for Phase 1
1. Gradle project with NDK + CMake setup
2. CameraX preview + ImageAnalysis
3. OpenGL ES rendering surface
4. DeepLab v3+ model loading + inference
5. COCO SSD model loading + inference
6. Green outline shader (Sobel on mask)
7. Label rendering (Canvas → texture → GL composite)
8. Object description database (JSON → Room)
9. Terminal green theme (scanlines, HUD brackets)
10. AndroidManifest + permissions + theme

### Models to Bundle
All TFLite models will be bundled in `assets/models/`. For Phase 1, we need:
- `deeplab_v3_mnv2_int8.tflite` (6.7 MB)
- `coco_ssd_mnv2_int8.tflite` (3 MB)

### Critical Files to Create First
1. `build.gradle.kts` (root + app)
2. `AndroidManifest.xml`
3. `CMakeLists.txt`
4. `MainActivity.java`
5. `CameraManager.java`
6. `GLRenderer.java`
7. `SegmentationPipeline.java`
8. `DetectionPipeline.java`
9. `NativeProcessor.java` (JNI bridge)
10. `jni_bridge.cpp`
11. `composite_fragment.glsl`
12. `object_descriptions.json`

---

*Plan created: 2025-07-09*
*Status: READY TO BUILD — Awaiting user's "Start building" command*
