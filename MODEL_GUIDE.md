# Sync Vision — Pre-Trained Model Guide

> All 7 models are included in `app/src/main/assets/models/` and are real, properly pre-trained models from official sources.

---

## Model Overview

| # | Model File | Purpose | Size | Source |
|---|-----------|---------|------|--------|
| 1 | `blazeface_short.tflite` | Face Detection | ~224 KB | MediaPipe / Google |
| 2 | `coco_ssd_mobilenet_v2_300.tflite` | Object Detection (80 classes) | ~4.0 MB | TensorFlow / Google |
| 3 | `deeplab_v3_mnv2_257.tflite` | Segmentation (21 classes) | ~2.8 MB | TF Hub / Kaggle |
| 4 | `midas_v21_small_256.tflite` | Depth Estimation | ~64 MB | Intel ISL / GitHub |
| 5 | `inaturalist_mnv2_int8.tflite` | Plant & Species ID | ~9 MB | Kaggle / Google |
| 6 | `landmark_effnet_int8.tflite` | Landmark Recognition | ~4 MB | Kaggle / Google |
| 7 | `weather_classifier_int8.tflite` | Sky/Weather Classification | ~4 MB | Custom trained |

**Total: ~88 MB**

---

## Model Details

### 1. BlazeFace Short-Range — Face Detection

| Property | Value |
|----------|-------|
| **File** | `blazeface_short.tflite` |
| **Size** | ~224 KB |
| **Source** | MediaPipe / Google |
| **Input** | 128×128 RGB |
| **Output** | Bounding boxes + 6 face keypoints |
| **License** | Apache 2.0 |

Detects faces in selfie-range images (within ~2 meters). Returns face bounding boxes and 6 keypoints (left eye, right eye, nose tip, mouth center, left ear, right ear). Runs at 30-60+ FPS on mobile.

---

### 2. COCO SSD MobileNet V1 Quantized — Object Detection

| Property | Value |
|----------|-------|
| **File** | `coco_ssd_mobilenet_v2_300.tflite` |
| **Size** | ~4.0 MB |
| **Source** | TensorFlow / Google |
| **Input** | 300×300 RGB |
| **Output** | Bounding boxes + class labels + confidence scores |
| **License** | Apache 2.0 |

**80 COCO classes**: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

INT8 quantized for fast mobile inference (18-30 FPS on Snapdragon 660). The backbone of Sync Vision's object labeling system.

---

### 3. DeepLab V3 MobileNet V2 INT8 — Semantic Segmentation

| Property | Value |
|----------|-------|
| **File** | `deeplab_v3_mnv2_257.tflite` |
| **Size** | ~2.8 MB |
| **Source** | TensorFlow Hub / Kaggle (Google) |
| **Input** | 257×257 RGB |
| **Output** | Per-pixel segmentation mask (21 classes) |
| **License** | Apache 2.0 |

**21 classes**: 0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv.

This is the core of Sync Vision's **green outline** rendering — boundaries of each segmented region are extracted using OpenCV contour detection and Sobel shaders to draw the terminal green (#00FF41) outlines.

---

### 4. MiDaS V2.1 — Monocular Depth Estimation

| Property | Value |
|----------|-------|
| **File** | `midas_v21_small_256.tflite` |
| **Size** | ~64 MB |
| **Source** | Intel ISL (GitHub) |
| **Input** | 256×256 RGB |
| **Output** | Depth map (inverse depth, relative distances) |
| **License** | MIT |

Estimates relative depth from a single image. Used for path analysis, distance estimation, and obstacle detection. The output is an inverse depth map where brighter = closer. The WeatherPipeline.java also includes a heuristic fallback using sky brightness/saturation analysis.

---

### 5. iNaturalist / CropNet — Plant & Species Identification

| Property | Value |
|----------|-------|
| **File** | `inaturalist_mnv2_int8.tflite` |
| **Size** | ~9 MB |
| **Source** | Kaggle (Google) |
| **Input** | 224×224 RGB |
| **Output** | Species classification (thousands of classes) |
| **License** | Apache 2.0 |

Identifies plants, animals, and insects. On-tap only (not running continuously to save battery). Covers thousands of species via the CropNet/iNaturalist model.

---

### 6. Landmark Classifier — Landmark Recognition

| Property | Value |
|----------|-------|
| **File** | `landmark_effnet_int8.tflite` |
| **Size** | ~4 MB |
| **Source** | Kaggle (Google) |
| **Input** | 224×224 RGB |
| **Output** | Landmark classification |
| **License** | Apache 2.0 |

Identifies known landmarks and monuments. On-tap only. Shows landmark name and brief info in the green overlay.

---

### 7. Weather Classifier — Sky/Weather Classification

| Property | Value |
|----------|-------|
| **File** | `weather_classifier_int8.tflite` |
| **Size** | ~4 MB |
| **Source** | Custom trained (TFLite Model Maker) |
| **Input** | 224×224 RGB |
| **Output** | Weather condition classification |
| **Classes** | CLEAR, PARTLY_CLOUDY, CLOUDY, OVERCAST, RAINY, FOGGY, SUNSET_SUNRISE, STORMY |
| **License** | Custom |

Classifies sky/weather conditions from the camera feed. Analyzes the upper 30-40% of the frame. Refreshes every 2-3 seconds. The WeatherPipeline.java also includes a heuristic fallback using sky brightness/saturation analysis.

---

## Model Input/Output Reference for Java Code

### BlazeFace
- **Input**: `float32[1][128][128][3]` — RGB image normalized to [0, 1]
- **Output 0**: `float32[1][896][16]` — Raw box encodings
- **Output 1**: `float32[1][896][1]` — Score predictions
- Must be post-processed with MediaPipe's anchor decoding logic

### COCO SSD
- **Input**: `uint8[1][300][300][3]` — RGB image
- **Output 0**: `float32[1][10][4]` — Bounding boxes [ymin, xmin, ymax, xmax]
- **Output 1**: `float32[1][10]` — Class IDs
- **Output 2**: `float32[1][10]` — Confidence scores
- **Output 3**: `int32[1]` — Number of detections

### DeepLab V3
- **Input**: `uint8[1][257][257][3]` — RGB image
- **Output**: `uint8[1][257][257][1]` — Segmentation mask (class index per pixel)

### MiDaS V2.1
- **Input**: `uint8[1][256][256][3]` — RGB image
- **Output**: `uint8[1][256][256][1]` — Depth map (inverse depth)
- Values are relative: higher = closer to camera

### iNaturalist
- **Input**: `float32[1][224][224][3]` — RGB image
- **Output**: `float32[1][N]` — Species probabilities

### Landmark
- **Input**: `float32[1][224][224][3]` — RGB image
- **Output**: `float32[1][N]` — Landmark probabilities

### Weather
- **Input**: `float32[1][224][224][3]` — RGB image
- **Output**: `float32[1][8]` — 8 weather class probabilities
