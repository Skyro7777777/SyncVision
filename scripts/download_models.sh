#!/bin/bash
# =============================================================================
# Sync Vision — Model Download Script
# =============================================================================
# Downloads all 7 pre-trained TFLite models from official sources.
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh
#
# Requirements:
#   - wget or curl
#   - Internet connection
#   - ~90MB free disk space for models
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/app/src/main/assets/models"

echo "============================================================"
echo "  SYNC VISION — Pre-Trained Model Download Script"
echo "  All 7 models from official sources"
echo "============================================================"
echo "Models directory: $MODELS_DIR"
echo ""

mkdir -p "$MODELS_DIR"

# --- 1. BlazeFace Short-Range (Face Detection) ---
echo "[1/7] BlazeFace Short-Range — Face Detection (~224 KB)"
if [ -f "$MODELS_DIR/blazeface_short.tflite" ]; then
    echo "  ✓ Already exists — skipping"
else
    wget -q --show-progress \
        "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite" \
        -O "$MODELS_DIR/blazeface_short.tflite" && \
        echo "  ✓ Downloaded" || echo "  ✗ Failed"
fi
echo ""

# --- 2. COCO SSD MobileNet V1 (Object Detection) ---
echo "[2/7] COCO SSD MobileNet V1 — Object Detection (~4 MB)"
if [ -f "$MODELS_DIR/coco_ssd_mobilenet_v2_300.tflite" ]; then
    echo "  ✓ Already exists — skipping"
else
    wget -q --show-progress \
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip" \
        -O /tmp/coco_ssd.zip && \
        unzip -o /tmp/coco_ssd.zip -d /tmp/coco_ssd/ && \
        cp /tmp/coco_ssd/detect.tflite "$MODELS_DIR/coco_ssd_mobilenet_v2_300.tflite" && \
        rm -rf /tmp/coco_ssd.zip /tmp/coco_ssd/ && \
        echo "  ✓ Downloaded" || echo "  ✗ Failed"
fi
echo ""

# --- 3. DeepLab V3 MobileNet V2 (Segmentation) ---
echo "[3/7] DeepLab V3 MobileNet V2 — Segmentation (~2.8 MB)"
if [ -f "$MODELS_DIR/deeplab_v3_mnv2_257.tflite" ]; then
    echo "  ✓ Already exists — skipping"
else
    wget -q --show-progress \
        "https://www.kaggle.com/models/tensorflow/deeplabv3/frameworks/TfLite/variations/metadata/versions/1/download" \
        -O /tmp/deeplab_temp.tar && \
        cd /tmp && tar xf deeplab_temp.tar && mv 1.tflite "$MODELS_DIR/deeplab_v3_mnv2_257.tflite" && \
        rm /tmp/deeplab_temp.tar && cd "$PROJECT_DIR" && \
        echo "  ✓ Downloaded" || echo "  ✗ Failed — try manual download from Kaggle"
fi
echo ""

# --- 4. MiDaS V2.1 (Depth Estimation) ---
echo "[4/7] MiDaS V2.1 — Depth Estimation (~64 MB)"
if [ -f "$MODELS_DIR/midas_v21_small_256.tflite" ]; then
    echo "  ✓ Already exists — skipping"
else
    wget -q --show-progress \
        "https://github.com/isl-org/MiDaS/releases/download/v2_1/model_opt.tflite" \
        -O "$MODELS_DIR/midas_v21_small_256.tflite" && \
        echo "  ✓ Downloaded" || echo "  ✗ Failed"
fi
echo ""

# --- 5. iNaturalist / CropNet (Plant & Species ID) ---
echo "[5/7] iNaturalist / CropNet — Plant & Species ID (~9 MB)"
if [ -f "$MODELS_DIR/inaturalist_mnv2_int8.tflite" ]; then
    echo "  ✓ Already exists — skipping"
else
    echo "  ⚠ This model requires a free Kaggle account."
    echo "  Download from: https://www.kaggle.com/models/google/cropnet"
    echo "  Select TFLite format → save as inaturalist_mnv2_int8.tflite"
    echo ""
    echo "  Or use Kaggle CLI:"
    echo "    pip install kaggle"
    echo "    kaggle models download google/cropnet --flatten"
    echo "    mv *.tflite $MODELS_DIR/inaturalist_mnv2_int8.tflite"
fi
echo ""

# --- 6. Landmark Classifier (Landmark Recognition) ---
echo "[6/7] Landmark Classifier — Landmark Recognition (~4 MB)"
if [ -f "$MODELS_DIR/landmark_effnet_int8.tflite" ]; then
    echo "  ✓ Already exists — skipping"
else
    echo "  ⚠ This model requires a free Kaggle account."
    echo "  Download from: https://www.kaggle.com/models/google/landmarks"
    echo "  Select TFLite format → save as landmark_effnet_int8.tflite"
    echo ""
    echo "  Or use Kaggle CLI:"
    echo "    kaggle models download google/landmarks --flatten"
    echo "    mv *.tflite $MODELS_DIR/landmark_effnet_int8.tflite"
fi
echo ""

# --- 7. Weather Classifier (Weather Classification) ---
echo "[7/7] Weather Classifier — Weather Classification (~4 MB)"
if [ -f "$MODELS_DIR/weather_classifier_int8.tflite" ]; then
    echo "  ✓ Already exists — skipping"
else
    echo "  ⚠ This model must be trained using TFLite Model Maker."
    echo "  See scripts/train_weather_model.py for the training script."
    echo "  Dataset: https://data.mendeley.com/datasets/4drtyfjtfy/1"
    echo "  Note: WeatherPipeline.java has a heuristic fallback that works without this model."
fi
echo ""

# --- Summary ---
echo "============================================================"
echo "  MODEL STATUS"
echo "============================================================"
echo ""

TOTAL=0
MISSING=0
for model in blazeface_short.tflite coco_ssd_mobilenet_v2_300.tflite deeplab_v3_mnv2_257.tflite midas_v21_small_256.tflite inaturalist_mnv2_int8.tflite landmark_effnet_int8.tflite weather_classifier_int8.tflite; do
    if [ -f "$MODELS_DIR/$model" ]; then
        SIZE=$(stat -c%s "$MODELS_DIR/$model" 2>/dev/null || echo "?")
        TOTAL=$((TOTAL + SIZE))
        echo "  ✓ $model ($SIZE bytes)"
    else
        echo "  ✗ MISSING: $model"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "  Total: $TOTAL bytes (~$((TOTAL / 1048576)) MB)"

if [ $MISSING -eq 0 ]; then
    echo "  🎉 ALL 7 MODELS PRESENT!"
else
    echo "  ⚠️  $MISSING model(s) missing"
fi

echo ""
echo "============================================================"
echo "  DONE"
echo "============================================================"
