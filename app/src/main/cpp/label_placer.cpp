// =============================================================================
// File: label_placer.cpp
// Description: Implementation of the LabelPlacer class. Provides smart label
//              placement with collision avoidance, depth-based font sizing,
//              and multi-directional placement attempts.
// Author: Sync Vision Team
// =============================================================================

#include "label_placer.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <android/log.h>

#define LOG_TAG "SyncVision_Label"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace syncvision {

// ============================================================================
// placeLabels — main entry point
// ============================================================================
std::vector<LabelPlacement> LabelPlacer::placeLabels(
    const std::vector<DetectedObj>& objects,
    int canvasWidth,
    int canvasHeight)
{
    if (canvasWidth <= 0 || canvasHeight <= 0) {
        LOGE("placeLabels: invalid canvas dimensions (%d x %d)",
             canvasWidth, canvasHeight);
        return {};
    }

    std::vector<LabelPlacement> placements;
    placements.reserve(objects.size());

    // Track the bounding rectangles of already-placed labels for collision
    std::vector<Rect> placedRects;

    for (const auto& obj : objects) {
        int fontSize = computeFontSize(obj, canvasHeight);
        std::string text = buildLabelText(obj);

        // Bounding box of the detected object
        Rect objRect(obj.bboxX, obj.bboxY, obj.bboxW, obj.bboxH);

        // Try each direction in priority order
        static const char* directions[] = {"right", "left", "above", "below"};

        LabelPlacement placement;
        bool placed = false;

        for (const char* dir : directions) {
            placement = tryPlacement(obj, fontSize, text, objRect,
                                      placedRects, canvasWidth,
                                      canvasHeight, dir);
            if (placement.x >= 0) {
                placed = true;
                break;
            }
        }

        if (!placed) {
            // Fallback: place at the top-left corner of the bounding box
            // even if it overlaps. This ensures every object gets a label.
            LOGW("placeLabels: could not find non-overlapping position for "
                 "object %d ('%s'), using fallback", obj.id, obj.name.c_str());
            placement = LabelPlacement(obj.bboxX, obj.bboxY - fontSize - 2,
                                        fontSize, text);
            // Clamp to canvas
            placement.y = std::max(fontSize, placement.y);
        }

        // Record the placed label's rectangle for future collision checks
        Rect labelRect = estimateLabelRect(text, fontSize,
                                            placement.x, placement.y);
        placedRects.push_back(labelRect);

        placements.push_back(placement);
    }

    LOGI("placeLabels: placed %zu labels on %d x %d canvas",
         placements.size(), canvasWidth, canvasHeight);
    return placements;
}

// ============================================================================
// Private: computeFontSize
// ============================================================================
int LabelPlacer::computeFontSize(const DetectedObj& obj, int canvasHeight)
{
    if (canvasHeight <= 0) return MIN_FONT_SIZE;

    // Normalize the vertical position: 0.0 = top (far), 1.0 = bottom (near)
    float normalizedY = static_cast<float>(obj.bboxY + obj.bboxH / 2)
                        / static_cast<float>(canvasHeight);
    normalizedY = std::max(0.0f, std::min(1.0f, normalizedY));

    // Map to font size: top → MIN_FONT_SIZE, bottom → MAX_FONT_SIZE
    float fontSizeF = MIN_FONT_SIZE +
        normalizedY * (MAX_FONT_SIZE - MIN_FONT_SIZE);

    // Also consider confidence: high-confidence objects get slightly larger text
    float confidenceBoost = obj.confidence * 2.0f; // 0–2 px bonus
    fontSizeF += confidenceBoost;

    int fontSize = static_cast<int>(std::round(fontSizeF));
    return std::max(MIN_FONT_SIZE, std::min(MAX_FONT_SIZE, fontSize));
}

// ============================================================================
// Private: buildLabelText
// ============================================================================
std::string LabelPlacer::buildLabelText(const DetectedObj& obj)
{
    std::ostringstream oss;
    oss << obj.name;

    // Append confidence percentage if reasonable
    if (obj.confidence > 0.0f) {
        int pct = static_cast<int>(std::round(obj.confidence * 100.0f));
        oss << " " << pct << "%";
    }
    return oss.str();
}

// ============================================================================
// Private: estimateLabelRect
// ============================================================================
LabelPlacer::Rect LabelPlacer::estimateLabelRect(
    const std::string& text, int fontSize, int anchorX, int anchorY)
{
    // Estimate character count (handle multi-byte UTF-8 by counting bytes
    // that are not continuation bytes)
    int charCount = 0;
    for (unsigned char c : text) {
        if ((c & 0xC0) != 0x80) { // Not a UTF-8 continuation byte
            ++charCount;
        }
    }

    int textWidth = static_cast<int>(
        charCount * fontSize * CHAR_WIDTH_RATIO + 2 * LABEL_PADDING);
    int textHeight = static_cast<int>(
        fontSize * LINE_HEIGHT_RATIO + 2 * LABEL_PADDING);

    return Rect(anchorX, anchorY - fontSize, textWidth, textHeight);
}

// ============================================================================
// Private: rectsOverlap
// ============================================================================
bool LabelPlacer::rectsOverlap(const Rect& a, const Rect& b, int margin)
{
    return !(a.x + a.w + margin <= b.x ||
             b.x + b.w + margin <= a.x ||
             a.y + a.h + margin <= b.y ||
             b.y + b.h + margin <= a.y);
}

// ============================================================================
// Private: isInBounds
// ============================================================================
bool LabelPlacer::isInBounds(const Rect& r, int canvasWidth, int canvasHeight)
{
    return r.x >= 0 && r.y >= 0 &&
           r.x + r.w <= canvasWidth &&
           r.y + r.h <= canvasHeight;
}

// ============================================================================
// Private: tryPlacement
// ============================================================================
LabelPlacement LabelPlacer::tryPlacement(
    const DetectedObj& obj,
    int fontSize,
    const std::string& text,
    const Rect& objRect,
    const std::vector<Rect>& placedRects,
    int canvasWidth,
    int canvasHeight,
    const std::string& direction)
{
    int anchorX = 0, anchorY = 0;

    // Compute anchor position based on direction
    if (direction == "right") {
        anchorX = objRect.x + objRect.w + LABEL_PADDING;
        anchorY = objRect.y + objRect.h / 2 + fontSize / 2;
    } else if (direction == "left") {
        // We'll estimate width first, then position from the left edge
        Rect tempRect = estimateLabelRect(text, fontSize, 0, 0);
        anchorX = objRect.x - tempRect.w - LABEL_PADDING;
        anchorY = objRect.y + objRect.h / 2 + fontSize / 2;
    } else if (direction == "above") {
        anchorX = objRect.x;
        anchorY = objRect.y - LABEL_PADDING;
    } else if (direction == "below") {
        anchorX = objRect.x;
        anchorY = objRect.y + objRect.h + fontSize + LABEL_PADDING;
    } else {
        return LabelPlacement(-1, -1, fontSize, text); // Invalid
    }

    // Estimate the label's bounding rectangle at this position
    Rect labelRect = estimateLabelRect(text, fontSize, anchorX, anchorY);

    // Check canvas bounds
    if (!isInBounds(labelRect, canvasWidth, canvasHeight)) {
        return LabelPlacement(-1, -1, fontSize, text); // Invalid
    }

    // Check for collisions with already-placed labels
    for (const auto& placed : placedRects) {
        if (rectsOverlap(labelRect, placed)) {
            return LabelPlacement(-1, -1, fontSize, text); // Invalid
        }
    }

    // Success — return the placement
    return LabelPlacement(anchorX, anchorY, fontSize, text);
}

} // namespace syncvision
