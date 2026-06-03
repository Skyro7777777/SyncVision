// =============================================================================
// File: label_placer.h
// Description: Header for the LabelPlacer class. Implements smart label
//              placement with collision avoidance, depth-based sizing, and
//              multi-directional placement attempts (right, left, above, below).
// Author: Sync Vision Team
// =============================================================================

#ifndef LABEL_PLACER_H
#define LABEL_PLACER_H

#include <vector>
#include <string>

namespace syncvision {

// ---------------------------------------------------------------------------
// Represents a detected object from the ML pipeline, passed from Java/Kotlin.
// ---------------------------------------------------------------------------
struct DetectedObj {
    int id;                 // Unique object identifier
    std::string name;       // Object class name (e.g., "person", "car")
    int bboxX;              // Bounding box top-left X
    int bboxY;              // Bounding box top-left Y
    int bboxW;              // Bounding box width
    int bboxH;              // Bounding box height
    float confidence;       // Detection confidence [0, 1]

    DetectedObj()
        : id(0), name(""), bboxX(0), bboxY(0),
          bboxW(0), bboxH(0), confidence(0.0f) {}

    DetectedObj(int id_, const std::string& name_,
                int x, int y, int w, int h, float conf)
        : id(id_), name(name_), bboxX(x), bboxY(y),
          bboxW(w), bboxH(h), confidence(conf) {}
};

// ---------------------------------------------------------------------------
// Represents a placed label with position, font size, and text.
// ---------------------------------------------------------------------------
struct LabelPlacement {
    int x;                  // Label anchor X (left edge of text)
    int y;                  // Label anchor Y (baseline of text)
    int fontSize;           // Font size in pixels [12, 24]
    std::string text;       // Display text (name + confidence %)

    LabelPlacement() : x(0), y(0), fontSize(12), text("") {}

    LabelPlacement(int x_, int y_, int size, const std::string& txt)
        : x(x_), y(y_), fontSize(size), text(txt) {}
};

// ---------------------------------------------------------------------------
// LabelPlacer
//
// Places labels around detected objects with the following strategy:
//   1. Try placing the label to the RIGHT of the bounding box
//   2. If that fails (collision or out-of-bounds), try LEFT
//   3. If that fails, try ABOVE
//   4. If that fails, try BELOW
//   5. Collision avoidance: check for overlap with already-placed labels
//   6. Depth-based sizing: objects higher in the frame (farther away) get
//      smaller font sizes, mimicking perspective. Range: 12px–24px.
// ---------------------------------------------------------------------------
class LabelPlacer {
public:
    // Minimum and maximum font sizes
    static constexpr int MIN_FONT_SIZE = 12;
    static constexpr int MAX_FONT_SIZE = 24;

    // Padding between label and bounding box edge (in pixels)
    static constexpr int LABEL_PADDING = 8;

    // Estimated character width as a fraction of font size
    static constexpr float CHAR_WIDTH_RATIO = 0.6f;

    // Line height as a multiplier of font size
    static constexpr float LINE_HEIGHT_RATIO = 1.2f;

    LabelPlacer() = default;
    ~LabelPlacer() = default;

    // -----------------------------------------------------------------------
    // placeLabels
    //   Given a list of detected objects and canvas dimensions, computes
    //   optimal label positions for each object.
    //
    //   objects       — list of detected objects
    //   canvasWidth   — canvas / viewport width in pixels
    //   canvasHeight  — canvas / viewport height in pixels
    //
    //   Returns a vector of LabelPlacement, one per object, in the same
    //   order as the input.
    // -----------------------------------------------------------------------
    std::vector<LabelPlacement> placeLabels(
        const std::vector<DetectedObj>& objects,
        int canvasWidth,
        int canvasHeight);

private:
    // -----------------------------------------------------------------------
    // Internal: Estimate the bounding box of a label given its text and
    // font size. Returns {x, y, width, height} of the label rectangle.
    // -----------------------------------------------------------------------
    struct Rect {
        int x, y, w, h;

        Rect() : x(0), y(0), w(0), h(0) {}
        Rect(int x_, int y_, int w_, int h_)
            : x(x_), y(y_), w(w_), h(h_) {}
    };

    // -----------------------------------------------------------------------
    // Internal: Check if two rectangles overlap (with optional margin).
    // -----------------------------------------------------------------------
    static bool rectsOverlap(const Rect& a, const Rect& b, int margin = 4);

    // -----------------------------------------------------------------------
    // Internal: Check if a rectangle is fully within the canvas bounds.
    // -----------------------------------------------------------------------
    static bool isInBounds(const Rect& r, int canvasWidth, int canvasHeight);

    // -----------------------------------------------------------------------
    // Internal: Compute font size based on the vertical position of the
    // object. Objects higher in the frame (smaller bboxY) are considered
    // farther away and get smaller text.
    // -----------------------------------------------------------------------
    static int computeFontSize(const DetectedObj& obj,
                                int canvasHeight);

    // -----------------------------------------------------------------------
    // Internal: Build the display text for a label.
    // -----------------------------------------------------------------------
    static std::string buildLabelText(const DetectedObj& obj);

    // -----------------------------------------------------------------------
    // Internal: Estimate the pixel bounding box of a text label.
    // -----------------------------------------------------------------------
    static Rect estimateLabelRect(const std::string& text,
                                   int fontSize,
                                   int anchorX,
                                   int anchorY);

    // -----------------------------------------------------------------------
    // Internal: Try placing a label in a specific direction relative to
    // the object's bounding box. Returns the placement if successful,
    // or an invalid placement (x < 0) if the position is out of bounds
    // or collides with existing labels.
    // -----------------------------------------------------------------------
    LabelPlacement tryPlacement(
        const DetectedObj& obj,
        int fontSize,
        const std::string& text,
        const Rect& objRect,
        const std::vector<Rect>& placedRects,
        int canvasWidth,
        int canvasHeight,
        const std::string& direction);
};

} // namespace syncvision

#endif // LABEL_PLACER_H
