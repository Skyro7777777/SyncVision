// =============================================================================
// File: contour_processor.h
// Description: Header for the ContourProcessor class. Handles segmentation mask
//              processing, contour extraction, simplification, and thinning.
//              When OpenCV is available, uses cv::findContours and
//              cv::approxPolyDP; otherwise falls back to basic implementations.
// Author: Sync Vision Team
// =============================================================================

#ifndef CONTOUR_PROCESSOR_H
#define CONTOUR_PROCESSOR_H

#include <vector>
#include <cstdint>

namespace syncvision {

// ---------------------------------------------------------------------------
// Simple 2D point structure used throughout the contour processing pipeline.
// ---------------------------------------------------------------------------
struct Point {
    int x;
    int y;

    Point() : x(0), y(0) {}
    Point(int x_, int y_) : x(x_), y(y_) {}

    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
    bool operator!=(const Point& other) const {
        return !(*this == other);
    }
};

// ---------------------------------------------------------------------------
// ContourProcessor
//
// Extracts and processes contours from segmentation masks. Provides methods
// for mask-to-contour conversion, contour simplification (Douglas-Peucker),
// outline extraction, and contour thinning.
// ---------------------------------------------------------------------------
class ContourProcessor {
public:
    ContourProcessor() = default;
    ~ContourProcessor() = default;

    // Non-copyable, movable
    ContourProcessor(const ContourProcessor&) = delete;
    ContourProcessor& operator=(const ContourProcessor&) = delete;
    ContourProcessor(ContourProcessor&&) = default;
    ContourProcessor& operator=(ContourProcessor&&) = default;

    // -----------------------------------------------------------------------
    // processMask
    //   Converts a segmentation mask (flat array, row-major) into a list of
    //   contours. Non-zero pixels are treated as foreground.
    //
    //   maskData  — pointer to the mask pixel data (1 row = width ints)
    //   width     — image width in pixels
    //   height    — image height in pixels
    //
    //   Returns a vector of contours, where each contour is a vector of Points.
    // -----------------------------------------------------------------------
    std::vector<std::vector<Point>> processMask(const int* maskData,
                                                 int width,
                                                 int height);

    // -----------------------------------------------------------------------
    // simplifyContour
    //   Applies Douglas-Peucker simplification to reduce the number of points
    //   in a contour while preserving its shape.
    //
    //   contour  — the input contour
    //   epsilon  — the approximation accuracy (maximum distance from the
    //              original contour to the simplified one)
    //
    //   Returns the simplified contour.
    // -----------------------------------------------------------------------
    std::vector<Point> simplifyContour(const std::vector<Point>& contour,
                                        double epsilon);

    // -----------------------------------------------------------------------
    // extractOutlines
    //   Filters contours to return only the outermost (non-hole) outlines.
    //   This uses a simple area-based heuristic: contours that are not
    //   contained within another contour.
    //
    //   contours — all contours from processMask
    //
    //   Returns only the outer outline contours.
    // -----------------------------------------------------------------------
    std::vector<std::vector<Point>> extractOutlines(
        const std::vector<std::vector<Point>>& contours);

    // -----------------------------------------------------------------------
    // thinContour
    //   Reduces a contour's effective thickness by eroding the contour toward
    //   its medial axis. Useful for producing thin, single-pixel-wide
    //   outlines for rendering.
    //
    //   contour          — input contour
    //   targetThickness  — desired thickness in pixels (1 = single pixel)
    //
    //   Returns the thinned contour.
    // -----------------------------------------------------------------------
    std::vector<Point> thinContour(const std::vector<Point>& contour,
                                    int targetThickness);

private:
    // -----------------------------------------------------------------------
    // Internal: Basic contour tracing using a border-following algorithm
    // (Moore neighborhood tracing). Used when OpenCV is not available.
    // -----------------------------------------------------------------------
    std::vector<std::vector<Point>> traceContours(const int* maskData,
                                                   int width,
                                                   int height);

    // -----------------------------------------------------------------------
    // Internal: Douglas-Peucker line simplification (recursive).
    // -----------------------------------------------------------------------
    std::vector<Point> douglasPeucker(const std::vector<Point>& points,
                                       int start,
                                       int end,
                                       double epsilon);

    // -----------------------------------------------------------------------
    // Internal: Compute perpendicular distance from a point to a line
    // defined by two endpoints.
    // -----------------------------------------------------------------------
    static double perpendicularDistance(const Point& p,
                                        const Point& lineStart,
                                        const Point& lineEnd);

    // -----------------------------------------------------------------------
    // Internal: Compute the signed area of a contour (shoelace formula).
    // Positive = counter-clockwise. Used for orientation / hierarchy checks.
    // -----------------------------------------------------------------------
    static double contourArea(const std::vector<Point>& contour);

    // -----------------------------------------------------------------------
    // Internal: Check if contour A is inside contour B (simple bounding-box
    // + point-in-polygon test).
    // -----------------------------------------------------------------------
    static bool isInside(const std::vector<Point>& inner,
                          const std::vector<Point>& outer);
};

} // namespace syncvision

#endif // CONTOUR_PROCESSOR_H
