// =============================================================================
// File: contour_processor.cpp
// Description: Implementation of the ContourProcessor class. Provides contour
//              extraction from segmentation masks, Douglas-Peucker
//              simplification, outline extraction, and contour thinning.
//              Uses OpenCV when available, otherwise falls back to custom
//              implementations.
// Author: Sync Vision Team
// =============================================================================

#include "contour_processor.h"

#ifdef SYNCVISION_USE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <android/log.h>

#define LOG_TAG "SyncVision_Contour"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace syncvision {

// ============================================================================
// processMask
// ============================================================================
std::vector<std::vector<Point>> ContourProcessor::processMask(
    const int* maskData, int width, int height)
{
    if (!maskData || width <= 0 || height <= 0) {
        LOGE("processMask: invalid parameters (data=%p, w=%d, h=%d)",
             maskData, width, height);
        return {};
    }

#ifdef SYNCVISION_USE_OPENCV
    // -----------------------------------------------------------------------
    // OpenCV path: convert mask to cv::Mat and use cv::findContours
    // -----------------------------------------------------------------------
    try {
        cv::Mat mask(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                mask.at<uint8_t>(y, x) =
                    (maskData[y * width + x] != 0) ? 255 : 0;
            }
        }

        std::vector<std::vector<cv::Point>> cvContours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask, cvContours, hierarchy,
                         cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<std::vector<Point>> result;
        result.reserve(cvContours.size());
        for (const auto& cvContour : cvContours) {
            std::vector<Point> contour;
            contour.reserve(cvContour.size());
            for (const auto& pt : cvContour) {
                contour.emplace_back(pt.x, pt.y);
            }
            result.push_back(std::move(contour));
        }
        LOGI("processMask (OpenCV): found %zu contours", result.size());
        return result;
    } catch (const cv::Exception& e) {
        LOGE("OpenCV exception in processMask: %s", e.what());
        // Fall through to basic implementation
    } catch (const std::exception& e) {
        LOGE("Exception in processMask: %s", e.what());
    }
#endif

    // -----------------------------------------------------------------------
    // Basic fallback: Moore neighborhood border tracing
    // -----------------------------------------------------------------------
    return traceContours(maskData, width, height);
}

// ============================================================================
// simplifyContour
// ============================================================================
std::vector<Point> ContourProcessor::simplifyContour(
    const std::vector<Point>& contour, double epsilon)
{
    if (contour.size() <= 2) {
        return contour;
    }
    if (epsilon < 0.0) {
        LOGW("simplifyContour: negative epsilon (%.4f) clamped to 0", epsilon);
        epsilon = 0.0;
    }

#ifdef SYNCVISION_USE_OPENCV
    try {
        std::vector<cv::Point> cvContour;
        cvContour.reserve(contour.size());
        for (const auto& pt : contour) {
            cvContour.emplace_back(pt.x, pt.y);
        }

        std::vector<cv::Point> cvSimplified;
        cv::approxPolyDP(cvContour, cvSimplified, epsilon, true /* closed */);

        std::vector<Point> result;
        result.reserve(cvSimplified.size());
        for (const auto& pt : cvSimplified) {
            result.emplace_back(pt.x, pt.y);
        }
        return result;
    } catch (const cv::Exception& e) {
        LOGE("OpenCV exception in simplifyContour: %s", e.what());
    } catch (const std::exception& e) {
        LOGE("Exception in simplifyContour: %s", e.what());
    }
#endif

    // -----------------------------------------------------------------------
    // Basic fallback: custom Douglas-Peucker implementation
    // -----------------------------------------------------------------------
    auto result = douglasPeucker(contour, 0,
                                  static_cast<int>(contour.size()) - 1,
                                  epsilon);

    // For closed contours, ensure the last point connects back to the first
    if (!result.empty() && result.front() != result.back()) {
        result.push_back(result.front());
    }
    return result;
}

// ============================================================================
// extractOutlines
// ============================================================================
std::vector<std::vector<Point>> ContourProcessor::extractOutlines(
    const std::vector<std::vector<Point>>& contours)
{
    if (contours.empty()) {
        return {};
    }

    // Sort contours by area (descending) so larger contours come first
    std::vector<size_t> indices(contours.size());
    for (size_t i = 0; i < contours.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(),
              [&contours](size_t a, size_t b) {
                  return std::abs(contourArea(contours[a])) >
                         std::abs(contourArea(contours[b]));
              });

    // A contour is an "outline" if it is not inside any other contour
    std::vector<std::vector<Point>> outlines;
    for (size_t i = 0; i < indices.size(); ++i) {
        const auto& candidate = contours[indices[i]];
        bool isHole = false;

        for (size_t j = 0; j < indices.size(); ++j) {
            if (i == j) continue;
            const auto& other = contours[indices[j]];
            // Only check containment by larger contours
            if (std::abs(contourArea(other)) > std::abs(contourArea(candidate))) {
                if (isInside(candidate, other)) {
                    isHole = true;
                    break;
                }
            }
        }

        if (!isHole) {
            outlines.push_back(candidate);
        }
    }

    LOGI("extractOutlines: %zu contours â†’ %zu outlines",
         contours.size(), outlines.size());
    return outlines;
}

// ============================================================================
// thinContour
// ============================================================================
std::vector<Point> ContourProcessor::thinContour(
    const std::vector<Point>& contour, int targetThickness)
{
    if (contour.empty()) {
        return {};
    }
    if (targetThickness < 1) {
        LOGW("thinContour: targetThickness %d clamped to 1", targetThickness);
        targetThickness = 1;
    }

    // For targetThickness == 1, sample every other point and simplify
    if (targetThickness == 1) {
        // Apply simplification with a small epsilon to thin the contour
        double epsilon = 1.0;
        return simplifyContour(contour, epsilon);
    }

    // For thicker contours, reduce point density proportionally
    int step = std::max(1, targetThickness);
    std::vector<Point> result;
    result.reserve(contour.size() / step + 1);
    for (size_t i = 0; i < contour.size(); i += step) {
        result.push_back(contour[i]);
    }

    // Ensure the contour is still closed
    if (!result.empty() && result.front() != result.back()) {
        result.push_back(result.front());
    }
    return result;
}

// ============================================================================
// Private: traceContours (Moore neighborhood border tracing)
// ============================================================================
std::vector<std::vector<Point>> ContourProcessor::traceContours(
    const int* maskData, int width, int height)
{
    // Build a binary grid: 1 = foreground, 0 = background
    std::vector<uint8_t> grid(width * height, 0);
    for (int i = 0; i < width * height; ++i) {
        grid[i] = (maskData[i] != 0) ? 1 : 0;
    }

    // Track which border pixels have been visited
    std::vector<uint8_t> visited(width * height, 0);

    // 8-connected neighborhood (Moore neighborhood) â€” clockwise from East
    static const int dx[8] = { 1,  1,  0, -1, -1, -1,  0,  1};
    static const int dy[8] = { 0,  1,  1,  1,  0, -1, -1, -1};

    std::vector<std::vector<Point>> contours;

    auto inBounds = [&](int x, int y) -> bool {
        return x >= 0 && x < width && y >= 0 && y < height;
    };

    for (int startY = 0; startY < height; ++startY) {
        for (int startX = 0; startX < width; ++startX) {
            int idx = startY * width + startX;

            // Look for an unvisited foreground pixel that has a background
            // neighbor (i.e., it's on the border)
            if (!grid[idx] || visited[idx]) continue;

            bool isBorder = false;
            for (int d = 0; d < 8; ++d) {
                int nx = startX + dx[d];
                int ny = startY + dy[d];
                if (!inBounds(nx, ny) || !grid[ny * width + nx]) {
                    isBorder = true;
                    break;
                }
            }
            if (!isBorder) continue;

            // Trace this border using Moore neighborhood tracing
            std::vector<Point> contour;
            int cx = startX, cy = startY;
            int dir = 7; // Start searching from the previous direction

            do {
                contour.emplace_back(cx, cy);
                visited[cy * width + cx] = 1;

                // Search the 8-neighbors clockwise starting from (dir + 1) % 8
                bool found = false;
                for (int step = 0; step < 8; ++step) {
                    int searchDir = (dir + 7 + step) % 8;
                    int nx = cx + dx[searchDir];
                    int ny = cy + dy[searchDir];

                    if (inBounds(nx, ny) && grid[ny * width + nx]) {
                        cx = nx;
                        cy = ny;
                        dir = searchDir;
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    // Isolated pixel
                    break;
                }

                // Safety: prevent infinite loops
                if (contour.size() > static_cast<size_t>(width * height)) {
                    LOGW("traceContours: contour exceeded max size, truncating");
                    break;
                }
            } while (cx != startX || cy != startY);

            if (contour.size() >= 3) {
                contours.push_back(std::move(contour));
            }
        }
    }

    LOGI("traceContours (basic): found %zu contours", contours.size());
    return contours;
}

// ============================================================================
// Private: douglasPeucker (iterative implementation for stack safety)
// ============================================================================
std::vector<Point> ContourProcessor::douglasPeucker(
    const std::vector<Point>& points, int start, int end, double epsilon)
{
    if (end <= start + 1) {
        return {points[start], points[end]};
    }

    // Find the point with maximum distance from the line startâ†’end
    double maxDist = 0.0;
    int splitIndex = start;

    for (int i = start + 1; i < end; ++i) {
        double dist = perpendicularDistance(points[i], points[start], points[end]);
        if (dist > maxDist) {
            maxDist = dist;
            splitIndex = i;
        }
    }

    if (maxDist <= epsilon) {
        // All points are close enough to the line; simplify to just endpoints
        return {points[start], points[end]};
    }

    // Recursively simplify both halves
    auto left  = douglasPeucker(points, start, splitIndex, epsilon);
    auto right = douglasPeucker(points, splitIndex, end, epsilon);

    // Merge, avoiding duplicate at the split point
    std::vector<Point> result;
    result.reserve(left.size() + right.size() - 1);
    result.insert(result.end(), left.begin(), left.end());
    result.insert(result.end(), right.begin() + 1, right.end());

    return result;
}

// ============================================================================
// Private: perpendicularDistance
// ============================================================================
double ContourProcessor::perpendicularDistance(
    const Point& p, const Point& lineStart, const Point& lineEnd)
{
    double dx = static_cast<double>(lineEnd.x - lineStart.x);
    double dy = static_cast<double>(lineEnd.y - lineStart.y);
    double lineLenSq = dx * dx + dy * dy;

    if (lineLenSq < 1e-10) {
        // Degenerate line: both endpoints are the same
        double px = static_cast<double>(p.x - lineStart.x);
        double py = static_cast<double>(p.y - lineStart.y);
        return std::sqrt(px * px + py * py);
    }

    // Cross product magnitude / line length
    double cross = std::abs(
        static_cast<double>(p.x - lineStart.x) * dy -
        static_cast<double>(p.y - lineStart.y) * dx
    );
    return cross / std::sqrt(lineLenSq);
}

// ============================================================================
// Private: contourArea (shoelace formula)
// ============================================================================
double ContourProcessor::contourArea(const std::vector<Point>& contour)
{
    if (contour.size() < 3) return 0.0;

    double area = 0.0;
    size_t n = contour.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        area += static_cast<double>(contour[i].x) * contour[j].y;
        area -= static_cast<double>(contour[j].x) * contour[i].y;
    }
    return area * 0.5;
}

// ============================================================================
// Private: isInside (point-in-polygon for contour A inside contour B)
// ============================================================================
bool ContourProcessor::isInside(
    const std::vector<Point>& inner, const std::vector<Point>& outer)
{
    if (inner.empty() || outer.empty()) return false;

    // Quick bounding-box check first
    int innerMinX = inner[0].x, innerMaxX = inner[0].x;
    int innerMinY = inner[0].y, innerMaxY = inner[0].y;
    int outerMinX = outer[0].x, outerMaxX = outer[0].x;
    int outerMinY = outer[0].y, outerMaxY = outer[0].y;

    for (const auto& p : inner) {
        innerMinX = std::min(innerMinX, p.x);
        innerMaxX = std::max(innerMaxX, p.x);
        innerMinY = std::min(innerMinY, p.y);
        innerMaxY = std::max(innerMaxY, p.y);
    }
    for (const auto& p : outer) {
        outerMinX = std::min(outerMinX, p.x);
        outerMaxX = std::max(outerMaxX, p.x);
        outerMinY = std::min(outerMinY, p.y);
        outerMaxY = std::max(outerMaxY, p.y);
    }

    // If inner bounding box is not fully within outer bounding box, it's not inside
    if (innerMinX < outerMinX || innerMaxX > outerMaxX ||
        innerMinY < outerMinY || innerMaxY > outerMaxY) {
        return false;
    }

    // Ray-casting test: check if the centroid of inner is inside outer
    // Use the average point of inner as a test point
    double testX = 0.0, testY = 0.0;
    for (const auto& p : inner) {
        testX += p.x;
        testY += p.y;
    }
    testX /= inner.size();
    testY /= inner.size();

    // Ray-casting algorithm
    bool inside = false;
    size_t n = outer.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        double xi = outer[i].x, yi = outer[i].y;
        double xj = outer[j].x, yj = outer[j].y;

        if (((yi > testY) != (yj > testY)) &&
            (testX < (xj - xi) * (testY - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

} // namespace syncvision
