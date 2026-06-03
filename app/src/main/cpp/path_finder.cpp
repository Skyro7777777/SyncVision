// =============================================================================
// File: path_finder.cpp
// Description: Implementation of the PathFinder class. Provides A* pathfinding
//              on a discretized ground-plane grid extracted from depth maps,
//              with obstacle avoidance and path simplification.
// Author: Sync Vision Team
// =============================================================================

#include "path_finder.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <limits>
#include <android/log.h>

#define LOG_TAG "SyncVision_Path"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace syncvision {

// ============================================================================
// Constructor
// ============================================================================
PathFinder::PathFinder(int gridResolution)
    : gridResolution_(gridResolution)
{
    if (gridResolution_ <= 0) {
        LOGW("PathFinder: invalid grid resolution %d, using default %d",
             gridResolution, DEFAULT_GRID_RESOLUTION);
        gridResolution_ = DEFAULT_GRID_RESOLUTION;
    }
}

// ============================================================================
// findPath — main entry point
// ============================================================================
PathResult PathFinder::findPath(
    const float* depthMap,
    int width,
    int height,
    const float* obstacles,
    int numObstacles)
{
    PathResult result;

    if (!depthMap || width <= 0 || height <= 0) {
        LOGE("findPath: invalid depth map (data=%p, w=%d, h=%d)",
             depthMap, width, height);
        return result;
    }

    // Step 1: Extract ground plane
    auto groundPlane = extractGroundPlane(depthMap, width, height);
    int gridW = static_cast<int>(groundPlane.empty() ? 0 : groundPlane[0].size());
    int gridH = static_cast<int>(groundPlane.size());

    if (gridW == 0 || gridH == 0) {
        LOGE("findPath: empty ground plane grid");
        return result;
    }

    // Step 2: Stamp obstacles onto the ground plane
    if (obstacles && numObstacles > 0) {
        stampObstacles(groundPlane, obstacles, numObstacles, gridW, gridH);
    }

    // Step 3: Define start and goal
    // Start: bottom-center of the frame (user's position)
    int startGX = gridW / 2;
    int startGY = gridH - 1;

    // Goal: top-center of the frame (forward direction)
    int goalGX = gridW / 2;
    int goalGY = 0;

    // If start or goal is not traversable, try nearby cells
    if (!isTraversable(startGX, startGY, groundPlane)) {
        bool found = false;
        for (int dy = 0; dy <= 3 && !found; ++dy) {
            for (int dx = -3; dx <= 3 && !found; ++dx) {
                if (isTraversable(startGX + dx, startGY - dy, groundPlane)) {
                    startGX += dx;
                    startGY -= dy;
                    found = true;
                }
            }
        }
        if (!found) {
            LOGW("findPath: no traversable start position found");
            return result;
        }
    }

    if (!isTraversable(goalGX, goalGY, groundPlane)) {
        bool found = false;
        for (int dy = 0; dy <= 3 && !found; ++dy) {
            for (int dx = -3; dx <= 3 && !found; ++dx) {
                if (isTraversable(goalGX + dx, goalGY + dy, groundPlane)) {
                    goalGX += dx;
                    goalGY += dy;
                    found = true;
                }
            }
        }
        if (!found) {
            LOGW("findPath: no traversable goal position found");
            return result;
        }
    }

    // Step 4: A* pathfinding
    // Node key for hash map: gridY * gridW + gridX
    auto nodeKey = [gridW](int x, int y) -> int {
        return y * gridW + x;
    };

    std::unordered_map<int, AStarNode> nodes;
    nodes.reserve(gridW * gridH / 4);

    // Priority queue: (fCost, key)
    using PQEntry = std::pair<float, int>;
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> openSet;

    // Initialize start node
    AStarNode startNode;
    startNode.gridX = startGX;
    startNode.gridY = startGY;
    startNode.gCost = 0.0f;
    startNode.hCost = heuristic(startGX, startGY, goalGX, goalGY);
    startNode.parentX = -1;
    startNode.parentY = -1;
    startNode.closed = false;

    int startKey = nodeKey(startGX, startGY);
    nodes[startKey] = startNode;
    openSet.push({startNode.fCost(), startKey});

    // 8-directional movement
    static const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static const float moveCost[8] = {
        1.414f, 1.0f, 1.414f,  // top-left, top, top-right
        1.0f,   1.0f,           // left, right
        1.414f, 1.0f, 1.414f   // bottom-left, bottom, bottom-right
    };

    int exploredCount = 0;
    bool pathFound = false;

    while (!openSet.empty() && exploredCount < MAX_EXPLORED_NODES) {
        auto [currentFCost, currentKey] = openSet.top();
        openSet.pop();

        auto it = nodes.find(currentKey);
        if (it == nodes.end()) continue;

        AStarNode& current = it->second;
        if (current.closed) continue;
        current.closed = true;
        ++exploredCount;

        // Check if we reached the goal
        if (current.gridX == goalGX && current.gridY == goalGY) {
            pathFound = true;
            break;
        }

        // Explore neighbors
        for (int d = 0; d < 8; ++d) {
            int nx = current.gridX + dx[d];
            int ny = current.gridY + dy[d];

            if (!isTraversable(nx, ny, groundPlane)) continue;

            float newGCost = current.gCost + moveCost[d];
            int nKey = nodeKey(nx, ny);

            auto nit = nodes.find(nKey);
            if (nit != nodes.end()) {
                if (nit->second.closed) continue;
                if (newGCost < nit->second.gCost) {
                    nit->second.gCost = newGCost;
                    nit->second.parentX = current.gridX;
                    nit->second.parentY = current.gridY;
                    openSet.push({nit->second.fCost(), nKey});
                }
            } else {
                AStarNode newNode;
                newNode.gridX = nx;
                newNode.gridY = ny;
                newNode.gCost = newGCost;
                newNode.hCost = heuristic(nx, ny, goalGX, goalGY);
                newNode.parentX = current.gridX;
                newNode.parentY = current.gridY;
                newNode.closed = false;
                nodes[nKey] = newNode;
                openSet.push({newNode.fCost(), nKey});
            }
        }
    }

    LOGI("findPath: A* explored %d nodes, pathFound=%d", exploredCount, pathFound);

    if (!pathFound) {
        // Return partial result — isClear = false
        result.isClear = false;
        result.totalCost = std::numeric_limits<float>::max();
        return result;
    }

    // Step 5: Reconstruct the path by backtracking from goal
    std::vector<Waypoint> rawPath;
    int cx = goalGX, cy = goalGY;
    int key = nodeKey(cx, cy);

    while (nodes.find(key) != nodes.end()) {
        const AStarNode& node = nodes[key];
        rawPath.emplace_back(
            toPixelX(node.gridX, gridW),
            toPixelY(node.gridY, gridH),
            node.gCost
        );

        if (node.parentX < 0 || node.parentY < 0) break;
        cx = node.parentX;
        cy = node.parentY;
        key = nodeKey(cx, cy);
    }

    // Reverse to get start → goal order
    std::reverse(rawPath.begin(), rawPath.end());

    // Step 6: Simplify the path
    auto simplified = simplifyPath(rawPath, groundPlane);

    result.waypoints = std::move(simplified);
    result.totalCost = nodes[nodeKey(goalGX, goalGY)].gCost;
    result.isClear = true;

    LOGI("findPath: result has %zu waypoints, cost=%.2f",
         result.waypoints.size(), result.totalCost);
    return result;
}

// ============================================================================
// extractGroundPlane
// ============================================================================
std::vector<std::vector<bool>> PathFinder::extractGroundPlane(
    const float* depthMap, int width, int height)
{
    if (!depthMap || width <= 0 || height <= 0) {
        LOGE("extractGroundPlane: invalid parameters");
        return {};
    }

    int gridW = (width + gridResolution_ - 1) / gridResolution_;
    int gridH = (height + gridResolution_ - 1) / gridResolution_;

    std::vector<std::vector<bool>> groundPlane(
        gridH, std::vector<bool>(gridW, false));

    // For each grid cell, sample the depth values within it and classify
    for (int gy = 0; gy < gridH; ++gy) {
        for (int gx = 0; gx < gridW; ++gx) {
            // Pixel region covered by this grid cell
            int pxStart = gx * gridResolution_;
            int pyStart = gy * gridResolution_;
            int pxEnd   = std::min(pxStart + gridResolution_, width);
            int pyEnd   = std::min(pyStart + gridResolution_, height);

            float depthSum = 0.0f;
            float depthSqSum = 0.0f;
            int count = 0;

            for (int py = pyStart; py < pyEnd; ++py) {
                for (int px = pxStart; px < pxEnd; ++px) {
                    float d = depthMap[py * width + px];
                    depthSum += d;
                    depthSqSum += d * d;
                    ++count;
                }
            }

            if (count == 0) {
                groundPlane[gy][gx] = false;
                continue;
            }

            float mean = depthSum / count;
            float variance = (depthSqSum / count) - (mean * mean);

            // Ground criteria:
            //   1. Mean depth is in the ground range (close to camera)
            //   2. Variance is low (flat, consistent surface)
            bool isGround = (mean >= GROUND_DEPTH_MIN &&
                             mean <= GROUND_DEPTH_MAX &&
                             variance < 0.01f);

            groundPlane[gy][gx] = isGround;
        }
    }

    // Count ground cells for logging
    int groundCells = 0;
    for (const auto& row : groundPlane) {
        for (bool cell : row) {
            if (cell) ++groundCells;
        }
    }
    LOGI("extractGroundPlane: %d/%d cells are ground (%.1f%%)",
         groundCells, gridW * gridH,
         100.0 * groundCells / (gridW * gridH));

    return groundPlane;
}

// ============================================================================
// scorePath
// ============================================================================
float PathFinder::scorePath(
    const std::vector<Waypoint>& path,
    const float* depthMap,
    int width,
    int height)
{
    if (path.empty() || !depthMap || width <= 0 || height <= 0) {
        return std::numeric_limits<float>::max();
    }

    float totalScore = 0.0f;

    for (const auto& wp : path) {
        int px = static_cast<int>(std::round(wp.x));
        int py = static_cast<int>(std::round(wp.y));

        // Clamp to valid range
        px = std::max(0, std::min(width - 1, px));
        py = std::max(0, std::min(height - 1, py));

        float depth = depthMap[py * width + px];

        // Score: lower depth = closer = easier to navigate
        // Higher depth = farther / obstacle = worse
        totalScore += depth;
    }

    return totalScore;
}

// ============================================================================
// Private: toGridX / toGridY
// ============================================================================
int PathFinder::toGridX(int pixelX) const {
    return pixelX / gridResolution_;
}

int PathFinder::toGridY(int pixelY) const {
    return pixelY / gridResolution_;
}

// ============================================================================
// Private: toPixelX / toPixelY
// ============================================================================
float PathFinder::toPixelX(int gridX, int /*gridWidth*/) const {
    return static_cast<float>(gridX * gridResolution_ + gridResolution_ / 2);
}

float PathFinder::toPixelY(int gridY, int /*gridHeight*/) const {
    return static_cast<float>(gridY * gridResolution_ + gridResolution_ / 2);
}

// ============================================================================
// Private: heuristic
// ============================================================================
float PathFinder::heuristic(int x1, int y1, int x2, int y2)
{
    float dx = static_cast<float>(x2 - x1);
    float dy = static_cast<float>(y2 - y1);
    return std::sqrt(dx * dx + dy * dy);
}

// ============================================================================
// Private: isTraversable
// ============================================================================
bool PathFinder::isTraversable(
    int gridX, int gridY,
    const std::vector<std::vector<bool>>& groundPlane) const
{
    if (gridY < 0 || gridY >= static_cast<int>(groundPlane.size())) {
        return false;
    }
    if (gridX < 0 || gridX >= static_cast<int>(groundPlane[gridY].size())) {
        return false;
    }
    return groundPlane[gridY][gridX];
}

// ============================================================================
// Private: stampObstacles
// ============================================================================
void PathFinder::stampObstacles(
    std::vector<std::vector<bool>>& groundPlane,
    const float* obstacles,
    int numObstacles,
    int gridWidth,
    int gridHeight) const
{
    if (!obstacles || numObstacles <= 0) return;

    for (int i = 0; i < numObstacles; ++i) {
        int base = i * 4;
        float ox = obstacles[base + 0];
        float oy = obstacles[base + 1];
        float radius = obstacles[base + 2];
        // float intensity = obstacles[base + 3]; // reserved for future use

        // Convert obstacle center to grid coordinates
        int gcx = static_cast<int>(ox / gridResolution_);
        int gcy = static_cast<int>(oy / gridResolution_);
        int gradius = static_cast<int>(
            std::ceil(radius / gridResolution_)) + 1;

        // Mark all grid cells within the obstacle radius as non-traversable
        for (int dy = -gradius; dy <= gradius; ++dy) {
            for (int dx = -gradius; dx <= gradius; ++dx) {
                int gx = gcx + dx;
                int gy = gcy + dy;

                if (gx < 0 || gx >= gridWidth ||
                    gy < 0 || gy >= gridHeight) {
                    continue;
                }

                // Check if this grid cell center is within the obstacle radius
                float cellCenterX = gx * gridResolution_ + gridResolution_ * 0.5f;
                float cellCenterY = gy * gridResolution_ + gridResolution_ * 0.5f;
                float distSq = (cellCenterX - ox) * (cellCenterX - ox) +
                               (cellCenterY - oy) * (cellCenterY - oy);

                if (distSq <= radius * radius) {
                    groundPlane[gy][gx] = false;
                }
            }
        }
    }
}

// ============================================================================
// Private: simplifyPath
// ============================================================================
std::vector<Waypoint> PathFinder::simplifyPath(
    const std::vector<Waypoint>& rawPath,
    const std::vector<std::vector<bool>>& groundPlane) const
{
    if (rawPath.size() <= 2) {
        return rawPath;
    }

    std::vector<Waypoint> simplified;
    simplified.push_back(rawPath.front());

    size_t current = 0;
    while (current < rawPath.size() - 1) {
        // Try to skip as many intermediate waypoints as possible
        size_t farthest = current + 1;

        for (size_t test = rawPath.size() - 1; test > current + 1; --test) {
            int x0 = toGridX(static_cast<int>(rawPath[current].x));
            int y0 = toGridY(static_cast<int>(rawPath[current].y));
            int x1 = toGridX(static_cast<int>(rawPath[test].x));
            int y1 = toGridY(static_cast<int>(rawPath[test].y));

            if (hasLineOfSight(x0, y0, x1, y1, groundPlane)) {
                farthest = test;
                break;
            }
        }

        simplified.push_back(rawPath[farthest]);
        current = farthest;
    }

    LOGI("simplifyPath: %zu → %zu waypoints", rawPath.size(), simplified.size());
    return simplified;
}

// ============================================================================
// Private: hasLineOfSight (Bresenham's line algorithm)
// ============================================================================
bool PathFinder::hasLineOfSight(
    int x0, int y0, int x1, int y1,
    const std::vector<std::vector<bool>>& groundPlane) const
{
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    int cx = x0, cy = y0;

    while (true) {
        if (!isTraversable(cx, cy, groundPlane)) {
            return false;
        }

        if (cx == x1 && cy == y1) break;

        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            cx += sx;
        }
        if (e2 < dx) {
            err += dx;
            cy += sy;
        }

        // Safety: prevent infinite loops
        if (std::abs(cx - x0) > dx + 2 || std::abs(cy - y0) > dy + 2) {
            break;
        }
    }

    return true;
}

} // namespace syncvision
