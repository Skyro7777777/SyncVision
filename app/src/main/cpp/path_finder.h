// =============================================================================
// File: path_finder.h
// Description: Header for the PathFinder class. Implements A* pathfinding on
//              a discretized grid derived from depth maps, with ground-plane
//              extraction and obstacle detection.
// Author: Sync Vision Team
// =============================================================================

#ifndef PATH_FINDER_H
#define PATH_FINDER_H

#include <vector>
#include <cstdint>

namespace syncvision {

// ---------------------------------------------------------------------------
// A single waypoint along a computed path.
// ---------------------------------------------------------------------------
struct Waypoint {
    float x;           // X coordinate in pixel space
    float y;           // Y coordinate in pixel space
    float cost;        // Cumulative cost to reach this waypoint

    Waypoint() : x(0.0f), y(0.0f), cost(0.0f) {}
    Waypoint(float x_, float y_, float cost_ = 0.0f)
        : x(x_), y(y_), cost(cost_) {}
};

// ---------------------------------------------------------------------------
// Result of a pathfinding operation.
// ---------------------------------------------------------------------------
struct PathResult {
    std::vector<Waypoint> waypoints;  // Ordered list of path waypoints
    float totalCost;                   // Total path cost
    bool isClear;                      // True if a clear (obstacle-free) path
                                       // was found

    PathResult() : totalCost(0.0f), isClear(false) {}
};

// ---------------------------------------------------------------------------
// PathFinder
//
// Finds navigable paths through a scene using depth map data and optional
// obstacle markers. The algorithm:
//   1. Extracts the ground plane from the depth map (consistent low-depth
//      regions).
//   2. Discretizes the ground plane into a navigation grid.
//   3. Runs A* pathfinding from bottom-center of the frame to the target.
//   4. Post-processes the path to produce smooth waypoints.
// ---------------------------------------------------------------------------
class PathFinder {
public:
    // Grid resolution for pathfinding (pixels per grid cell)
    static constexpr int DEFAULT_GRID_RESOLUTION = 8;

    // Depth thresholds for ground-plane classification
    static constexpr float GROUND_DEPTH_MIN = 0.0f;
    static constexpr float GROUND_DEPTH_MAX = 0.5f;

    // Maximum number of cells A* will explore before giving up
    static constexpr int MAX_EXPLORED_NODES = 100000;

    PathFinder(int gridResolution = DEFAULT_GRID_RESOLUTION);
    ~PathFinder() = default;

    // Non-copyable, movable
    PathFinder(const PathFinder&) = delete;
    PathFinder& operator=(const PathFinder&) = delete;
    PathFinder(PathFinder&&) = default;
    PathFinder& operator=(PathFinder&&) = default;

    // -----------------------------------------------------------------------
    // findPath
    //   Computes a navigable path through the scene.
    //
    //   depthMap     — flat float array of depth values [0, 1], row-major
    //   width        — depth map width
    //   height       — depth map height
    //   obstacles    — flat float array marking obstacle positions; each
    //                  obstacle is encoded as 4 floats: x, y, radius, intensity
    //   numObstacles — number of obstacle entries (each = 4 floats)
    //
    //   Returns a PathResult with waypoints and cost information.
    // -----------------------------------------------------------------------
    PathResult findPath(const float* depthMap,
                        int width,
                        int height,
                        const float* obstacles,
                        int numObstacles);

    // -----------------------------------------------------------------------
    // extractGroundPlane
    //   Analyzes the depth map to identify ground-plane regions.
    //   Ground = areas with consistent, low depth values.
    //
    //   Returns a 2D boolean grid where true = ground, false = obstacle/wall.
    //   The grid dimensions are ceil(width/resolution) x ceil(height/resolution).
    // -----------------------------------------------------------------------
    std::vector<std::vector<bool>> extractGroundPlane(
        const float* depthMap,
        int width,
        int height);

    // -----------------------------------------------------------------------
    // scorePath
    //   Evaluates the quality of a path by summing depth-map costs along it.
    //   Lower scores = smoother, more navigable paths.
    // -----------------------------------------------------------------------
    float scorePath(const std::vector<Waypoint>& path,
                    const float* depthMap,
                    int width,
                    int height);

private:
    int gridResolution_;

    // -----------------------------------------------------------------------
    // Internal A* node
    // -----------------------------------------------------------------------
    struct AStarNode {
        int gridX, gridY;       // Grid coordinates
        float gCost;            // Cost from start
        float hCost;            // Heuristic cost to goal
        float fCost() const { return gCost + hCost; }
        int parentX, parentY;   // Parent node for path reconstruction
        bool closed;            // Whether this node has been evaluated

        AStarNode()
            : gridX(0), gridY(0), gCost(0.0f), hCost(0.0f),
              parentX(-1), parentY(-1), closed(false) {}
    };

    // -----------------------------------------------------------------------
    // Internal: Convert pixel coordinates to grid coordinates
    // -----------------------------------------------------------------------
    int toGridX(int pixelX) const;
    int toGridY(int pixelY) const;

    // -----------------------------------------------------------------------
    // Internal: Convert grid coordinates back to pixel coordinates (center)
    // -----------------------------------------------------------------------
    float toPixelX(int gridX, int gridWidth) const;
    float toPixelY(int gridY, int gridHeight) const;

    // -----------------------------------------------------------------------
    // Internal: Compute heuristic (Euclidean distance in grid space)
    // -----------------------------------------------------------------------
    static float heuristic(int x1, int y1, int x2, int y2);

    // -----------------------------------------------------------------------
    // Internal: Check if a grid cell is traversable
    // -----------------------------------------------------------------------
    bool isTraversable(int gridX, int gridY,
                       const std::vector<std::vector<bool>>& groundPlane) const;

    // -----------------------------------------------------------------------
    // Internal: Stamp obstacles onto the ground plane grid
    // -----------------------------------------------------------------------
    void stampObstacles(std::vector<std::vector<bool>>& groundPlane,
                        const float* obstacles,
                        int numObstacles,
                        int gridWidth,
                        int gridHeight) const;

    // -----------------------------------------------------------------------
    // Internal: Simplify the A* path by removing unnecessary waypoints
    // (line-of-sight check between non-adjacent waypoints).
    // -----------------------------------------------------------------------
    std::vector<Waypoint> simplifyPath(
        const std::vector<Waypoint>& rawPath,
        const std::vector<std::vector<bool>>& groundPlane) const;

    // -----------------------------------------------------------------------
    // Internal: Check if there's a clear line of sight between two grid cells
    // using Bresenham's line algorithm.
    // -----------------------------------------------------------------------
    bool hasLineOfSight(int x0, int y0, int x1, int y1,
                        const std::vector<std::vector<bool>>& groundPlane) const;
};

} // namespace syncvision

#endif // PATH_FINDER_H
