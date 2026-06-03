// =============================================================================
// File: sync_diagram.cpp
// Description: Implementation of the SyncDiagramGenerator class. Generates
//              relationship diagrams from detected objects using force-directed
//              layout and spatial analysis for 5 relationship types.
// Author: Sync Vision Team
// =============================================================================

#include "sync_diagram.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <android/log.h>

#define LOG_TAG "SyncVision_Diagram"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace syncvision {

// ============================================================================
// relationshipToString
// ============================================================================
const char* relationshipToString(RelationshipType type)
{
    switch (type) {
        case RelationshipType::ON:       return "ON";
        case RelationshipType::NEAR:     return "NEAR";
        case RelationshipType::CONTAINS: return "CONTAINS";
        case RelationshipType::BLOCKS:   return "BLOCKS";
        case RelationshipType::SUPPORTS: return "SUPPORTS";
        default:                         return "UNKNOWN";
    }
}

// ============================================================================
// generate — main entry point
// ============================================================================
DiagramResult SyncDiagramGenerator::generate(
    const std::vector<DetectedObj>& objects)
{
    DiagramResult result;

    if (objects.empty()) {
        LOGI("generate: no objects to diagram");
        return result;
    }

    // Step 1: Initialize nodes from objects
    result.nodes = initializeNodes(objects);

    // Step 2: Extract spatial relationships
    result.edges = extractRelationships(objects);

    // Step 3: Run force-directed layout
    runForceLayout(result.nodes, result.edges);

    LOGI("generate: %zu nodes, %zu edges",
         result.nodes.size(), result.edges.size());
    return result;
}

// ============================================================================
// Private: initializeNodes
// ============================================================================
std::vector<DiagramNode> SyncDiagramGenerator::initializeNodes(
    const std::vector<DetectedObj>& objects) const
{
    std::vector<DiagramNode> nodes;
    nodes.reserve(objects.size());

    // Find the overall scene bounds for normalization
    // (Use a default canvas if no bbox info is available)
    int maxW = 1, maxH = 1;
    for (const auto& obj : objects) {
        maxW = std::max(maxW, obj.bboxX + obj.bboxW);
        maxH = std::max(maxH, obj.bboxY + obj.bboxH);
    }

    for (const auto& obj : objects) {
        DiagramNode node;
        node.id = obj.id;
        node.name = obj.name;
        node.iconType = classifyIconType(obj.name);

        // Initialize position at the center of the object's bounding box,
        // normalized to [0, 1]
        float cx = static_cast<float>(obj.bboxX + obj.bboxW / 2) / maxW;
        float cy = static_cast<float>(obj.bboxY + obj.bboxH / 2) / maxH;

        // Clamp to [0.05, 0.95] to keep nodes away from edges
        node.x = std::max(0.05f, std::min(0.95f, cx));
        node.y = std::max(0.05f, std::min(0.95f, cy));

        nodes.push_back(node);
    }

    return nodes;
}

// ============================================================================
// Private: extractRelationships
// ============================================================================
std::vector<DiagramEdge> SyncDiagramGenerator::extractRelationships(
    const std::vector<DetectedObj>& objects) const
{
    std::vector<DiagramEdge> edges;

    if (objects.size() < 2) {
        return edges;
    }

    // Find the overall scene bounds for distance normalization
    int maxW = 1, maxH = 1;
    for (const auto& obj : objects) {
        maxW = std::max(maxW, obj.bboxX + obj.bboxW);
        maxH = std::max(maxH, obj.bboxY + obj.bboxH);
    }

    // Track which pairs already have an edge (use the stronger relationship)
    // Key: min(id1,id2) * 10000 + max(id1,id2)
    std::unordered_set<int> edgeSet;

    auto addEdge = [&](int fromId, int toId, RelationshipType rel) {
        int key = std::min(fromId, toId) * 10000 + std::max(fromId, toId);

        // Only add one edge per pair — keep the strongest relationship
        // Priority: CONTAINS > ON > SUPPORTS > BLOCKS > NEAR
        if (edgeSet.count(key)) {
            return; // Already have an edge for this pair
        }

        edgeSet.insert(key);
        edges.emplace_back(fromId, toId, rel);
    };

    for (size_t i = 0; i < objects.size(); ++i) {
        for (size_t j = i + 1; j < objects.size(); ++j) {
            const auto& objA = objects[i];
            const auto& objB = objects[j];

            // 1. Check CONTAINMENT: does one bbox contain the other?
            float containmentAB = computeContainment(objA, objB);
            float containmentBA = computeContainment(objB, objA);

            if (containmentAB >= CONTAINMENT_THRESHOLD) {
                addEdge(objA.id, objB.id, RelationshipType::CONTAINS);
                continue;
            }
            if (containmentBA >= CONTAINMENT_THRESHOLD) {
                addEdge(objB.id, objA.id, RelationshipType::CONTAINS);
                continue;
            }

            // 2. Check OVERLAP for BLOCKS relationship
            float overlap = computeOverlap(objA, objB);
            if (overlap >= BLOCK_THRESHOLD) {
                // The object that is in front (higher bboxY in screen space,
                // i.e., lower in the image) blocks the one behind
                int frontId = (objA.bboxY > objB.bboxY) ? objA.id : objB.id;
                int behindId = (objA.bboxY > objB.bboxY) ? objB.id : objA.id;
                addEdge(frontId, behindId, RelationshipType::BLOCKS);
                continue;
            }

            // 3. Check ON / SUPPORTS: one object is directly above the other
            //    and their X ranges overlap significantly
            float xOverlapRatio = 0.0f;
            {
                int overlapStart = std::max(objA.bboxX, objB.bboxX);
                int overlapEnd = std::min(objA.bboxX + objA.bboxW,
                                          objB.bboxX + objB.bboxW);
                if (overlapEnd > overlapStart) {
                    int overlapW = overlapEnd - overlapStart;
                    int minW = std::min(objA.bboxW, objB.bboxW);
                    xOverlapRatio = (minW > 0) ?
                        static_cast<float>(overlapW) / minW : 0.0f;
                }
            }

            // Check vertical adjacency (A is above B)
            int aBottom = objA.bboxY + objA.bboxH;
            int bBottom = objB.bboxY + objB.bboxH;
            int gapThreshold = std::max(objA.bboxH, objB.bboxH) / 4;

            if (xOverlapRatio > 0.3f) {
                // A is above B and close to B's top
                if (aBottom >= objB.bboxY - gapThreshold &&
                    aBottom <= objB.bboxY + gapThreshold &&
                    objA.bboxY < objB.bboxY) {
                    addEdge(objB.id, objA.id, RelationshipType::ON);
                    continue;
                }
                // B is above A and close to A's top
                if (bBottom >= objA.bboxY - gapThreshold &&
                    bBottom <= objA.bboxY + gapThreshold &&
                    objB.bboxY < objA.bboxY) {
                    addEdge(objA.id, objB.id, RelationshipType::ON);
                    continue;
                }

                // SUPPORTS: same as ON but with more gap (larger support
                // structure underneath)
                int supportGap = std::max(objA.bboxH, objB.bboxH) / 2;
                if (aBottom >= objB.bboxY - supportGap &&
                    aBottom <= objB.bboxY + supportGap &&
                    objA.bboxY < objB.bboxY) {
                    addEdge(objB.id, objA.id, RelationshipType::SUPPORTS);
                    continue;
                }
                if (bBottom >= objA.bboxY - supportGap &&
                    bBottom <= objA.bboxY + supportGap &&
                    objB.bboxY < objA.bboxY) {
                    addEdge(objA.id, objB.id, RelationshipType::SUPPORTS);
                    continue;
                }
            }

            // 4. NEAR: spatial proximity without other relationships
            float dist = centerDistance(objA, objB, maxW, maxH);
            if (dist < NEAR_THRESHOLD) {
                addEdge(objA.id, objB.id, RelationshipType::NEAR);
            }
        }
    }

    LOGI("extractRelationships: found %zu relationships", edges.size());
    return edges;
}

// ============================================================================
// Private: runForceLayout
// ============================================================================
void SyncDiagramGenerator::runForceLayout(
    std::vector<DiagramNode>& nodes,
    const std::vector<DiagramEdge>& edges) const
{
    if (nodes.size() <= 1) return;

    const size_t N = nodes.size();

    // Velocity vectors for each node
    std::vector<float> vx(N, 0.0f);
    std::vector<float> vy(N, 0.0f);

    // Build an adjacency set for fast edge lookup
    // Key: pair of node indices (i, j) where i < j
    std::unordered_set<long long> connectedPairs;
    for (const auto& edge : edges) {
        int i = -1, j = -1;
        for (size_t k = 0; k < N; ++k) {
            if (nodes[k].id == edge.fromId) i = static_cast<int>(k);
            if (nodes[k].id == edge.toId)   j = static_cast<int>(k);
        }
        if (i >= 0 && j >= 0 && i != j) {
            int lo = std::min(i, j), hi = std::max(i, j);
            connectedPairs.insert(
                static_cast<long long>(lo) * 10000LL + hi);
        }
    }

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Compute forces for each node
        std::vector<float> fx(N, 0.0f);
        std::vector<float> fy(N, 0.0f);

        // Repulsive force between all pairs (Coulomb's law)
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                float dx = nodes[j].x - nodes[i].x;
                float dy = nodes[j].y - nodes[i].y;
                float distSq = dx * dx + dy * dy;
                float dist = std::sqrt(distSq);

                if (dist < MIN_DISTANCE) {
                    dist = MIN_DISTANCE;
                    distSq = dist * dist;
                }

                // Repulsive force magnitude: F = k / d^2
                float force = REPULSION_STRENGTH / distSq;

                // Direction: from j to i (repulsion pushes apart)
                float dirX = dx / dist;
                float dirY = dy / dist;

                fx[i] -= force * dirX;
                fy[i] -= force * dirY;
                fx[j] += force * dirX;
                fy[j] += force * dirY;
            }
        }

        // Attractive force between connected pairs (Hooke's law)
        for (const auto& edge : edges) {
            int i = -1, j = -1;
            for (size_t k = 0; k < N; ++k) {
                if (nodes[k].id == edge.fromId) i = static_cast<int>(k);
                if (nodes[k].id == edge.toId)   j = static_cast<int>(k);
            }
            if (i < 0 || j < 0 || i == j) continue;

            float dx = nodes[j].x - nodes[i].x;
            float dy = nodes[j].y - nodes[i].y;
            float dist = std::sqrt(dx * dx + dy * dy);

            if (dist < 1e-6f) continue;

            // Attractive force: F = -k * d (spring)
            float force = ATTRACTION_STRENGTH * dist;

            float dirX = dx / dist;
            float dirY = dy / dist;

            fx[i] += force * dirX;
            fy[i] += force * dirY;
            fx[j] -= force * dirX;
            fy[j] -= force * dirY;
        }

        // Gravity toward center (0.5, 0.5)
        for (size_t i = 0; i < N; ++i) {
            fx[i] += CENTER_GRAVITY * (0.5f - nodes[i].x);
            fy[i] += CENTER_GRAVITY * (0.5f - nodes[i].y);
        }

        // Update velocities and positions
        for (size_t i = 0; i < N; ++i) {
            vx[i] = (vx[i] + fx[i]) * DAMPING;
            vy[i] = (vy[i] + fy[i]) * DAMPING;

            // Limit maximum displacement per iteration
            float speed = std::sqrt(vx[i] * vx[i] + vy[i] * vy[i]);
            if (speed > MAX_DISPLACEMENT) {
                float scale = MAX_DISPLACEMENT / speed;
                vx[i] *= scale;
                vy[i] *= scale;
            }

            nodes[i].x += vx[i];
            nodes[i].y += vy[i];

            // Keep nodes within [0.02, 0.98]
            nodes[i].x = std::max(0.02f, std::min(0.98f, nodes[i].x));
            nodes[i].y = std::max(0.02f, std::min(0.98f, nodes[i].y));
        }
    }

    LOGI("runForceLayout: completed %d iterations for %zu nodes",
         NUM_ITERATIONS, N);
}

// ============================================================================
// Private: classifyIconType
// ============================================================================
int SyncDiagramGenerator::classifyIconType(const std::string& name)
{
    // Simple heuristic based on object name
    // 0=default, 1=person, 2=vehicle, 3=furniture, 4=electronics, 5=nature

    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (lower.find("person") != std::string::npos ||
        lower.find("man") != std::string::npos ||
        lower.find("woman") != std::string::npos ||
        lower.find("boy") != std::string::npos ||
        lower.find("girl") != std::string::npos) {
        return 1;
    }
    if (lower.find("car") != std::string::npos ||
        lower.find("truck") != std::string::npos ||
        lower.find("bus") != std::string::npos ||
        lower.find("bicycle") != std::string::npos ||
        lower.find("motorcycle") != std::string::npos) {
        return 2;
    }
    if (lower.find("chair") != std::string::npos ||
        lower.find("table") != std::string::npos ||
        lower.find("desk") != std::string::npos ||
        lower.find("sofa") != std::string::npos ||
        lower.find("couch") != std::string::npos ||
        lower.find("bed") != std::string::npos) {
        return 3;
    }
    if (lower.find("phone") != std::string::npos ||
        lower.find("laptop") != std::string::npos ||
        lower.find("tv") != std::string::npos ||
        lower.find("monitor") != std::string::npos ||
        lower.find("computer") != std::string::npos ||
        lower.find("keyboard") != std::string::npos) {
        return 4;
    }
    if (lower.find("tree") != std::string::npos ||
        lower.find("plant") != std::string::npos ||
        lower.find("flower") != std::string::npos ||
        lower.find("grass") != std::string::npos ||
        lower.find("bush") != std::string::npos) {
        return 5;
    }

    return 0; // Default icon
}

// ============================================================================
// Private: computeOverlap
// ============================================================================
float SyncDiagramGenerator::computeOverlap(
    const DetectedObj& a, const DetectedObj& b)
{
    // Intersection rectangle
    int xStart = std::max(a.bboxX, b.bboxX);
    int yStart = std::max(a.bboxY, b.bboxY);
    int xEnd   = std::min(a.bboxX + a.bboxW, b.bboxX + b.bboxW);
    int yEnd   = std::min(a.bboxY + a.bboxH, b.bboxY + b.bboxH);

    if (xEnd <= xStart || yEnd <= yStart) {
        return 0.0f; // No overlap
    }

    int overlapArea = (xEnd - xStart) * (yEnd - yStart);
    int areaA = a.bboxW * a.bboxH;
    int areaB = b.bboxW * b.bboxH;
    int minArea = std::min(areaA, areaB);

    if (minArea <= 0) return 0.0f;

    return static_cast<float>(overlapArea) / static_cast<float>(minArea);
}

// ============================================================================
// Private: computeContainment
// ============================================================================
float SyncDiagramGenerator::computeContainment(
    const DetectedObj& outer, const DetectedObj& inner)
{
    // Check what fraction of 'inner' is inside 'outer'
    int xStart = std::max(inner.bboxX, outer.bboxX);
    int yStart = std::max(inner.bboxY, outer.bboxY);
    int xEnd   = std::min(inner.bboxX + inner.bboxW, outer.bboxX + outer.bboxW);
    int yEnd   = std::min(inner.bboxY + inner.bboxH, outer.bboxY + outer.bboxH);

    if (xEnd <= xStart || yEnd <= yStart) {
        return 0.0f;
    }

    int containedArea = (xEnd - xStart) * (yEnd - yStart);
    int innerArea = inner.bboxW * inner.bboxH;

    if (innerArea <= 0) return 0.0f;

    return static_cast<float>(containedArea) / static_cast<float>(innerArea);
}

// ============================================================================
// Private: centerDistance
// ============================================================================
float SyncDiagramGenerator::centerDistance(
    const DetectedObj& a, const DetectedObj& b,
    int canvasW, int canvasH)
{
    if (canvasW <= 0 || canvasH <= 0) return 1.0f;

    float ax = static_cast<float>(a.bboxX + a.bboxW / 2) / canvasW;
    float ay = static_cast<float>(a.bboxY + a.bboxH / 2) / canvasH;
    float bx = static_cast<float>(b.bboxX + b.bboxW / 2) / canvasW;
    float by = static_cast<float>(b.bboxY + b.bboxH / 2) / canvasH;

    float dx = bx - ax;
    float dy = by - ay;
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace syncvision
