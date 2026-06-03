// =============================================================================
// File: sync_diagram.h
// Description: Header for the SyncDiagramGenerator class. Generates
//              relationship diagrams (nodes + edges) from detected objects
//              using force-directed layout and spatial relationship analysis.
// Author: Sync Vision Team
// =============================================================================

#ifndef SYNC_DIAGRAM_H
#define SYNC_DIAGRAM_H

#include <vector>
#include <string>
#include "label_placer.h"  // For DetectedObj

namespace syncvision {

// ---------------------------------------------------------------------------
// Relationship types between detected objects.
// ---------------------------------------------------------------------------
enum class RelationshipType {
    ON,         // Object A is on top of object B (e.g., cup ON table)
    NEAR,       // Object A is spatially near object B
    CONTAINS,   // Object A contains object B (e.g., room CONTAINS person)
    BLOCKS,     // Object A partially blocks/occludes object B
    SUPPORTS    // Object A physically supports object B (e.g., table SUPPORTS cup)
};

// ---------------------------------------------------------------------------
// Convert a RelationshipType to its string representation.
// ---------------------------------------------------------------------------
const char* relationshipToString(RelationshipType type);

// ---------------------------------------------------------------------------
// A node in the sync diagram, representing a detected object.
// ---------------------------------------------------------------------------
struct DiagramNode {
    int id;             // Matches DetectedObj::id
    std::string name;   // Object class name
    float x;            // Layout X position [0, 1] normalized
    float y;            // Layout Y position [0, 1] normalized
    int iconType;       // Icon type hint for rendering (0=default, 1=person,
                        //   2=vehicle, 3=furniture, 4=electronics, 5=nature, etc.)

    DiagramNode()
        : id(0), name(""), x(0.0f), y(0.0f), iconType(0) {}

    DiagramNode(int id_, const std::string& name_,
                float x_, float y_, int icon = 0)
        : id(id_), name(name_), x(x_), y(y_), iconType(icon) {}
};

// ---------------------------------------------------------------------------
// An edge in the sync diagram, representing a relationship between two objects.
// ---------------------------------------------------------------------------
struct DiagramEdge {
    int fromId;                 // Source node ID
    int toId;                   // Target node ID
    RelationshipType relationship; // Type of relationship

    DiagramEdge()
        : fromId(0), toId(0), relationship(RelationshipType::NEAR) {}

    DiagramEdge(int from, int to, RelationshipType rel)
        : fromId(from), toId(to), relationship(rel) {}
};

// ---------------------------------------------------------------------------
// The complete result of diagram generation.
// ---------------------------------------------------------------------------
struct DiagramResult {
    std::vector<DiagramNode> nodes;
    std::vector<DiagramEdge> edges;
};

// ---------------------------------------------------------------------------
// SyncDiagramGenerator
//
// Generates a force-directed layout diagram showing relationships between
// detected objects in a scene. The algorithm:
//   1. Creates a node for each detected object.
//   2. Extracts spatial relationships (ON, NEAR, CONTAINS, BLOCKS, SUPPORTS)
//      based on bounding-box geometry.
//   3. Runs a force-directed layout simulation to position nodes.
//   4. Returns the positioned nodes and edges.
// ---------------------------------------------------------------------------
class SyncDiagramGenerator {
public:
    // Force-directed layout parameters
    static constexpr float REPULSION_STRENGTH   = 0.5f;
    static constexpr float ATTRACTION_STRENGTH   = 0.05f;
    static constexpr float CENTER_GRAVITY        = 0.01f;
    static constexpr int   NUM_ITERATIONS        = 100;
    static constexpr float DAMPING               = 0.9f;
    static constexpr float MIN_DISTANCE          = 0.05f;
    static constexpr float MAX_DISPLACEMENT      = 0.1f;

    // Spatial relationship thresholds (in normalized [0,1] coordinates)
    static constexpr float NEAR_THRESHOLD        = 0.25f;
    static constexpr float CONTAINMENT_THRESHOLD = 0.7f;  // % of smaller bbox
    static constexpr float BLOCK_THRESHOLD       = 0.3f;  // % overlap

    SyncDiagramGenerator() = default;
    ~SyncDiagramGenerator() = default;

    // -----------------------------------------------------------------------
    // generate
    //   Creates a sync diagram from the given list of detected objects.
    //
    //   objects — list of detected objects from the ML pipeline
    //
    //   Returns a DiagramResult with positioned nodes and relationship edges.
    // -----------------------------------------------------------------------
    DiagramResult generate(const std::vector<DetectedObj>& objects);

private:
    // -----------------------------------------------------------------------
    // Internal: Initialize node positions based on object bounding boxes.
    //   Positions are normalized to [0, 1] range.
    // -----------------------------------------------------------------------
    std::vector<DiagramNode> initializeNodes(
        const std::vector<DetectedObj>& objects) const;

    // -----------------------------------------------------------------------
    // Internal: Extract spatial relationships between objects based on their
    // bounding-box positions and overlaps.
    // -----------------------------------------------------------------------
    std::vector<DiagramEdge> extractRelationships(
        const std::vector<DetectedObj>& objects) const;

    // -----------------------------------------------------------------------
    // Internal: Run the force-directed layout simulation.
    //   - Repulsion between all node pairs (Coulomb's law)
    //   - Attraction between connected nodes (Hooke's law)
    //   - Gravity toward the center
    // -----------------------------------------------------------------------
    void runForceLayout(std::vector<DiagramNode>& nodes,
                        const std::vector<DiagramEdge>& edges) const;

    // -----------------------------------------------------------------------
    // Internal: Determine the icon type for a given object name.
    // -----------------------------------------------------------------------
    static int classifyIconType(const std::string& name);

    // -----------------------------------------------------------------------
    // Internal: Compute the overlap area between two bounding boxes as a
    // fraction of the smaller box's area.
    // -----------------------------------------------------------------------
    static float computeOverlap(const DetectedObj& a,
                                 const DetectedObj& b);

    // -----------------------------------------------------------------------
    // Internal: Check if bounding box A contains bounding box B (or vice
    // versa). Returns the containment ratio.
    // -----------------------------------------------------------------------
    static float computeContainment(const DetectedObj& outer,
                                     const DetectedObj& inner);

    // -----------------------------------------------------------------------
    // Internal: Compute the Euclidean distance between the centers of two
    // bounding boxes, normalized to [0, sqrt(2)].
    // -----------------------------------------------------------------------
    static float centerDistance(const DetectedObj& a,
                                const DetectedObj& b,
                                int canvasW,
                                int canvasH);
};

} // namespace syncvision

#endif // SYNC_DIAGRAM_H
