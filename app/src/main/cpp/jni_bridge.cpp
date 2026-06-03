// =============================================================================
// File: jni_bridge.cpp
// Description: JNI entry points for the Sync Vision native library. Provides
//              the Java/Kotlin layer with access to C++ processing functions
//              for contour extraction, label placement, pathfinding, and
//              diagram generation.
//
//              Package: com.syncvision.app.nativelib.NativeProcessor
//
//              All methods currently return stub/placeholder data. Real OpenCV
//              integration will be filled in when building with the actual
//              OpenCV SDK.
// Author: Sync Vision Team
// =============================================================================

#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>

#include <vector>
#include <string>
#include <cstring>

// Project headers
#include "contour_processor.h"
#include "label_placer.h"
#include "path_finder.h"
#include "sync_diagram.h"

#define LOG_TAG "SyncVision_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace syncvision;

// ============================================================================
// Helper: Convert a vector of contour points to a flat jintArray
//   Format: [x1, y1, x2, y2, x3, y3, ...]
// ============================================================================
static jintArray contoursToJintArray(JNIEnv* env,
                                      const std::vector<std::vector<Point>>& contours)
{
    // Count total points
    int totalPoints = 0;
    for (const auto& contour : contours) {
        totalPoints += static_cast<int>(contour.size());
    }

    // We need 2 ints per point (x, y), plus 1 int for the number of contours
    // at the start, plus (numContours) ints for contour sizes.
    // Layout: [numContours, size0, size1, ..., sizeN, x0,y0, x1,y1, ...]
    int headerSize = 1 + static_cast<int>(contours.size());
    int dataSize = totalPoints * 2;
    int totalSize = headerSize + dataSize;

    jintArray result = env->NewIntArray(totalSize);
    if (!result) {
        LOGE("contoursToJintArray: failed to allocate jintArray of size %d", totalSize);
        return nullptr;
    }

    std::vector<jint> buffer(totalSize);

    // Header: number of contours
    buffer[0] = static_cast<jint>(contours.size());

    // Header: size of each contour
    for (size_t i = 0; i < contours.size(); ++i) {
        buffer[1 + i] = static_cast<jint>(contours[i].size());
    }

    // Data: flattened points
    int offset = headerSize;
    for (const auto& contour : contours) {
        for (const auto& pt : contour) {
            buffer[offset++] = static_cast<jint>(pt.x);
            buffer[offset++] = static_cast<jint>(pt.y);
        }
    }

    env->SetIntArrayRegion(result, 0, totalSize, buffer.data());
    return result;
}

// ============================================================================
// Helper: Convert a vector of Points to a flat jintArray
//   Format: [x1, y1, x2, y2, ...]
// ============================================================================
static jintArray pointsToJintArray(JNIEnv* env,
                                    const std::vector<Point>& points)
{
    int totalSize = static_cast<int>(points.size()) * 2;

    jintArray result = env->NewIntArray(totalSize);
    if (!result) {
        LOGE("pointsToJintArray: failed to allocate jintArray of size %d", totalSize);
        return nullptr;
    }

    std::vector<jint> buffer(totalSize);
    for (size_t i = 0; i < points.size(); ++i) {
        buffer[i * 2]     = static_cast<jint>(points[i].x);
        buffer[i * 2 + 1] = static_cast<jint>(points[i].y);
    }

    env->SetIntArrayRegion(result, 0, totalSize, buffer.data());
    return result;
}

// ============================================================================
// Helper: Parse a jobjectArray of DetectedObj from Java
//   Expected Java type: Array of NativeProcessor.DetectedObject
//   Fields: int id, String name, int bboxX, int bboxY, int bboxW, int bboxH,
//           float confidence
// ============================================================================
static std::vector<DetectedObj> parseDetectedObjects(JNIEnv* env,
                                                       jobjectArray objArray)
{
    std::vector<DetectedObj> objects;

    if (!objArray) {
        LOGW("parseDetectedObjects: null object array");
        return objects;
    }

    jsize len = env->GetArrayLength(objArray);
    objects.reserve(len);

    for (jsize i = 0; i < len; ++i) {
        jobject obj = env->GetObjectArrayElement(objArray, i);
        if (!obj) {
            LOGW("parseDetectedObjects: null object at index %d", i);
            continue;
        }

        // Get the class of the object
        jclass cls = env->GetObjectClass(obj);
        if (!cls) {
            LOGW("parseDetectedObjects: could not get class at index %d", i);
            env->DeleteLocalRef(obj);
            continue;
        }

        // Read fields via JNI
        jfieldID fidId    = env->GetFieldID(cls, "id", "I");
        jfieldID fidName  = env->GetFieldID(cls, "name", "Ljava/lang/String;");
        jfieldID fidBboxX = env->GetFieldID(cls, "bboxX", "I");
        jfieldID fidBboxY = env->GetFieldID(cls, "bboxY", "I");
        jfieldID fidBboxW = env->GetFieldID(cls, "bboxW", "I");
        jfieldID fidBboxH = env->GetFieldID(cls, "bboxH", "I");
        jfieldID fidConf  = env->GetFieldID(cls, "confidence", "F");

        if (!fidId || !fidName || !fidBboxX || !fidBboxY ||
            !fidBboxW || !fidBboxH || !fidConf) {
            LOGW("parseDetectedObjects: could not find field IDs at index %d", i);
            env->DeleteLocalRef(cls);
            env->DeleteLocalRef(obj);
            continue;
        }

        int id       = env->GetIntField(obj, fidId);
        jstring jstr = static_cast<jstring>(env->GetObjectField(obj, fidName));
        int bboxX    = env->GetIntField(obj, fidBboxX);
        int bboxY    = env->GetIntField(obj, fidBboxY);
        int bboxW    = env->GetIntField(obj, fidBboxW);
        int bboxH    = env->GetIntField(obj, fidBboxH);
        float conf   = env->GetFloatField(obj, fidConf);

        std::string name;
        if (jstr) {
            const char* cstr = env->GetStringUTFChars(jstr, nullptr);
            if (cstr) {
                name = cstr;
                env->ReleaseStringUTFChars(jstr, cstr);
            }
            env->DeleteLocalRef(jstr);
        }

        objects.emplace_back(id, name, bboxX, bboxY, bboxW, bboxH, conf);

        env->DeleteLocalRef(cls);
        env->DeleteLocalRef(obj);
    }

    return objects;
}

// ============================================================================
// JNI: processContours
//   Input:  flat mask data (int array), width, height
//   Output: flattened contour points (x1,y1,x2,y2,...) with header
// ============================================================================
extern "C" JNIEXPORT jintArray JNICALL
Java_com_syncvision_app_nativelib_NativeProcessor_processContours(
    JNIEnv* env,
    jobject /* this */,
    jintArray maskData,
    jint width,
    jint height)
{
    LOGI("processContours: w=%d, h=%d", width, height);

    if (!maskData || width <= 0 || height <= 0) {
        LOGE("processContours: invalid parameters");
        return env->NewIntArray(0);
    }

    // Copy mask data from Java array
    jsize dataLen = env->GetArrayLength(maskData);
    if (dataLen != width * height) {
        LOGE("processContours: mask data length (%d) != width*height (%d)",
             dataLen, width * height);
        return env->NewIntArray(0);
    }

    jint* maskPtr = env->GetIntArrayElements(maskData, nullptr);
    if (!maskPtr) {
        LOGE("processContours: failed to get mask data");
        return env->NewIntArray(0);
    }

    // Process the mask
    ContourProcessor processor;
    auto contours = processor.processMask(
        reinterpret_cast<const int*>(maskPtr),
        static_cast<int>(width),
        static_cast<int>(height));

    env->ReleaseIntArrayElements(maskData, maskPtr, JNI_ABORT);

    // Convert to jintArray
    return contoursToJintArray(env, contours);
}

// ============================================================================
// JNI: simplifyContours
//   Input:  flat contour points (x1,y1,x2,y2,...), epsilon
//   Output: simplified contour points
// ============================================================================
extern "C" JNIEXPORT jintArray JNICALL
Java_com_syncvision_app_nativelib_NativeProcessor_simplifyContours(
    JNIEnv* env,
    jobject /* this */,
    jintArray contourPoints,
    jdouble epsilon)
{
    LOGI("simplifyContours: epsilon=%.4f", epsilon);

    if (!contourPoints) {
        LOGE("simplifyContours: null contour points");
        return env->NewIntArray(0);
    }

    jsize len = env->GetArrayLength(contourPoints);
    if (len < 4 || len % 2 != 0) {
        LOGE("simplifyContours: invalid contour length %d", len);
        return env->NewIntArray(0);
    }

    jint* pts = env->GetIntArrayElements(contourPoints, nullptr);
    if (!pts) {
        LOGE("simplifyContours: failed to get contour data");
        return env->NewIntArray(0);
    }

    // Convert to vector<Point>
    std::vector<Point> contour;
    contour.reserve(len / 2);
    for (jsize i = 0; i < len; i += 2) {
        contour.emplace_back(pts[i], pts[i + 1]);
    }

    env->ReleaseIntArrayElements(contourPoints, pts, JNI_ABORT);

    // Simplify
    ContourProcessor processor;
    auto simplified = processor.simplifyContour(contour, epsilon);

    return pointsToJintArray(env, simplified);
}

// ============================================================================
// JNI: placeLabels
//   Input:  array of DetectedObj, canvas width, canvas height
//   Output: array of label placements (each = [x, y, fontSize])
//           as a flattened float array: [x1, y1, fontSize1, ...]
//           with text strings returned separately
// ============================================================================
extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_syncvision_app_nativelib_NativeProcessor_placeLabels(
    JNIEnv* env,
    jobject /* this */,
    jobjectArray objects,
    jint width,
    jint height)
{
    LOGI("placeLabels: canvas=%dx%d", width, height);

    if (!objects || width <= 0 || height <= 0) {
        LOGE("placeLabels: invalid parameters");
        return nullptr;
    }

    // Parse detected objects from Java
    auto detectedObjs = parseDetectedObjects(env, objects);
    if (detectedObjs.empty()) {
        LOGW("placeLabels: no objects to place labels for");
        return nullptr;
    }

    // Run label placement
    LabelPlacer placer;
    auto placements = placer.placeLabels(detectedObjs, width, height);

    // Find the LabelPlacement Java class
    // Expected: com.syncvision.app.nativelib.NativeProcessor$LabelPlacement
    jclass placementClass = env->FindClass(
        "com/syncvision/app/nativelib/NativeProcessor$LabelPlacement");
    if (!placementClass) {
        LOGE("placeLabels: could not find LabelPlacement class");

        // Fallback: return an Object[] of int[3] arrays (x, y, fontSize)
        // This is a simpler format that doesn't require a custom Java class
        jclass intArrayClass = env->FindClass("[I");
        jobjectArray result = env->NewObjectArray(
            static_cast<jsize>(placements.size()), intArrayClass, nullptr);

        for (jsize i = 0; i < static_cast<jsize>(placements.size()); ++i) {
            jint arr[3] = {
                placements[i].x,
                placements[i].y,
                placements[i].fontSize
            };
            jintArray elem = env->NewIntArray(3);
            env->SetIntArrayRegion(elem, 0, 3, arr);
            env->SetObjectArrayElement(result, i, elem);
            env->DeleteLocalRef(elem);
        }
        return result;
    }

    // Get constructor and field IDs
    jmethodID constructor = env->GetMethodID(placementClass, "<init>", "()V");
    jfieldID fidX        = env->GetFieldID(placementClass, "x", "I");
    jfieldID fidY        = env->GetFieldID(placementClass, "y", "I");
    jfieldID fidFontSize = env->GetFieldID(placementClass, "fontSize", "I");
    jfieldID fidText     = env->GetFieldID(placementClass, "text",
                                             "Ljava/lang/String;");

    if (!constructor || !fidX || !fidY || !fidFontSize || !fidText) {
        LOGE("placeLabels: could not find LabelPlacement fields");
        env->DeleteLocalRef(placementClass);
        return nullptr;
    }

    // Build the result array
    jobjectArray result = env->NewObjectArray(
        static_cast<jsize>(placements.size()), placementClass, nullptr);

    for (jsize i = 0; i < static_cast<jsize>(placements.size()); ++i) {
        jobject labelObj = env->NewObject(placementClass, constructor);
        env->SetIntField(labelObj, fidX, placements[i].x);
        env->SetIntField(labelObj, fidY, placements[i].y);
        env->SetIntField(labelObj, fidFontSize, placements[i].fontSize);

        jstring jtext = env->NewStringUTF(placements[i].text.c_str());
        env->SetObjectField(labelObj, fidText, jtext);
        env->DeleteLocalRef(jtext);

        env->SetObjectArrayElement(result, i, labelObj);
        env->DeleteLocalRef(labelObj);
    }

    env->DeleteLocalRef(placementClass);
    return result;
}

// ============================================================================
// JNI: findPath
//   Input:  depth map (float array), width, height, obstacles (float array)
//   Output: path waypoints (float array: x1,y1,cost1, x2,y2,cost2, ...)
//           First 3 floats: totalCost, isClear (0/1), numWaypoints
// ============================================================================
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_syncvision_app_nativelib_NativeProcessor_findPath(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray depthMap,
    jint width,
    jint height,
    jfloatArray obstacles)
{
    LOGI("findPath: depth=%dx%d", width, height);

    if (!depthMap || width <= 0 || height <= 0) {
        LOGE("findPath: invalid parameters");
        return env->NewFloatArray(0);
    }

    // Copy depth map
    jsize depthLen = env->GetArrayLength(depthMap);
    if (depthLen != width * height) {
        LOGE("findPath: depth map length mismatch (%d != %d)",
             depthLen, width * height);
        return env->NewFloatArray(0);
    }

    float* depthPtr = env->GetFloatArrayElements(depthMap, nullptr);
    if (!depthPtr) {
        LOGE("findPath: failed to get depth data");
        return env->NewFloatArray(0);
    }

    // Copy obstacles (optional)
    float* obsPtr = nullptr;
    int numObstacles = 0;
    if (obstacles) {
        jsize obsLen = env->GetArrayLength(obstacles);
        if (obsLen > 0 && obsLen % 4 == 0) {
            obsPtr = env->GetFloatArrayElements(obstacles, nullptr);
            numObstacles = obsLen / 4;
        }
    }

    // Run pathfinding
    PathFinder pathFinder;
    auto result = pathFinder.findPath(depthPtr, width, height,
                                       obsPtr, numObstacles);

    // Release JNI arrays
    env->ReleaseFloatArrayElements(depthMap, depthPtr, JNI_ABORT);
    if (obsPtr) {
        env->ReleaseFloatArrayElements(obstacles, obsPtr, JNI_ABORT);
    }

    // Build output: [totalCost, isClear, numWaypoints, x1,y1,cost1, ...]
    int headerSize = 3;
    int waypointDataSize = static_cast<int>(result.waypoints.size()) * 3;
    int totalSize = headerSize + waypointDataSize;

    jfloatArray output = env->NewFloatArray(totalSize);
    if (!output) {
        LOGE("findPath: failed to allocate output array");
        return env->NewFloatArray(0);
    }

    std::vector<jfloat> buffer(totalSize);
    buffer[0] = result.totalCost;
    buffer[1] = result.isClear ? 1.0f : 0.0f;
    buffer[2] = static_cast<jfloat>(result.waypoints.size());

    for (size_t i = 0; i < result.waypoints.size(); ++i) {
        int offset = headerSize + static_cast<int>(i) * 3;
        buffer[offset]     = result.waypoints[i].x;
        buffer[offset + 1] = result.waypoints[i].y;
        buffer[offset + 2] = result.waypoints[i].cost;
    }

    env->SetFloatArrayRegion(output, 0, totalSize, buffer.data());
    return output;
}

// ============================================================================
// JNI: generateSyncDiagram
//   Input:  array of DetectedObj
//   Output: float array with diagram data
//           Layout:
//             [numNodes, numEdges,
//              node0: id, x, y, iconType,
//              node1: id, x, y, iconType,
//              ...
//              edge0: fromId, toId, relationshipInt,
//              edge1: fromId, toId, relationshipInt,
//              ...]
//           Node names are not included in the float array; they should be
//           looked up by ID from the input objects.
// ============================================================================
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_syncvision_app_nativelib_NativeProcessor_generateSyncDiagram(
    JNIEnv* env,
    jobject /* this */,
    jobjectArray objects)
{
    LOGI("generateSyncDiagram");

    if (!objects) {
        LOGE("generateSyncDiagram: null objects array");
        return env->NewFloatArray(0);
    }

    // Parse detected objects
    auto detectedObjs = parseDetectedObjects(env, objects);
    if (detectedObjs.empty()) {
        LOGW("generateSyncDiagram: no objects");
        return env->NewFloatArray(0);
    }

    // Generate diagram
    SyncDiagramGenerator generator;
    auto result = generator.generate(detectedObjs);

    // Build output float array
    int headerSize = 2;
    int nodeDataSize = static_cast<int>(result.nodes.size()) * 4; // id, x, y, iconType
    int edgeDataSize = static_cast<int>(result.edges.size()) * 3; // fromId, toId, relInt
    int totalSize = headerSize + nodeDataSize + edgeDataSize;

    jfloatArray output = env->NewFloatArray(totalSize);
    if (!output) {
        LOGE("generateSyncDiagram: failed to allocate output array");
        return env->NewFloatArray(0);
    }

    std::vector<jfloat> buffer(totalSize);
    buffer[0] = static_cast<jfloat>(result.nodes.size());
    buffer[1] = static_cast<jfloat>(result.edges.size());

    int offset = headerSize;

    // Nodes
    for (const auto& node : result.nodes) {
        buffer[offset++] = static_cast<jfloat>(node.id);
        buffer[offset++] = node.x;
        buffer[offset++] = node.y;
        buffer[offset++] = static_cast<jfloat>(node.iconType);
    }

    // Edges
    for (const auto& edge : result.edges) {
        buffer[offset++] = static_cast<jfloat>(edge.fromId);
        buffer[offset++] = static_cast<jfloat>(edge.toId);
        buffer[offset++] = static_cast<jfloat>(static_cast<int>(edge.relationship));
    }

    env->SetFloatArrayRegion(output, 0, totalSize, buffer.data());
    return output;
}

// ============================================================================
// JNI: applyCannyEdge
//   Input:  image data (ARGB int array), width, height, thresholds
//   Output: edge mask (int array: 0 or 255 per pixel)
// ============================================================================
extern "C" JNIEXPORT jintArray JNICALL
Java_com_syncvision_app_nativelib_NativeProcessor_applyCannyEdge(
    JNIEnv* env,
    jobject /* this */,
    jintArray imageData,
    jint width,
    jint height,
    jdouble lowThreshold,
    jdouble highThreshold)
{
    LOGI("applyCannyEdge: %dx%d, low=%.2f, high=%.2f",
         width, height, lowThreshold, highThreshold);

    if (!imageData || width <= 0 || height <= 0) {
        LOGE("applyCannyEdge: invalid parameters");
        return env->NewIntArray(0);
    }

    jsize dataLen = env->GetArrayLength(imageData);
    if (dataLen != width * height) {
        LOGE("applyCannyEdge: data length mismatch (%d != %d)",
             dataLen, width * height);
        return env->NewIntArray(0);
    }

    jint* imgPtr = env->GetIntArrayElements(imageData, nullptr);
    if (!imgPtr) {
        LOGE("applyCannyEdge: failed to get image data");
        return env->NewIntArray(0);
    }

#ifdef SYNCVISION_USE_OPENCV
    // -----------------------------------------------------------------------
    // OpenCV path: convert ARGB to grayscale, apply Canny
    // -----------------------------------------------------------------------
    try {
        // Create grayscale image from ARGB data
        cv::Mat gray(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int argb = imgPtr[y * width + x];
                // Extract RGB components (ARGB format)
                int r = (argb >> 16) & 0xFF;
                int g = (argb >> 8)  & 0xFF;
                int b =  argb        & 0xFF;
                // Luminance formula
                gray.at<uint8_t>(y, x) = static_cast<uint8_t>(
                    0.299 * r + 0.587 * g + 0.114 * b);
            }
        }

        cv::Mat edges;
        cv::Canny(gray, edges, lowThreshold, highThreshold, 3);

        // Convert edge mask to jintArray
        jintArray result = env->NewIntArray(width * height);
        if (result) {
            std::vector<jint> edgeData(width * height);
            for (int i = 0; i < width * height; ++i) {
                edgeData[i] = edges.at<uint8_t>(i) > 0 ? 255 : 0;
            }
            env->SetIntArrayRegion(result, 0, width * height, edgeData.data());
        }

        env->ReleaseIntArrayElements(imageData, imgPtr, JNI_ABORT);
        return result ? result : env->NewIntArray(0);

    } catch (const cv::Exception& e) {
        LOGE("OpenCV exception in applyCannyEdge: %s", e.what());
    } catch (const std::exception& e) {
        LOGE("Exception in applyCannyEdge: %s", e.what());
    }
#endif

    // -----------------------------------------------------------------------
    // Basic fallback: Sobel edge detection
    // -----------------------------------------------------------------------
    // Convert ARGB to grayscale
    std::vector<uint8_t> gray(width * height);
    for (int i = 0; i < width * height; ++i) {
        int argb = imgPtr[i];
        int r = (argb >> 16) & 0xFF;
        int g = (argb >> 8)  & 0xFF;
        int b =  argb        & 0xFF;
        gray[i] = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
    }

    env->ReleaseIntArrayElements(imageData, imgPtr, JNI_ABORT);

    // Apply Sobel operator
    std::vector<int> edges(width * height, 0);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            // Sobel X kernel
            int gx = -gray[(y-1)*width + (x-1)] - 2*gray[y*width + (x-1)] - gray[(y+1)*width + (x-1)]
                     +gray[(y-1)*width + (x+1)] + 2*gray[y*width + (x+1)] + gray[(y+1)*width + (x+1)];

            // Sobel Y kernel
            int gy = -gray[(y-1)*width + (x-1)] - 2*gray[(y-1)*width + x] - gray[(y-1)*width + (x+1)]
                     +gray[(y+1)*width + (x-1)] + 2*gray[(y+1)*width + x] + gray[(y+1)*width + (x+1)];

            int magnitude = static_cast<int>(std::sqrt(
                static_cast<double>(gx * gx + gy * gy)));

            // Apply threshold
            double threshold = (lowThreshold + highThreshold) / 2.0;
            edges[y * width + x] = (magnitude > threshold) ? 255 : 0;
        }
    }

    // Convert to jintArray
    jintArray result = env->NewIntArray(width * height);
    if (result) {
        std::vector<jint> edgeData(width * height);
        for (int i = 0; i < width * height; ++i) {
            edgeData[i] = edges[i];
        }
        env->SetIntArrayRegion(result, 0, width * height, edgeData.data());
    }

    return result ? result : env->NewIntArray(0);
}

// ============================================================================
// Library initialization â€” called when the native library is loaded
// ============================================================================
extern "C" JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* /* reserved */)
{
    LOGI("JNI_OnLoad: Sync Vision native library loaded");
    return JNI_VERSION_1_6;
}

// ============================================================================
// Library cleanup â€” called when the VM is shutting down
// ============================================================================
extern "C" JNIEXPORT void JNICALL
JNI_OnUnload(JavaVM* vm, void* /* reserved */)
{
    LOGI("JNI_OnUnload: Sync Vision native library unloaded");
}
