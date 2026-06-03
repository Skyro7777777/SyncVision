# =============================================================================
# File: CMakeLists.txt
# Description: CMake build configuration for the Sync Vision native library.
#              Builds the "syncvision_native" shared library with OpenCV
#              integration for contour processing, pathfinding, and diagram
#              generation.
# Author: Sync Vision Team
# =============================================================================

cmake_minimum_required(VERSION 3.22.1)

project("syncvision_native" LANGUAGES CXX)

# ---------------------------------------------------------------------------
# OpenCV SDK path — override with -DOPENCV_ANDROID_SDK=<path>
# When building without OpenCV, the native code falls back to stub/basic
# implementations so the library still compiles and links.
# ---------------------------------------------------------------------------
set(OPENCV_ANDROID_SDK "" CACHE PATH "Path to OpenCV Android SDK root")

if(OPENCV_ANDROID_SDK)
    set(OpenCV_DIR "${OPENCV_ANDROID_SDK}/sdk/native/jni")
    find_package(OpenCV REQUIRED COMPONENTS core imgproc)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    add_definitions(-DSYNCVISION_USE_OPENCV)
else()
    message(STATUS "OpenCV SDK not provided — building with stub implementations")
endif()

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(SYNCVISION_SOURCES
    jni_bridge.cpp
    contour_processor.cpp
    label_placer.cpp
    path_finder.cpp
    sync_diagram.cpp
)

# ---------------------------------------------------------------------------
# Shared library target
# ---------------------------------------------------------------------------
add_library(syncvision_native SHARED ${SYNCVISION_SOURCES})

# ---------------------------------------------------------------------------
# Include directories — project headers + OpenCV (if available)
# ---------------------------------------------------------------------------
target_include_directories(syncvision_native
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(OPENCV_ANDROID_SDK)
    target_include_directories(syncvision_native
        PRIVATE
            ${OpenCV_INCLUDE_DIRS}
    )
endif()

# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
target_compile_features(syncvision_native PRIVATE cxx_std_17)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
target_compile_options(syncvision_native
    PRIVATE
        -O3
        -frtti
        -fexceptions
        -Wall
        -Wextra
        -Wno-unused-parameter
)

# ---------------------------------------------------------------------------
# Link libraries
# ---------------------------------------------------------------------------
# Android platform libraries
find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)
find_library(glesv3-lib GLESv3)
find_library(egl-lib EGL)

target_link_libraries(syncvision_native
    PRIVATE
        ${log-lib}
        ${android-lib}
        ${jnigraphics-lib}
        ${glesv3-lib}
        ${egl-lib}
)

# OpenCV libraries (if available)
if(OPENCV_ANDROID_SDK)
    target_link_libraries(syncvision_native
        PRIVATE
            ${OpenCV_LIBS}
    )
endif()

# ---------------------------------------------------------------------------
# ABI support — arm64-v8a, armeabi-v7a, x86_64
# The actual ABI filtering is handled by the Gradle ndk.abiFilters config
# in app/build.gradle.kts, but we validate here for clarity.
# ---------------------------------------------------------------------------
if(ANDROID_ABI)
    message(STATUS "Building for ABI: ${ANDROID_ABI}")
endif()

# ---------------------------------------------------------------------------
# Stl configuration — use c++_shared for full C++ standard library support
# ---------------------------------------------------------------------------
set(ANDROID_STL c++_shared)
