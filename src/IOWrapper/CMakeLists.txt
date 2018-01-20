SET(PROJECT_NAME dso_IOWrapper)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(BUILD_TYPE Release)
# decide if we have pangolin
if (Pangolin_FOUND)
    message("--- found PANGOLIN, compiling dso_pangolin library.")
    include_directories(${Pangolin_INCLUDE_DIRS})
    set(dso_pangolin_SOURCE_FILES
            Pangolin/KeyFrameDisplay.cpp
            Pangolin/PangolinDSOViewer.cpp)
    set(HAS_PANGOLIN 1)
else ()
    message("--- could not find PANGOLIN, not compiling dso_pangolin library.")
    message("    this means there will be no 3D display / GUI available for dso_dataset.")
    set(dso_pangolin_SOURCE_FILES)
    set(HAS_PANGOLIN 0)
endif ()

# decide if we have openCV
if (OpenCV_FOUND)
    message("--- found OpenCV, compiling dso_opencv library.")
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(dso_opencv_SOURCE_FILES
            OpenCV/ImageDisplay_OpenCV.cpp
            OpenCV/ImageRW_OpenCV.cpp)
    set(HAS_OPENCV 1)
else ()
    message("--- could not find OpenCV, not compiling dso_opencv library.")
    message("    this means there will be no image display, and image read / load functionality.")
    set(dso_opencv_SOURCE_FILES
            ImageDisplay_dummy.cpp
            ImageRW_dummy.cpp)
    set(HAS_OPENCV 0)
endif ()

add_library(${PROJECT_NAME} ${dso_pangolin_SOURCE_FILES} ${dso_opencv_SOURCE_FILES})
#target_link_libraries(${PROJECT_NAME} dso_util)
