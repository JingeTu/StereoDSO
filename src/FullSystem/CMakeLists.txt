SET(PROJECT_NAME dso_FullSystem)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(BUILD_TYPE Release)

add_library(${PROJECT_NAME} FullSystem.cpp FullSystemOptimize.cpp FullSystemOptPoint.cpp
        FullSystemDebugStuff.cpp FullSystemMarginalize.cpp Residuals.cpp IMUResiduals.cpp
        CoarseTracker.cpp CoarseInitializer.cpp ImmaturePoint.cpp
        HessianBlocks.cpp PixelSelector2.cpp)
target_link_libraries(${PROJECT_NAME} dso_util ${GLOG_LIBRARY})
