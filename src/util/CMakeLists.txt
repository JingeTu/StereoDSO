SET(PROJECT_NAME dso_util)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(BUILD_TYPE Release)

# decide if we have ziplib.
if (LIBZIP_LIBRARY)
    message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
    add_definitions(-DHAS_ZIPLIB=1)
    include_directories(${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF})
else ()
    message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
    set(LIBZIP_LIBRARY "")
endif ()

add_library(${PROJECT_NAME}
        settings.cpp
        Undistort.cpp
        globalCalib.cpp
        IMUPropagation.cpp)
target_link_libraries(${PROJECT_NAME} okvis_kinematics)