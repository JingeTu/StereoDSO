SET(PROJECT_NAME DSO)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
#set(CMAKE_VERBOSE_MAKEFILE ON)


set(BUILD_TYPE Release)
#set(BUILD_TYPE RelWithDebInfo)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# required libraries
find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread) 

# optional libraries
find_package(LibZip QUIET)
find_package(Pangolin 0.2 QUIET)
find_package(OpenCV QUIET)

# flags
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O3 -g -std=c++0x -march=native"
#   "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
)

if (MSVC)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif (MSVC)


include_directories(
		${PROJECT_SOURCE_DIR}/src
		${PROJECT_SOURCE_DIR}/thirdparty/Sophus
		${PROJECT_SOURCE_DIR}/thirdparty/sse2neon
		${EIGEN3_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src/okvis_kinematics/include
    ${PROJECT_SOURCE_DIR}/src/okvis_util/include
)

include_directories( ${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR})

add_subdirectory(src/util)
add_subdirectory(src/FullSystem)
add_subdirectory(src/OptimizationBackend)
add_subdirectory(src/IOWrapper)
add_subdirectory(src/okvis_util)
add_subdirectory(src/okvis_kinematics)
add_dependencies(okvis_kinematics okvis_util)

#add_library(dso)
#target_link_libraries(dso util FullSystem OptimizationBackend IOWrapper)

#set_property( TARGET dso APPEND_STRING PROPERTY COMPILE_FLAGS -Wall )

# decide if we have ziplib.
if (LIBZIP_LIBRARY)
    message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
    add_definitions(-DHAS_ZIPLIB=1)
    include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} ) 
else()
    message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
    set(LIBZIP_LIBRARY "")
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # OSX
    set(BOOST_THREAD_LIBRARY boost_thread-mt)
else()
    set(BOOST_THREAD_LIBRARY boost_thread)
endif()

# build main executable (only if we have both OpenCV and Pangolin)
if (OpenCV_FOUND AND Pangolin_FOUND)
	message("--- compiling dso_dataset.")
	add_executable(convert_groundtruth_euroc ${PROJECT_SOURCE_DIR}/src/main_convert_groundtruth_euroc_dataset.cpp)
	target_link_libraries(convert_groundtruth_euroc)

	add_executable(convert_groundtruth_kitti ${PROJECT_SOURCE_DIR}/src/main_convert_groundtruth_kitti_dataset.cpp)
	target_link_libraries(convert_groundtruth_kitti)

	add_executable(dso_dataset ${PROJECT_SOURCE_DIR}/src/main_dso_pangolin.cpp )
    target_link_libraries(dso_dataset dso_util dso_FullSystem dso_IOWrapper dso_OptimizationBackend boost_system cxsparse ${BOOST_THREAD_LIBRARY} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OpenCV_LIBS})

	add_executable(dso_dataset_euroc ${PROJECT_SOURCE_DIR}/src/main_dso_pangolin_euroc.cpp )
	target_link_libraries(dso_dataset_euroc dso_util dso_FullSystem dso_IOWrapper dso_OptimizationBackend boost_system cxsparse ${BOOST_THREAD_LIBRARY} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OpenCV_LIBS})

	add_executable(euroc_rectify ${PROJECT_SOURCE_DIR}/src/main_rectify_euroc_dataset.cpp)
	target_link_libraries(euroc_rectify ${OpenCV_LIBS})
else()
	message("--- not building dso_dataset, since either don't have openCV or Pangolin.")
endif()

