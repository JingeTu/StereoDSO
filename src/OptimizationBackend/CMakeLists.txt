SET(PROJECT_NAME dso_OptimizationBackend)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(BUILD_TYPE Release)

add_library(${PROJECT_NAME}
        EnergyFunctional.cpp
        AccumulatedTopHessian.cpp
        AccumulatedSCHessian.cpp
        EnergyFunctionalStructs.cpp)

target_link_libraries(${PROJECT_NAME} ${GLOG_LIBRARY})
