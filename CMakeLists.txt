cmake_minimum_required(VERSION 3.10)
project(lis_nv)

if(NOT CMAKE_GENERATOR)
  message(STATUS "Suggesting to use Ninja for faster builds.")
  set(CMAKE_GENERATOR "Ninja" CACHE STRING "Use Ninja for building" FORCE)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default: Release)" FORCE)
endif()

# Dependencies
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
  spinnaker_camera_driver
  message_runtime
)
find_package(OpenCV 4 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBZSTD REQUIRED libzstd)
find_package(dv-processing REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs message_runtime
)

# Paths
set(SPINNAKER_INCLUDE_DIR "/opt/spinnaker/include")
set(SPINNAKER_LIB_DIR "/opt/spinnaker/lib")

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${LIBZSTD_INCLUDE_DIRS}
  ${dv-processing_INCLUDE_DIRS}
  ${SPINNAKER_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/lis_nv
)

# Common source file
set(DEFS_SRC include/lis_nv/defs.cpp)

# Node: vision_node
add_executable(vision_node
  src/vision_node.cpp
  src/visionNode.cpp
  ${DEFS_SRC}
)

target_link_libraries(vision_node
  ${catkin_LIBRARIES}
  dv::processing
  ${OpenCV_LIBS}
  ${LIBZSTD_LIBRARIES}
  ${SPINNAKER_LIB_DIR}/libSpinnaker.so
  stdc++fs
)

# Node: vision_node_frames
add_executable(vision_node_frames
  src/vision_node_frames.cpp
  src/visionNodeFrames.cpp
  ${DEFS_SRC}
)

target_link_libraries(vision_node_frames
  ${catkin_LIBRARIES}
  dv::processing
  ${OpenCV_LIBS}
  ${LIBZSTD_LIBRARIES}
  ${SPINNAKER_LIB_DIR}/libSpinnaker.so
  stdc++fs
)

#ARM optimizations, change in case you use a different architecture
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm*")
  foreach(target vision_node vision_node_frames)
    target_compile_options(${target} PRIVATE
      -O3 -march=native -mtune=native -funroll-loops -fomit-frame-pointer
      -flto -ftree-vectorize -fgraphite-identity -floop-nest-optimize
      -fstack-protector-strong -frename-registers -Ofast
    )
    target_link_libraries(${target} -flto)
  endforeach()
endif()

# Configs
install(FILES config/config.yaml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config)

# Python scripts (if they exist)
catkin_install_python(PROGRAMS
  scripts/mavros.py
  scripts/time.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
