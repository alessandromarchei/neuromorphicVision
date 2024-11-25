cmake_minimum_required(VERSION 3.10)
project(lis_nv)

# Suggest using Ninja for faster builds if not specified
if(NOT CMAKE_GENERATOR)
  message(STATUS "Suggesting to use Ninja for faster builds.")
  set(CMAKE_GENERATOR "Ninja" CACHE STRING "Use Ninja for building" FORCE)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build type for optimized performance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default: Release)" FORCE)
endif()

# Find necessary packages
find_package(dv-processing REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(yaml-cpp REQUIRED)

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  message_runtime
)

# Use pkg-config to find and include libzstd
pkg_check_modules(LIBZSTD REQUIRED libzstd)

# Catkin package configuration
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs message_runtime
)

include_directories(${PROJECT_SOURCE_DIR}/include/lis_nv)

# Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${LIBZSTD_INCLUDE_DIRS}
  ${dv-processing_INCLUDE_DIRS}
)

# Define sources for each node
set(VISION_SOURCE
    src/vision_node.cpp
    src/visionNode.cpp
    include/lis_nv/defs.cpp
)

add_executable(vision_node ${VISION_SOURCE})

# Link libraries for each node
target_link_libraries(vision_node
  ${catkin_LIBRARIES}
  dv::processing
  ${OpenCV_LIBS}
  ${LIBZSTD_LIBRARIES}
)

# Add dependencies for each node
add_dependencies(vision_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Install configuration file
install(FILES config/config.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

# Enable the fastest optimization for ARM architecture
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm*")
    target_compile_options(vision_node PRIVATE
      -O3
      -march=native
      -mtune=native
      #-ffast-math
      -funroll-loops
      -fomit-frame-pointer
      -flto
      -ftree-vectorize
      -fgraphite-identity
      -floop-nest-optimize
      -fstack-protector-strong
      #-funsafe-math-optimizations
      #-fassociative-math
      #-freciprocal-math
      #-fno-trapping-math
      -frename-registers
      -Ofast
    )
    # Link-time optimizations
    target_link_libraries(vision_node -flto)
endif()
#
# PYTHON SCRIPTS
catkin_install_python(PROGRAMS
  scripts/controller.py
  scripts/mavros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
