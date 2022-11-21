cmake_minimum_required(VERSION 3.6)
project(forestryCpp)
find_package(Git REQUIRED)

FIND_PATH(ARMADILLO_INCLUDE_DIR armadillo
        HINTS ${ARMADILLO_DIR} $ENV{ARMADILLO_DIR}
        PATH_SUFFIXES include
        )

FIND_LIBRARY(ARMADILLO_LIBRARY NAMES armadillo
        HINTS ${ARMADILLO_DIR} $ENV{ARMADILLO_DIR}
        PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
        )

include_directories(${Armadillo_LIBRARY_DIR})

# Set up compiler and relevant files
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
# Add framework Accelerate flag
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -framework Accelerate -install_name forestryCpp.so")

include_directories(R-package/src)

set(SOURCE_FILES R-package/src/DataFrame.cpp R-package/src/DataFrame.h R-package/src/RFNode.cpp R-package/src/RFNode.h
R-package/src/forestryTree.cpp R-package/src/forestryTree.h R-package/src/forestry.cpp R-package/src/forestry.h R-package/src/utils.cpp R-package/src/utils.h R-package/src/treeSplitting.cpp R-package/src/treeSplitting.h R-package/src/api.h R-package/src/api.cpp)

# Build settings
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

ADD_LIBRARY(forestryCpp SHARED ${SOURCE_FILES})
TARGET_INCLUDE_DIRECTORIES(forestryCpp
        PUBLIC ${ARMADILLO_INCLUDE_DIR}
        )
TARGET_LINK_LIBRARIES(forestryCpp
        ${ARMADILLO_LIBRARY}
        )

add_custom_target(output_lib_name
        ALL
        COMMAND ${CMAKE_COMMAND} -E echo "forestryCpp location: $<TARGET_FILE:forestryCpp>"
        )
