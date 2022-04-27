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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -framework Accelerate -install_name forestryCpp.so")

set(SOURCE_FILES DataFrame.cpp DataFrame.h RFNode.cpp RFNode.h forestryTree.cpp forestryTree.h forestry.cpp forestry.h utils.cpp utils.h treeSplitting.cpp treeSplitting.h api.h api.cpp)

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
