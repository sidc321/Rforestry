cmake_minimum_required(VERSION 3.6)
project(forestryCpp)



FIND_PATH(ARMADILLO_INCLUDE_DIR armadillo
        HINTS ${ARMADILLO_DIR} $ENV{ARMADILLO_DIR}
        PATH_SUFFIXES include
        )

FIND_LIBRARY(ARMADILLO_LIBRARY NAMES armadillo
        HINTS ${ARMADILLO_DIR} $ENV{ARMADILLO_DIR}
        PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
        )

include_directories(${Armadillo_LIBRARY_DIR})
include_directories(include)

# Set up compiler and relevant files
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_COMPILER clang++)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
# Add framework Accelerate flag
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -framework Accelerate -install_name forestryCpp.so")

add_subdirectory(pybind11)
pybind11_add_module(extension src/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(extension PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})