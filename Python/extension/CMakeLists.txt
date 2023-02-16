cmake_minimum_required(VERSION 3.6)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -ftemplate-depth=1024")

project(forestry)

#### Armadillo

find_path(ARMADILLO_INCLUDE_DIR armadillo
        HINTS $ENV{ARMADILLO_DIR}
        PATH_SUFFIXES include
        REQUIRED
        )
message(STATUS "ARMADILLO_INCLUDE_DIR => ${ARMADILLO_INCLUDE_DIR}")

find_library(ARMADILLO_LIBRARY armadillo
        HINTS $ENV{ARMADILLO_DIR}
        PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
        REQUIRED
        )
message(STATUS "ARMADILLO_LIBRARY => ${ARMADILLO_LIBRARY}")

#### R

find_path(R_INCLUDE_DIR R.h
        HINTS "/usr/share/R" "/usr/lib64/R" $ENV{R_INCLUDE_DIR}
        PATH_SUFFIXES include
        REQUIRED
        )
message(STATUS "R_INCLUDE_DIR => ${R_INCLUDE_DIR}")

find_library(R_LIBRARY libR.so
        HINTS "/usr/lib64/R" $ENV{R_LIBRARY_DIR}
        PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib libs
        REQUIRED
        )
message(STATUS "R_LIBRARY => ${R_LIBRARY}")




message(STATUS "ARMADILLO_LIBRARY_DIR => ${Armadillo_LIBRARY_DIR}")
include_directories("include" ${Armadillo_LIBRARY_DIR} ${R_INCLUDE_DIR})

add_subdirectory(pybind11)
#pybind11_add_module(extension src/main.cpp include/forestry.cpp include/forestryTree.cpp include/DataFrame.cpp include/sampling.cpp)

###
add_library(extension MODULE src/main.cpp include/forestry.cpp include/forestryTree.cpp include/DataFrame.cpp include/treeSplitting.cpp include/RFNode.cpp include/sampling.cpp)
#add_library(extension MODULE src/main.cpp)

target_link_libraries(extension PRIVATE pybind11::module pybind11::lto pybind11::windows_extras PUBLIC ${ARMADILLO_LIBRARY} ${R_LIBRARY})

pybind11_extension(extension)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(extension)
endif()

set_target_properties(extension PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")

###

target_include_directories(extension PUBLIC ${ARMADILLO_INCLUDE_DIR} ${R_INCLUDE_DIR})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(extension PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})