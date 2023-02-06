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

#### Rcpp

find_path(RCPP_INCLUDE_DIR Rcpp.h
        HINTS "/usr/local/lib/R/site-library/Rcpp" "/usr/lib/R/site-library/Rcpp" "/usr/lib64/R/library/Rcpp" $ENV{RCPP_DIR}
        PATH_SUFFIXES include
        REQUIRED
        )
message(STATUS "RCPP_INCLUDE_DIR => ${RCPP_INCLUDE_DIR}")

find_library(RCPP_LIBRARY Rcpp.so
        HINTS "/usr/local/lib/R/site-library/Rcpp" "/usr/lib/R/site-library/Rcpp" "/usr/lib64/R/library/Rcpp" $ENV{RCPP_DIR}
        PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib libs
        REQUIRED
        )
message(STATUS "RCPP_LIBRARY => ${RCPP_LIBRARY}")

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

#### RcppArmadillo

find_path(RCPPARMADILLO_INCLUDE_DIR RcppArmadillo.h
        HINTS "/usr/local/lib/R/site-library/RcppArmadillo" "/usr/lib/R/site-library/RcppArmadillo" "/usr/lib64/R/library/RcppArmadillo" $ENV{RCPPARMADILLO_DIR}
        PATH_SUFFIXES include
        REQUIRED
        )
message(STATUS "RCPPARMADILLO_INCLUDE_DIR => ${RCPPARMADILLO_INCLUDE_DIR}")

find_library(RCPPARMADILLO_LIBRARY RcppArmadillo.so
        HINTS "/usr/local/lib/R/site-library/RcppArmadillo" "/usr/lib/R/site-library/RcppArmadillo" "/usr/lib64/R/library/RcppArmadillo" $ENV{RCPPARMADILLO_DIR}
        PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib libs
        REQUIRED
        )
message(STATUS "RCPPARMADILLO_LIBRARY => ${RCPPARMADILLO_LIBRARY}")




message(STATUS "ARMADILLO_LIBRARY_DIR => ${Armadillo_LIBRARY_DIR}")
include_directories("include" ${Armadillo_LIBRARY_DIR} ${RCPP_INCLUDE_DIR} ${R_INCLUDE_DIR} ${RCPPARMADILLO_INCLUDE_DIR})

add_subdirectory(pybind11)
#pybind11_add_module(extension src/main.cpp include/forestry.cpp include/forestryTree.cpp include/DataFrame.cpp)

###
add_library(extension MODULE src/main.cpp include/forestry.cpp include/forestryTree.cpp include/DataFrame.cpp include/treeSplitting.cpp include/RFNode.cpp)
#add_library(extension MODULE src/main.cpp)

target_link_libraries(extension PRIVATE pybind11::module pybind11::lto pybind11::windows_extras PUBLIC ${ARMADILLO_LIBRARY} ${RCPP_LIBRARY} ${R_LIBRARY} ${RCPPARMADILLO_LIBRARY})

pybind11_extension(extension)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(extension)
endif()

set_target_properties(extension PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")

###

target_include_directories(extension PUBLIC ${ARMADILLO_INCLUDE_DIR} ${RCPP_INCLUDE_DIR} ${R_INCLUDE_DIR} ${RCPPARMADILLO_INCLUDE_DIR})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(extension PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})