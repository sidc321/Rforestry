cmake_minimum_required(VERSION 3.4...3.18)
project(test)

include_directories(../../inc)
add_subdirectory(../../pybind11 ../pybind11)
pybind11_add_module(test src/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(test PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
