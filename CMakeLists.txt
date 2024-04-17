cmake_minimum_required(VERSION 3.11) # or whatever minimum version you require

include(FetchContent)

project(gemma_cpp_python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

FetchContent_Declare(gemma GIT_REPOSITORY https://github.com/google/gemma.cpp GIT_TAG 7122afed5a89c082fac028ab152cc50af3e57386)
FetchContent_MakeAvailable(gemma)

FetchContent_Declare(pybind11 GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v2.10.4)
FetchContent_MakeAvailable(pybind11)

# Create the Python module
pybind11_add_module(_pygemma src/_pygemma/gemma_binding.cpp)

target_link_libraries(_pygemma PRIVATE libgemma)

FetchContent_GetProperties(gemma)
target_include_directories(_pygemma PRIVATE ${gemma_SOURCE_DIR})
