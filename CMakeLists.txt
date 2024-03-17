cmake_minimum_required(VERSION 3.11) # or whatever minimum version you require

include(FetchContent)

project(gemma_cpp_python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

FetchContent_Declare(sentencepiece GIT_REPOSITORY https://github.com/google/sentencepiece GIT_TAG 53de76561cfc149d3c01037f0595669ad32a5e7c)
FetchContent_MakeAvailable(sentencepiece)

FetchContent_Declare(gemma GIT_REPOSITORY https://github.com/google/gemma.cpp GIT_TAG 8fb44ed6dd123f63dca95c20c561e8ca1de511d7)
FetchContent_MakeAvailable(gemma)

FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG da250571a45826b21eebbddc1e50d0c1137dee5f)
FetchContent_MakeAvailable(highway)

FetchContent_Declare(pybind11 GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v2.10.4)
FetchContent_MakeAvailable(pybind11)

# Create the Python module
pybind11_add_module(pygemma src/gemma_binding.cpp)

target_link_libraries(pygemma PRIVATE libgemma hwy hwy_contrib sentencepiece)

# Link against libgemma.a and any other necessary libraries
FetchContent_GetProperties(gemma)
FetchContent_GetProperties(sentencepiece)
target_include_directories(pygemma PRIVATE ${gemma_SOURCE_DIR})
target_include_directories(pygemma PRIVATE ${sentencepiece_SOURCE_DIR})
target_compile_definitions(libgemma PRIVATE $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS NOMINMAX>)
target_compile_options(libgemma PRIVATE $<$<PLATFORM_ID:Windows>:-Wno-deprecated-declarations>)
