cmake_minimum_required(VERSION 3.11)  # or whatever minimum version you require

include(FetchContent)

project(gemma_cpp_python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

FetchContent_Declare(sentencepiece GIT_REPOSITORY https://github.com/google/sentencepiece GIT_TAG 53de76561cfc149d3c01037f0595669ad32a5e7c)
FetchContent_MakeAvailable(sentencepiece)

FetchContent_Declare(gemma GIT_REPOSITORY https://github.com/google/gemma.cpp GIT_TAG origin/main)
FetchContent_MakeAvailable(gemma)


FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG da250571a45826b21eebbddc1e50d0c1137dee5f)
FetchContent_MakeAvailable(highway)

add_subdirectory(vendor/pybind11)
#add_subdirectory(vendor/gemma.cpp)

# Create the Python module
pybind11_add_module(pygemma src/gemma_binding.cpp)

target_link_libraries(pygemma PRIVATE libgemma hwy hwy_contrib sentencepiece)

# Link against libgemma.a and any other necessary libraries
# After linking, try to include the directories. Adjust this if necessary.
FetchContent_GetProperties(gemma)
FetchContent_GetProperties(sentencepiece)
target_include_directories(pygemma PRIVATE ${gemma_SOURCE_DIR})
target_include_directories(pygemma PRIVATE ${sentencepiece_SOURCE_DIR})
