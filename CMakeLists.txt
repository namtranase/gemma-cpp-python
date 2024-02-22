# Previous content...

# Fetch pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.6.1 # Use the latest version or the version you prefer
)
FetchContent_MakeAvailable(pybind11)

# Pybind11 module
pybind11_add_module(gemma_bindings bindings.cc)
target_link_libraries(gemma_bindings PRIVATE libgemma)