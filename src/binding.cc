#include <pybind11/pybind11.h>

// #include "vendor/gemma.cpp/gemma.h"

namespace py = pybind11;

PYBIND11_MODULE(gemma_bindings, m)
{
    m.doc() = "Python bindings for the GEMMA library";

    // Expose classes and functions here, for example:
    // m.def("function_name", &function_name, "A function from the gemma library.");
}