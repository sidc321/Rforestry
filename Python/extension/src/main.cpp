#include "pybind11/pybind11.h"
#include "utils.cpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(extension, m) {
    m.doc() = R"pbdoc(
        RForestry Python extension module
        -----------------------

        .. currentmodule:: RForestr.extension

        .. autosummary::
           :toctree: _generate

           vector_get
    )pbdoc";

    m.def("vector_get", &vector_get, R"pbdoc(
        Some help text here

        Some other explanation about the get_vector function.
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

