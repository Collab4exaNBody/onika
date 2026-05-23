#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <onika/scg/operator_factory.h>
#include "yaml_conv.h"

namespace py = pybind11;
using Factory = onika::scg::OperatorNodeFactory;

void bind_factory(py::module_& m)
{
  m.def("available_operators", []() {
    auto ops = Factory::instance()->available_operators();
    return std::vector<std::string>(ops.begin(), ops.end());
  }, "Return a sorted list of all registered operator names.");

  // config may be a dict (map node) or a list (sequence node, e.g. for "simulation").
  m.def("make_operator",
    [](const std::string& name, py::object config) {
      try {
        return Factory::instance()->make_operator(name, py_to_yaml(config));
      } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
      }
    },
    py::arg("name"), py::arg("config") = py::dict(),
    "Create a named operator. config may be a dict or a list (for sequence-type operators).");

  // Equivalent to the top-level non-simulation YAML keys in an .msp file.
  // Call before building a simulation graph so named operator defaults are known.
  m.def("set_operator_defaults",
    [](py::dict defaults) {
      Factory::instance()->set_operator_defaults(py_to_yaml(defaults));
    },
    py::arg("defaults"),
    "Register named operator defaults (the top-level non-simulation keys in an .msp file).");
}
