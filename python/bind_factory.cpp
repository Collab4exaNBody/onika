#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <onika/scg/operator_factory.h>
#include <onika/yaml/yaml_utils.h>
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

  // Return the current operator defaults as a Python dict (for inspection or
  // manual merging).  The defaults are set by init() from the loaded .msp files
  // and may subsequently be extended via set_operator_defaults().
  m.def("get_operator_defaults",
    []() -> py::object {
      return yaml_to_py(Factory::instance()->get_operator_defaults());
    },
    "Return the current operator defaults as a Python dict.");

  // Equivalent to the top-level non-simulation YAML keys in an .msp file.
  // Merges the given dict INTO the existing defaults (loaded by init() from the
  // .msp files) rather than replacing them, so that batch definitions such as
  // default_simulation remain available after the call.
  m.def("set_operator_defaults",
    [](py::dict defaults) {
      auto* factory = Factory::instance();
      YAML::Node merged = onika::yaml::merge_nodes(
          YAML::Clone(factory->get_operator_defaults()),
          py_to_yaml(defaults));
      factory->set_operator_defaults(merged);
    },
    py::arg("defaults"),
    "Merge named operator defaults into the existing defaults (does not erase "
    "definitions already loaded by init() from .msp files).");
}
