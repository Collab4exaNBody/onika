#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot_base.h>
#include "yaml_conv.h"
#include "bind_soatl.h"

namespace py = pybind11;
using OSB = onika::scg::OperatorSlotBase;
using ON  = onika::scg::OperatorNode;

void bind_scg(py::module_& m)
{
  // ------------------------------------------------------------------
  // OperatorSlotBase — non-owning view; lifetime tied to the operator.
  // ------------------------------------------------------------------
  py::class_<OSB>(m, "OperatorSlotBase")
    .def("name",          &OSB::name,          py::return_value_policy::reference)
    .def("pathname",      &OSB::pathname)
    .def("documentation", &OSB::documentation, py::return_value_policy::reference)
    .def("value_type",    &OSB::value_type,    py::return_value_policy::reference)
    .def("is_input",      &OSB::is_input)
    .def("is_output",     &OSB::is_output)
    .def("is_input_only", &OSB::is_input_only)
    .def("is_output_only",&OSB::is_output_only)
    .def("has_value",     &OSB::has_value)
    .def("is_required",   &OSB::is_required)
    .def("value_as_string", &OSB::value_as_string)
    .def("value_as_bool",   &OSB::value_as_bool)
    .def("yaml_initialize", [](OSB& slot, py::object cfg) {
      try { slot.yaml_initialize(py_to_yaml(cfg)); }
      catch (const std::exception& e) { throw std::runtime_error(e.what()); }
    }, py::arg("config"))
    .def("__repr__", [](const OSB& s) {
      std::string dir = s.is_input_only() ? "IN" : s.is_output_only() ? "OUT" : "IN_OUT";
      return "<Slot '" + s.pathname() + "' [" + dir + "] " + s.value_type() + ">";
    });

  // ------------------------------------------------------------------
  // OperatorNode — held by shared_ptr so factory-created nodes are
  // owned by Python; graph-internal nodes are returned as references.
  // ------------------------------------------------------------------
  py::class_<ON, std::shared_ptr<ON>>(m, "OperatorNode")
    .def("name",           &ON::name)
    .def("pathname",       &ON::pathname)
    .def("in_slot_count",  &ON::in_slot_count)
    .def("out_slot_count", &ON::out_slot_count)
    .def("compiled",       &ON::compiled)
    .def("depth",          &ON::depth)

    // Slot access — returns list of (name, OperatorSlotBase) tuples
    .def("in_slots", [](const ON& op) {
      py::list result;
      for (const auto& p : op.in_slots())
        result.append(py::make_tuple(p.first,
          py::cast(p.second, py::return_value_policy::reference)));
      return result;
    })
    .def("out_slots", [](const ON& op) {
      py::list result;
      for (const auto& p : op.out_slots())
        result.append(py::make_tuple(p.first,
          py::cast(p.second, py::return_value_policy::reference)));
      return result;
    })

    // Phase 2: configure an operator's slots from a Python dict
    .def("yaml_initialize", [](ON& op, py::object cfg) {
      try { op.yaml_initialize(py_to_yaml(cfg)); }
      catch (const std::exception& e) { throw std::runtime_error(e.what()); }
    }, py::arg("config"))

    // Read back the full slot config as a Python dict
    .def("slot_values", [](ON& op) {
      py::dict result;
      for (const auto& p : op.in_slots()) {
        if (p.second->has_value())
          result[py::str(p.first)] = py::str(p.second->value_as_string());
      }
      for (const auto& p : op.out_slots()) {
        if (p.second->has_value())
          result[py::str(p.first)] = py::str(p.second->value_as_string());
      }
      return result;
    })

    // Finalise slot resource allocation (needed for standalone make_operator nodes).
    .def("compile", [](ON& op) {
      try { op.compile(); }
      catch (const std::exception& e) { throw std::runtime_error(e.what()); }
    })

    // Return a numpy array view of a named slot (works for std::vector<T> types).
    .def("slot_as_array", [](ON& op, const std::string& name) -> py::object {
      OSB* slot = op.in_slot(name);
      if (!slot) slot = op.out_slot(name);
      if (!slot) return py::none();
      return slot_as_array(*slot);
    }, py::arg("name"))

    .def("apply_graph", [](ON& op, py::object cb) {
      op.apply_graph([&cb](ON* node) {
        cb(py::cast(node, py::return_value_policy::reference));
      });
    }, py::arg("callback"))
    .def("__repr__", [](const ON& op) {
      return "<OperatorNode '" + op.pathname() + "'>";
    });
}
