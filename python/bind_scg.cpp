#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot_base.h>
#include <onika/type_utils.h>
#include "yaml_conv.h"
#include "bind_soatl.h"

namespace py = pybind11;
using OSB = onika::scg::OperatorSlotBase;
using ON  = onika::scg::OperatorNode;

// Lifetimes: slots and graph-internal nodes are not owned by Python. They are
// returned with reference_internal, so the Python object of their owner (the
// operator, or the node a traversal started from) stays alive as long as they
// are referenced.  Root nodes (ctx.graph, make_operator, build_simulation_graph)
// are held by shared_ptr.

static const char* direction_name(const OSB& s)
{
  return s.is_input_only() ? "in" : s.is_output_only() ? "out" : "inout";
}

// Slot value parsed as YAML (numbers, lists, maps...), or its string form if
// it is not valid YAML.  None when the slot has no value.  The default value of
// an unconnected input slot is built on first access: an input-only slot has an
// allocator only when its declaration provides a default.
static py::object slot_value(OSB& s)
{
  if (!s.has_value() && s.is_input_only() && s.input() == nullptr && s.resource() && s.resource()->has_allocator())
    s.initialize_resource_pointer();
  if (!s.has_value()) return py::none();
  const std::string str = s.value_as_string();
  try { return yaml_to_py(YAML::Load(str)); }
  catch (const YAML::Exception&) { return py::str(str); }
}

static py::dict describe_slot(OSB& s)
{
  py::dict d;
  d["name"]          = s.name();
  d["type"]          = onika::pretty_short_type(s.value_type());
  d["direction"]     = direction_name(s);
  d["required"]      = s.is_required();
  d["documentation"] = s.documentation();
  d["value"]         = slot_value(s);
  return d;
}

static py::list slot_list(py::object self, bool inputs)
{
  const ON& op = self.cast<const ON&>();
  py::list result;
  for (const auto& p : (inputs ? op.in_slots() : op.out_slots()))
    result.append(py::make_tuple(p.first, py::cast(p.second, py::return_value_policy::reference_internal, self)));
  return result;
}

void bind_scg(py::module_& m)
{
  py::class_<OSB>(m, "OperatorSlotBase")
    .def("name",          &OSB::name,          py::return_value_policy::copy)
    .def("pathname",      &OSB::pathname)
    .def("documentation", &OSB::documentation, py::return_value_policy::copy)
    .def("value_type",    &OSB::value_type,    py::return_value_policy::copy,
         "Mangled C++ type name (the slot extractor registry key); see type_name() for a readable one")
    .def("type_name",     [](const OSB& s) { return onika::pretty_short_type(s.value_type()); })
    .def("direction",     [](const OSB& s) { return direction_name(s); }, "'in', 'out' or 'inout'")
    .def("is_input",      &OSB::is_input)
    .def("is_output",     &OSB::is_output)
    .def("is_input_only", &OSB::is_input_only)
    .def("is_output_only",&OSB::is_output_only)
    .def("has_value",     &OSB::has_value)
    .def("is_required",   &OSB::is_required)
    .def("value",           &slot_value, "Value parsed as YAML (number, list, dict...), or None")
    .def("value_as_string", &OSB::value_as_string)
    .def("value_as_bool",   &OSB::value_as_bool)
    .def("describe",        &describe_slot, "dict with name, type, direction, required, documentation, value")
    .def("yaml_initialize", [](OSB& slot, py::object cfg) {
      slot.yaml_initialize(py_to_yaml(cfg));
    }, py::arg("config"))
    .def("__repr__", [](const OSB& s) {
      return "<Slot '" + s.pathname() + "' [" + direction_name(s) + "] " + onika::pretty_short_type(s.value_type()) + ">";
    });

  py::class_<ON, std::shared_ptr<ON>>(m, "OperatorNode")
    .def("name",           &ON::name)
    .def("pathname",       &ON::pathname)
    .def("documentation",  &ON::documentation)
    .def("in_slot_count",  &ON::in_slot_count)
    .def("out_slot_count", &ON::out_slot_count)
    .def("compiled",       &ON::compiled)
    .def("depth",          &ON::depth)
    .def("is_terminal",    &ON::is_terminal)

    // lists of (name, OperatorSlotBase); an in-out slot appears in both
    .def("in_slots",  [](py::object self) { return slot_list(self, true); })
    .def("out_slots", [](py::object self) { return slot_list(self, false); })

    .def("describe", [](ON& op) {
      py::dict d;
      d["name"]          = op.name();
      d["pathname"]      = op.pathname();
      d["documentation"] = op.documentation();
      py::list slots;
      for (const auto& p : op.named_slots()) slots.append(describe_slot(*p.second));
      d["slots"] = slots;
      return d;
    }, "dict with name, pathname, documentation and slots (see OperatorSlotBase.describe)")

    .def("yaml_initialize", [](ON& op, py::object cfg) {
      op.yaml_initialize(py_to_yaml(cfg));
    }, py::arg("config"))

    // {slot name: value} for slots holding a value; values are strings unless
    // typed=True, which parses them as YAML (see OperatorSlotBase.value)
    .def("slot_values", [](ON& op, bool typed) {
      py::dict result;
      for (const auto& p : op.named_slots()) {
        if (!p.second->has_value()) continue;
        result[py::str(p.first)] = typed ? slot_value(*p.second) : py::str(p.second->value_as_string());
      }
      return result;
    }, py::arg("typed") = false)

    // Finalise slot resource allocation. NOTE: make_operator() already calls
    // compile() internally via the factory — calling it again will throw
    // "OperatorNode cannot be re-compiled". Exposed for advanced use only.
    .def("compile", [](ON& op) { op.compile(); })

    // Python object converted from a named slot (see pyonika.slot_as_array)
    .def("slot_as_array", [](ON& op, const std::string& name) -> py::object {
      OSB* slot = op.in_slot(name);
      if (!slot) slot = op.out_slot(name);
      if (!slot) return py::none();
      return slot_as_array(*slot);
    }, py::arg("name"))

    // calls callback(node) for this node and every node below it
    .def("apply_graph", [](py::object self, py::object cb) {
      ON& root = self.cast<ON&>();
      root.apply_graph([&](ON* node) {
        if (node == &root) cb(self); // no keep-alive of a node on itself
        else cb(py::cast(node, py::return_value_policy::reference_internal, self));
      });
    }, py::arg("callback"))
    .def("__repr__", [](const ON& op) {
      return "<OperatorNode '" + op.pathname() + "'>";
    });
}
