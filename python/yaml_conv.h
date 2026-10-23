#pragma once
#include <pybind11/pybind11.h>
#include <yaml-cpp/yaml.h>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <string>

// Conversions between Python objects and YAML nodes.
//
// YAML -> Python follows the YAML 1.2 core schema for plain (unquoted)
// scalars: true/false -> bool, decimal integers -> int (any size), decimal
// floats and .inf/.nan -> float, null/~ -> None; anything else, and every
// quoted scalar, stays a str.  So "007", "yes", "1.0 ang" or "3.0e-3 ps"
// remain strings, and a dict read with get_operator_defaults() can be passed
// back to set_operator_defaults() unchanged.

// Python object (dict/list/scalar/None, numpy scalar or array) → YAML::Node
inline YAML::Node py_to_yaml(pybind11::handle obj)
{
  namespace py = pybind11;
  if (obj.is_none())
    return YAML::Node(YAML::NodeType::Null);
  // bool must be checked before int: in CPython bool is a subclass of int
  if (py::isinstance<py::bool_>(obj))
    return YAML::Node(obj.cast<bool>());
  if (py::isinstance<py::int_>(obj))
    return YAML::Node(py::str(obj).cast<std::string>()); // exact for any size
  if (py::isinstance<py::float_>(obj)) {
    double v = obj.cast<double>();
    std::ostringstream oss;
    oss << std::setprecision(17) << v;
    std::string s = oss.str();
    if (s == "nan" || s == "-nan") return YAML::Node(".nan");
    if (s == "inf") return YAML::Node(".inf");
    if (s == "-inf") return YAML::Node("-.inf");
    // Guarantee a decimal point so the value is parsed as a float, not an integer.
    if (s.find_first_of(".eE") == std::string::npos) s += ".0";
    return YAML::Node(s);
  }
  if (py::isinstance<py::str>(obj) || py::isinstance<py::bytes>(obj)) {
    // tagged like a quoted scalar: read back as a str by yaml_to_py ("007" stays "007");
    // onika's own conversions (as<int>, as<double>...) ignore the tag
    YAML::Node node(obj.cast<std::string>());
    node.SetTag("!");
    return node;
  }
  if (py::isinstance<py::dict>(obj)) {
    YAML::Node node(YAML::NodeType::Map);
    for (auto& kv : obj.cast<py::dict>())
      node[py::str(kv.first).cast<std::string>()] = py_to_yaml(kv.second);
    return node;
  }
  // numpy scalars (np.int64, np.float32, 0-d arrays...): convert to the Python equivalent
  if (py::hasattr(obj, "item") && py::hasattr(obj, "ndim") && obj.attr("ndim").cast<int>() == 0)
    return py_to_yaml(obj.attr("item")());
  // any other sequence (list, tuple, numpy array, ...)
  if (py::isinstance<py::sequence>(obj)) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (auto item : obj)
      node.push_back(py_to_yaml(item));
    return node;
  }
  // fallback: stringify whatever Python gave us
  return YAML::Node(py::str(obj).cast<std::string>());
}

// YAML::Node → Python object (dict/list/int/float/bool/str/None)
inline pybind11::object yaml_to_py(const YAML::Node& node)
{
  namespace py = pybind11;
  switch (node.Type()) {
    case YAML::NodeType::Map: {
      py::dict d;
      for (auto item : node)
        d[py::str(item.first.as<std::string>())] = yaml_to_py(item.second);
      return d;
    }
    case YAML::NodeType::Sequence: {
      py::list l;
      for (auto item : node)
        l.append(yaml_to_py(item));
      return l;
    }
    case YAML::NodeType::Scalar: {
      const std::string& s = node.Scalar();
      if (node.Tag() == "!") return py::str(s); // quoted scalar: always a string
      static const std::regex int_re(R"([-+]?(0|[1-9][0-9]*))");
      static const std::regex float_re(R"([-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?)");
      if (s == "true" || s == "True" || s == "TRUE") return py::bool_(true);
      if (s == "false" || s == "False" || s == "FALSE") return py::bool_(false);
      if (std::regex_match(s, int_re))
        return py::reinterpret_steal<py::object>(PyLong_FromString(s.c_str(), nullptr, 10));
      if (std::regex_match(s, float_re)) return py::float_(std::stod(s));
      if (s == ".inf" || s == "+.inf" || s == ".Inf" || s == ".INF") return py::float_(std::numeric_limits<double>::infinity());
      if (s == "-.inf" || s == "-.Inf" || s == "-.INF") return py::float_(-std::numeric_limits<double>::infinity());
      if (s == ".nan" || s == ".NaN" || s == ".NAN") return py::float_(std::numeric_limits<double>::quiet_NaN());
      return py::str(s);
    }
    case YAML::NodeType::Null:
    default:
      return py::none();
  }
}
