#pragma once
#include <pybind11/pybind11.h>
#include <yaml-cpp/yaml.h>
#include <iomanip>
#include <sstream>
#include <string>

namespace py = pybind11;

// Python object (dict/list/scalar/None) → YAML::Node
inline YAML::Node py_to_yaml(py::handle obj)
{
  if (obj.is_none())
    return YAML::Node(YAML::NodeType::Null);
  // bool must be checked before int: in CPython bool is a subclass of int
  if (py::isinstance<py::bool_>(obj))
    return YAML::Node(obj.cast<bool>());
  if (py::isinstance<py::int_>(obj))
    return YAML::Node(obj.cast<long long>());
  if (py::isinstance<py::float_>(obj)) {
    double v = obj.cast<double>();
    std::ostringstream oss;
    oss << std::setprecision(17) << v;
    std::string s = oss.str();
    // Guarantee a decimal point so yaml-cpp parses this as float, not long.
    if (s.find('.') == std::string::npos &&
        s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos &&
        s.find('n') == std::string::npos &&   // nan
        s.find('i') == std::string::npos)     // inf
      s += ".0";
    return YAML::Node(s);
  }
  if (py::isinstance<py::str>(obj))
    return YAML::Node(obj.cast<std::string>());
  if (py::isinstance<py::dict>(obj)) {
    YAML::Node node(YAML::NodeType::Map);
    for (auto& kv : obj.cast<py::dict>())
      node[kv.first.cast<std::string>()] = py_to_yaml(kv.second);
    return node;
  }
  if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    YAML::Node node(YAML::NodeType::Sequence);
    for (auto item : obj)
      node.push_back(py_to_yaml(item));
    return node;
  }
  // fallback: stringify whatever Python gave us
  return YAML::Node(py::str(obj).cast<std::string>());
}

// YAML::Node → Python object (dict/list/int/float/bool/str/None)
inline py::object yaml_to_py(const YAML::Node& node)
{
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
      // bool literals (YAML 1.1 core schema)
      if (s=="true"||s=="True"||s=="TRUE"||s=="yes"||s=="on")  return py::bool_(true);
      if (s=="false"||s=="False"||s=="FALSE"||s=="no"||s=="off") return py::bool_(false);
      // integer
      try {
        size_t pos;
        long long v = std::stoll(s, &pos);
        if (pos == s.size()) return py::int_(v);
      } catch (...) {}
      // float
      try {
        size_t pos;
        double v = std::stod(s, &pos);
        if (pos == s.size()) return py::float_(v);
      } catch (...) {}
      return py::str(s);
    }
    case YAML::NodeType::Null:
      return py::none();
    default:
      return py::none();
  }
}
