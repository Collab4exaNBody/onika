#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <onika/scg/operator_slot_base.h>
#include <onika/scg/operator_slot.h>
#include "bind_soatl.h"

#include <functional>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using OSB = onika::scg::OperatorSlotBase;

// ---------------------------------------------------------------------------
// Registry: C++ type_index → function that turns a slot into a numpy array.
// ---------------------------------------------------------------------------
using SlotToArray = std::function<py::object(OSB&)>;
static std::unordered_map<std::type_index, SlotToArray> g_extractors;

// Register a 1-D array extractor for std::vector<T>.
// The returned numpy array is a non-owning VIEW of the vector's storage;
// keep the owning ApplicationContext alive for as long as the array is used.
template<typename T>
static void register_vector()
{
  using VecT = std::vector<T>;
  g_extractors[std::type_index(typeid(VecT))] = [](OSB& slot) -> py::object {
    auto* typed = dynamic_cast<onika::scg::OperatorSlot<VecT>*>(&slot);
    if (!typed || !typed->has_value()) return py::none();
    VecT& vec = **typed;
    if (vec.empty()) return py::array_t<T>(0);
    return py::array_t<T>(
      py::buffer_info(
        vec.data(),
        sizeof(T),
        py::format_descriptor<T>::format(),
        1,
        { static_cast<py::ssize_t>(vec.size()) },
        { static_cast<py::ssize_t>(sizeof(T))  }
      )
    );
  };
}

// Register a 2-D array extractor for std::vector<std::array<T,N>>.
template<typename T, std::size_t N>
static void register_vector_of_array()
{
  using ElemT = std::array<T, N>;
  using VecT  = std::vector<ElemT>;
  g_extractors[std::type_index(typeid(VecT))] = [](OSB& slot) -> py::object {
    auto* typed = dynamic_cast<onika::scg::OperatorSlot<VecT>*>(&slot);
    if (!typed || !typed->has_value()) return py::none();
    VecT& vec = **typed;
    if (vec.empty()) return py::array_t<T>(std::vector<py::ssize_t>{0, (py::ssize_t)N});
    return py::array_t<T>(
      py::buffer_info(
        vec.data()->data(),
        sizeof(T),
        py::format_descriptor<T>::format(),
        2,
        { static_cast<py::ssize_t>(vec.size()), static_cast<py::ssize_t>(N) },
        { static_cast<py::ssize_t>(sizeof(T) * N), static_cast<py::ssize_t>(sizeof(T)) }
      )
    );
  };
}

// Public entry-point used by OperatorNode.slot_as_array() and bind_scg.
py::object slot_as_array(OSB& slot)
{
  // Try each registered extractor until one succeeds (dynamic_cast determines
  // the match). The registry is small so a linear scan is fine.
  for (auto& [ti, fn] : g_extractors) {
    (void)ti;
    py::object result = fn(slot);
    if (!result.is_none()) return result;
  }
  return py::none();
}

void bind_soatl(py::module_& m)
{
  // Pre-register common 1-D vector types.
  register_vector<double>();
  register_vector<float>();
  register_vector<int>();
  register_vector<long>();
  register_vector<unsigned int>();
  register_vector<unsigned long>();
  register_vector<std::size_t>();
  register_vector<std::int8_t>();
  register_vector<std::int16_t>();
  register_vector<std::int32_t>();
  register_vector<std::int64_t>();
  register_vector<std::uint8_t>();
  register_vector<std::uint16_t>();
  register_vector<std::uint32_t>();
  register_vector<std::uint64_t>();

  // Common 2-D vector types (e.g. particle positions as Vec3).
  register_vector_of_array<double, 3>();
  register_vector_of_array<float,  3>();
  register_vector_of_array<double, 4>();
  register_vector_of_array<float,  4>();

  // slot_as_array(slot) — module-level helper.
  m.def("slot_as_array", [](OSB& slot) { return slot_as_array(slot); },
        py::arg("slot"),
        "Return a numpy array view of a slot's value, or None if the type is "
        "not registered or the value is not yet initialised.");
}
