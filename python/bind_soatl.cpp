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
// Registry: mangled type name (typeid(T).name()) → function that turns a slot
// into a Python object.  Keyed by the name string rather than std::type_index so
// that extractors registered from RTLD_LOCAL extension modules (e.g. _exanb_data)
// can match slots whose RTTI lives in a different DSO.  register_slot_extractor()
// takes a std::type_index for convenience and stores its name().  slot_as_array()
// does an O(1) lookup via OperatorSlotBase::value_type() (which is the same
// typeid name) and then calls the extractor with static_cast — no dynamic_cast.
// ---------------------------------------------------------------------------
static std::unordered_map<std::string, onika::python::SlotExtractorFn> g_extractors;

// Register a 1-D array extractor for std::vector<T>.
// The returned numpy array is a COPY of the vector's storage: pybind11 copies the
// buffer when no base object is given.  It stays valid after the graph is freed,
// and writing to it does not modify the slot.
template<typename T>
static void register_vector()
{
  using VecT = std::vector<T>;
  g_extractors[typeid(VecT).name()] = [](OSB& slot) -> py::object {
    auto* typed = static_cast<onika::scg::OperatorSlot<VecT>*>(&slot);
    if (!typed->has_value()) return py::none();
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
  g_extractors[typeid(VecT).name()] = [](OSB& slot) -> py::object {
    auto* typed = static_cast<onika::scg::OperatorSlot<VecT>*>(&slot);
    if (!typed->has_value()) return py::none();
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

// Public registration API — callable from other pybind11 extensions.
void register_slot_extractor(std::type_index ti, onika::python::SlotExtractorFn fn)
{
  g_extractors[ti.name()] = std::move(fn);
}

// Public entry-point used by OperatorNode.slot_as_array() and bind_scg.
py::object slot_as_array(OSB& slot)
{
  auto it = g_extractors.find(slot.value_type());
  if (it == g_extractors.end()) return py::none();
  return it->second(slot);
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
        "Return a numpy array copy of a slot's value, or None if the type is "
        "not registered or the value is not yet initialised.");

  // Expose register_slot_extractor to other pybind11 extensions (e.g.
  // pyexanbody._exanb_data) without RTLD_GLOBAL or linking against pyonika.
  // Use onika/python/slot_extractor_api.h to retrieve it: the capsule carries
  // version numbers checked at import time.
  static const onika::python::SlotExtractorAPI api {
    onika::python::SLOT_EXTRACTOR_API_VERSION, PYBIND11_INTERNALS_VERSION, &register_slot_extractor };
  m.attr("_slot_extractor_api") = py::capsule(&api, onika::python::SLOT_EXTRACTOR_API_CAPSULE);

  // Deprecated: unversioned raw function pointer, kept for modules built before
  // _slot_extractor_api existed.
  m.attr("_register_slot_extractor_fn") = py::capsule(
      reinterpret_cast<void*>(&register_slot_extractor),
      "_register_slot_extractor_fn");
}
