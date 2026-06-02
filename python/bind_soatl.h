#pragma once
#include <pybind11/pybind11.h>
#include <onika/scg/operator_slot_base.h>
#include <functional>
#include <typeindex>

// Returns a numpy array view of the slot's value if its type is registered,
// or py::none() if the type is unknown or the value is not yet initialised.
pybind11::object slot_as_array(onika::scg::OperatorSlotBase& slot);

// Register a custom extractor for a C++ type.  The function receives the slot
// by reference and must return a Python object (or py::none() if the slot does
// not hold the expected type or has no value).  Callable from other pybind11
// extension modules (e.g. pyexanb_data) to add domain-specific extractors.
using SlotExtractorFn =
    std::function<pybind11::object(onika::scg::OperatorSlotBase&)>;
void register_slot_extractor(std::type_index ti, SlotExtractorFn fn);

void bind_soatl(pybind11::module_& m);
