#pragma once
#include <pybind11/pybind11.h>
#include <onika/scg/operator_slot_base.h>
#include <onika/python/slot_extractor_api.h>
#include <typeindex>

// Returns a Python object converted from the slot's value if an extractor is
// registered for its type, or py::none() if the type is unknown or the value
// is not yet initialised.  Returned objects are copies.
pybind11::object slot_as_array(onika::scg::OperatorSlotBase& slot);

// Register an extractor for a C++ type (see onika/python/slot_extractor_api.h).
void register_slot_extractor(std::type_index ti, onika::python::SlotExtractorFn fn);

void bind_soatl(pybind11::module_& m);
