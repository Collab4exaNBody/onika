#pragma once
#include <pybind11/pybind11.h>
#include <onika/scg/operator_slot_base.h>

// Returns a numpy array view of the slot's value if its type is registered,
// or py::none() if the type is unknown or the value is not yet initialised.
pybind11::object slot_as_array(onika::scg::OperatorSlotBase& slot);

void bind_soatl(pybind11::module_& m);
