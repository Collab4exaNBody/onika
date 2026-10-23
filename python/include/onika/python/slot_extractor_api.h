#pragma once

// Slot extractor API shared by pyonika and the Python extension modules of
// applications built on onika (pyexanbody, pyexastamp, ...).
//
// An extractor converts the value held by an operator slot of a given C++ type
// into a Python object; pyonika's slot_as_array() looks it up by type.  Modules
// register their extractors at import time:
//
//   PYBIND11_MODULE(_my_data, m)
//   {
//     onika::python::register_slot_extractor<MyType>( [](const MyType& v) -> pybind11::object { ... } );
//   }
//
// pyonika publishes the registration function in the capsule
// pyonika._slot_extractor_api, together with version numbers checked here, so
// that a module built against an incompatible pyonika fails at import with a
// clear error instead of crashing.

#include <pybind11/pybind11.h>
#include <onika/scg/operator_slot_base.h>
#include <onika/scg/operator_slot.h>

#include <functional>
#include <string>
#include <typeindex>

namespace onika
{
  namespace python
  {
    using SlotExtractorFn = std::function<pybind11::object(onika::scg::OperatorSlotBase&)>;
    using RegisterSlotExtractorFn = void(*)(std::type_index, SlotExtractorFn);

    static constexpr unsigned int SLOT_EXTRACTOR_API_VERSION = 1;
    static constexpr const char* SLOT_EXTRACTOR_API_CAPSULE = "pyonika.slot_extractor_api";

    struct SlotExtractorAPI
    {
      unsigned int api_version;
      unsigned int pybind11_internals_version; // std::function and py::object cross module boundaries
      RegisterSlotExtractorFn register_extractor;
    };

    // Fetch the registration function from the imported pyonika module.
    inline RegisterSlotExtractorFn slot_extractor_registry()
    {
      namespace py = pybind11;
      py::object cap_obj = py::module_::import("pyonika").attr("_slot_extractor_api");
      py::capsule cap = cap_obj.cast<py::capsule>();
      if( cap.name() == nullptr || std::string(cap.name()) != SLOT_EXTRACTOR_API_CAPSULE )
      {
        throw py::import_error("pyonika._slot_extractor_api has an unexpected capsule name");
      }
      const auto* api = cap.get_pointer<SlotExtractorAPI>();
      if( api->api_version != SLOT_EXTRACTOR_API_VERSION || api->pybind11_internals_version != PYBIND11_INTERNALS_VERSION )
      {
        throw py::import_error( "incompatible pyonika: slot extractor API v" + std::to_string(api->api_version)
                              + " / pybind11 internals v" + std::to_string(api->pybind11_internals_version)
                              + ", this module expects v" + std::to_string(SLOT_EXTRACTOR_API_VERSION)
                              + " / v" + std::to_string(PYBIND11_INTERNALS_VERSION) + ": rebuild it against the installed pyonika" );
      }
      return api->register_extractor;
    }

    // Register a converter for slots holding a T. The converter receives the
    // value; slots without a value yield None.
    template<class T, class Fn>
    inline void register_slot_extractor(Fn to_python)
    {
      slot_extractor_registry()( std::type_index(typeid(T)) , [to_python](onika::scg::OperatorSlotBase& slot) -> pybind11::object
      {
        auto* typed = static_cast<onika::scg::OperatorSlot<T>*>( &slot );
        if( ! typed->has_value() ) return pybind11::none();
        return to_python( **typed );
      } );
    }
  }
}
