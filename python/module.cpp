#include <pybind11/pybind11.h>
#include <onika/log.h>
#include <onika/scg/operator_factory.h>
#include "bind_scg.h"
#include "bind_factory.h"
#include "bind_app.h"
#include "bind_soatl.h"

namespace py = pybind11;

PYBIND11_MODULE(pyonika, m)
{
  m.doc() = "Python bindings for the onika HPC simulation framework";

  // Turn onika fatal errors into Python RuntimeError instead of std::abort().
  onika::FatalErrorLogStream::enable_python_mode(true);

  // Register OperatorCreationException as a proper Python exception class.
  py::register_exception<onika::scg::OperatorCreationException>(m, "OnikaError");

  bind_scg(m);      // OperatorNode / OperatorSlotBase types
  bind_soatl(m);    // numpy buffer protocol for vector slots
  bind_factory(m);  // make_operator / available_operators
  bind_app(m);      // ApplicationContext, init/run/end
}
