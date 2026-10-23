#include <pybind11/pybind11.h>
#include <cstdio>
#include <mpi.h>
#include <iostream>
#include <onika/log.h>
#include <onika/scg/operator_factory.h>
#include <onika/app/api.h>
#include "bind_scg.h"
#include "bind_factory.h"
#include "bind_app.h"
#include "bind_soatl.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m)
{
  m.doc() = "Python bindings for the onika HPC simulation framework";

  // Turn onika fatal errors into Python RuntimeError instead of std::abort().
  onika::FatalErrorLogStream::enable_python_mode(true);

  // Defer MPI_Finalize to process exit so that several init/end cycles can run.
  onika::app::set_deferred_mpi_finalize(true);

  // onika errors raise pyonika.OnikaError, a subclass of RuntimeError:
  // OperatorCreationException (unknown operator, incompatible slot types) and
  // fatal_error() (FatalError, thrown instead of aborting in Python mode).
  // intentionally leaked: must not be decref'ed after interpreter finalization
  static auto* onika_error = new py::exception<onika::scg::OperatorCreationException>(m, "OnikaError", PyExc_RuntimeError);
  py::register_exception_translator([](std::exception_ptr p) {
    try { if (p) std::rethrow_exception(p); }
    catch (const onika::scg::OperatorCreationException& e) { py::set_error(*onika_error, e.what()); }
    catch (const onika::FatalError& e) { py::set_error(*onika_error, e.what()); }
  });

  // MPI helpers for pyonika.mpi (MPI_COMM_WORLD is implementation specific, not reachable from ctypes)
  m.def("_mpi_size", []() {
    int init = 0, fin = 0, n = 1;
    MPI_Initialized(&init);
    MPI_Finalized(&fin);
    if (init && !fin) MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
  });
  m.def("_mpi_abort", [](int code) {
    onika::lout << std::flush;
    onika::lerr << std::flush;
    std::fflush(nullptr);
    MPI_Abort(MPI_COMM_WORLD, code);
  });

  // Flush C++ output streams, used by pyonika.capture_output().
  m.def("_flush_output", []() {
    onika::lout << std::flush;
    onika::lerr << std::flush;
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);
  });

  bind_scg(m);      // OperatorNode / OperatorSlotBase types
  bind_soatl(m);    // numpy buffer protocol for vector slots
  bind_factory(m);  // make_operator / available_operators
  bind_app(m);      // ApplicationContext, init/run/end
}
