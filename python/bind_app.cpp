#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <onika/app/api.h>
#include "yaml_conv.h"

namespace py = pybind11;

// Translate any std::exception that escapes an onika call into RuntimeError.
// OperatorCreationException is already registered as OnikaError in module.cpp
// and will be translated before this catch fires.
#define ONIKA_PY_TRY   try {
#define ONIKA_PY_CATCH } catch (const std::exception& e) { throw std::runtime_error(e.what()); }

void bind_app(py::module_& m)
{
  py::class_<onika::app::ApplicationContext,
             std::shared_ptr<onika::app::ApplicationContext>>(m, "ApplicationContext")
    .def_readonly("mpi_rank",   &onika::app::ApplicationContext::m_mpi_rank)
    .def_readonly("mpi_nprocs", &onika::app::ApplicationContext::m_mpi_nprocs)
    .def_readonly("ngpus",      &onika::app::ApplicationContext::m_ngpus)
    .def_readonly("cpucount",   &onika::app::ApplicationContext::m_cpucount)
    .def_property_readonly("error_code", &onika::app::ApplicationContext::get_error_code)
    .def("node", &onika::app::ApplicationContext::node,
         py::return_value_policy::reference, py::arg("path"))
    .def("set_multiple_run", &onika::app::ApplicationContext::set_multiple_run,
         py::arg("enabled"));

  m.def("init", [](std::vector<std::string> argv) {
    ONIKA_PY_TRY
      return onika::app::init(argv);
    ONIKA_PY_CATCH
  }, py::arg("argv"));

  m.def("run", [](std::shared_ptr<onika::app::ApplicationContext> ctx) {
    ONIKA_PY_TRY
      py::gil_scoped_release release;
      onika::app::run(ctx);
    ONIKA_PY_CATCH
  }, py::arg("ctx"));

  m.def("run_node", [](std::shared_ptr<onika::app::ApplicationContext> ctx,
                       onika::scg::OperatorNode* node) {
    ONIKA_PY_TRY
      py::gil_scoped_release release;
      onika::app::run(ctx, node);
    ONIKA_PY_CATCH
  }, py::arg("ctx"), py::arg("node"));

  m.def("end", [](std::shared_ptr<onika::app::ApplicationContext> ctx) {
    ONIKA_PY_TRY
      py::gil_scoped_release release;
      onika::app::end(ctx);
    ONIKA_PY_CATCH
  }, py::arg("ctx"));

  // Build a simulation graph from a Python list, equivalent to the simulation:
  // sequence in an .msp file. Calls post_graph_build() on every node.
  // Call set_operator_defaults() first if your list references named operators
  // that are not builtins (e.g. custom batch definitions).
  m.def("build_simulation_graph",
    [](std::shared_ptr<onika::app::ApplicationContext> ctx, py::object simulation) {
      ONIKA_PY_TRY
        return onika::app::build_simulation_graph(*ctx->m_configuration,
                                                   py_to_yaml(simulation));
      ONIKA_PY_CATCH
    },
    py::arg("ctx"), py::arg("simulation"),
    "Build a simulation graph from a Python list of operator specs.");
}
