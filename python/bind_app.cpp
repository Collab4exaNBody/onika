#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <onika/app/api.h>
#include "yaml_conv.h"

namespace py = pybind11;

// Exceptions escaping onika calls are translated by pybind11 itself:
// OperatorCreationException -> pyonika.OnikaError (registered in module.cpp),
// any other std::exception -> RuntimeError (or the matching builtin, e.g. IndexError).

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
    return onika::app::init(argv);
  }, py::arg("argv"));

  m.def("run", [](std::shared_ptr<onika::app::ApplicationContext> ctx) {
    py::gil_scoped_release release;
    onika::app::run(ctx);
  }, py::arg("ctx"));

  m.def("run_node", [](std::shared_ptr<onika::app::ApplicationContext> ctx,
                       onika::scg::OperatorNode* node) {
    py::gil_scoped_release release;
    onika::app::run(ctx, node);
  }, py::arg("ctx"), py::arg("node"));

  m.def("end", [](std::shared_ptr<onika::app::ApplicationContext> ctx) {
    py::gil_scoped_release release;
    onika::app::end(ctx);
  }, py::arg("ctx"));

  // Build a simulation graph from a Python list, equivalent to the simulation:
  // sequence in an .msp file. Calls post_graph_build() on every node.
  // Call set_operator_defaults() first if your list references named operators
  // that are not builtins (e.g. custom batch definitions).
  //
  // If an item in the list is the string "simulation", it is resolved via
  // ctx->m_simulation_node (the value of the "simulation:" key in the .msp file,
  // e.g. "default_simulation").  That key is extracted by load_yaml_input and
  // therefore absent from the operator defaults, so the factory cannot look it
  // up by name without this pre-resolution step.
  m.def("build_simulation_graph",
    [](std::shared_ptr<onika::app::ApplicationContext> ctx, py::object simulation) {
      YAML::Node sim_yaml;
      if(simulation.is_none())
      {
        // No argument: use the simulation node stored during init(), identical
        // to what the internal init() → build_simulation_graph path does.
        sim_yaml = YAML::Clone(ctx->m_simulation_node);
      }
      else
      {
        sim_yaml = py_to_yaml(simulation);
        // Resolve "simulation" body items: the "simulation:" key is extracted by
        // load_yaml_input and removed from operator defaults, so the factory cannot
        // look it up by name.  Substitute it with ctx->m_simulation_node here.
        if( sim_yaml.IsSequence() && !ctx->m_simulation_node.IsNull() )
        {
          const YAML::Node& defaults = onika::scg::OperatorNodeFactory::instance()->get_operator_defaults();
          if( !defaults["simulation"] )
          {
            YAML::Node resolved(YAML::NodeType::Sequence);
            bool changed = false;
            for(const auto& item : sim_yaml)
            {
              if(item.IsScalar() && item.as<std::string>() == "simulation")
              {
                const YAML::Node& sim_node = ctx->m_simulation_node;
                if(sim_node.IsScalar())
                {
                  resolved.push_back(YAML::Clone(sim_node));
                }
                else if(sim_node.IsSequence())
                {
                  for(const auto& sub : sim_node)
                    resolved.push_back(YAML::Clone(sub));
                }
                changed = true;
              }
              else
              {
                resolved.push_back(YAML::Clone(item));
              }
            }
            if(changed) sim_yaml = resolved;
          }
        }
      }
      return onika::app::build_simulation_graph(*ctx->m_configuration, sim_yaml);
    },
    py::arg("ctx"), py::arg("simulation") = py::none(),
    "Build a simulation graph from a Python list of operator specs. "
    "If simulation is omitted, the simulation node from init() (ctx.m_simulation_node) is used.");
}
