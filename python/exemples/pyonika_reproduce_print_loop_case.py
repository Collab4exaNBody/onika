#!/usr/bin/env python3
"""
Python reproduction of data/exemples/print_loop.msp using pyonika.

Requires the pyonika module installed (make install with -DONIKA_BUILD_PYTHON=ON)
and the environment sourced:

    source <install-prefix>/bin/setup-env.sh
    python exemples/print_loop.py
"""
import os
import sys
import pyonika

# ---------------------------------------------------------------------------
# 1. Initialise the framework (MPI, OpenMP, plugins, logging).
#    We use main-config.msp only to bootstrap — we will NOT run its graph.
# ---------------------------------------------------------------------------
main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
ctx = pyonika.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

# ---------------------------------------------------------------------------
# 2. Register named operator default definitions.
#    These are the top-level keys in print_loop.msp (everything except
#    "simulation" and "configuration").
# ---------------------------------------------------------------------------
pyonika.set_operator_defaults({

    # print_timestep: a batch that reads the current timestep and prints it.
    #   rebind renames the "value" slot to "timestep" so it reads from global.
    "print_timestep": {
        "rebind": {"value": "timestep"},
        "body": [
            {"print_int": {"prefix": "Current timestep = ", "suffix": "\n"}}
        ],
    },

    # compute_loop_stop: checks whether the loop should continue.
    #   rebind maps "end_at" → "nsteps" and "result" → "compute_loop_continue".
    "compute_loop_stop": {
        "rebind": {"end_at": "nsteps", "result": "compute_loop_continue"},
        "body": ["sim_continue"],
    },

    # compute_loop: the main time-stepping loop.
    "compute_loop": {
        "loop":      True,
        "condition": "compute_loop_continue",
        "body":      ["print_timestep", "next_time_step", "compute_loop_stop"],
    },

    # global: shared simulation state (dt, nsteps, timestep, loop flag).
    "global": {
        "dt":                   1.0,
        "nsteps":               20,
        "timestep":             0,
        "compute_loop_continue": True,
    },
})

# ---------------------------------------------------------------------------
# 3. Build the simulation graph from the "simulation:" sequence.
#    build_simulation_graph() wraps make_operator("simulation", ...) and
#    calls post_graph_build() on every node, which is required.
# ---------------------------------------------------------------------------
graph = pyonika.build_simulation_graph(ctx, [
    "global",
    "mpi_comm_world",
    "init_cuda",
    "global",
    {
        "unit_system": {
            "verbose": True,
            "unit_system": {
                "length":      "meter",
                "mass":        "kilogram",
                "time":        "second",
                "charge":      "coulomb",
                "temperature": "kelvin",
                "amount":      "mol",
                "luminosity":  "candela",
                "angle":       "radian",
                "energy":      "joule",
            },
        }
    },
    "compute_loop",
])

# ---------------------------------------------------------------------------
# 4. Run the new graph, then finalise.
# ---------------------------------------------------------------------------
pyonika.run_node(ctx, graph)
pyonika.end(ctx)
