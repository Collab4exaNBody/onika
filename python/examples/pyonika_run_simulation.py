#!/usr/bin/env python3
"""
Run the simulation defined by main-config.msp twice in the same process, to
check that several init/run/end cycles work:

1. with a graph built from ["simulation"], the "simulation:" entry of the
   input files (resolved by build_simulation_graph),
2. with the graph built from the default simulation node (no argument).

Usage:
    python examples/pyonika_run_simulation.py
"""
import os
import sys
import pyonika

# ---------------------------------------------------------------------------
# 1. graph built from ["simulation"]
# ---------------------------------------------------------------------------
main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
ctx = pyonika.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

# "simulation" is the "simulation:" entry of main-config.msp, which the factory does
# not know by name: build_simulation_graph substitutes it before building the graph.
graph = pyonika.build_simulation_graph(ctx, ["simulation"])
pyonika.run_node(ctx, graph)
pyonika.end(ctx)

# ---------------------------------------------------------------------------
# 2. graph built from the default simulation node
# ---------------------------------------------------------------------------
main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
ctx = pyonika.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

# No simulation argument: build_simulation_graph defaults to ctx.m_simulation_node,
# which is exactly the simulation node loaded from main-config.msp by init().
graph = pyonika.build_simulation_graph(ctx)
pyonika.run_node(ctx, graph)
pyonika.end(ctx)
