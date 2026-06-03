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
# 1. WORKING
# ---------------------------------------------------------------------------
main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
ctx = pyonika.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

# "simulation" is an alias to "default_simulation" in main-config.msp.
# build_simulation_graph resolves it via ctx.m_simulation_node before passing to the factory.
graph = pyonika.build_simulation_graph(ctx, ["simulation"])
pyonika.run_node(ctx, graph)
pyonika.end(ctx)

# ---------------------------------------------------------------------------
# 4. WORKING
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
