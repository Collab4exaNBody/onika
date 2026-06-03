#!/usr/bin/env python3
"""
Demonstrate how to read slot values from a running simulation graph.

Unlike make_operator() standalone nodes, operators embedded in a graph built
with build_simulation_graph() have their slots connected to global shared
storage.  After run_node(), has_value() returns True for populated slots and
slot_values() returns a non-empty dict.

NOTE: ctx.node(path) searches the graph built internally by init(), not the
graph returned by build_simulation_graph().  When using run_node(ctx, graph),
always access nodes through the `graph` variable, not ctx.node().
ctx.node() is only useful in Pattern A, where run(ctx) executes the init graph.

Two access patterns are shown:
  - apply_graph() + name match : find a specific operator in the graph
  - apply_graph() (full walk)  : collect all operators with populated slots

Usage:
    python pyonika_read_slot_values.py
"""

import os, sys
import pyonika

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")

# Bootstrap: load plugins and operator definitions.
ctx = pyonika.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

# Build a minimal compute loop that increments a timestep counter.
pyonika.set_operator_defaults({
    "global": {
        "dt": 1.0, "nsteps": 10, "timestep": 0, "compute_loop_continue": True,
    },
    "compute_loop": {
        "loop": True, "condition": "compute_loop_continue",
        "body": ["next_time_step", "compute_loop_stop"],
    },
    "compute_loop_stop": {
        "rebind": {"end_at": "nsteps", "result": "compute_loop_continue"},
        "body": ["sim_continue"],
    },
})

graph = pyonika.build_simulation_graph(ctx, ["global", "mpi_comm_world", "compute_loop"])
pyonika.run_node(ctx, graph)

# ── Pattern 1: find a specific operator by name in the custom graph ─────────
# Use apply_graph() to search the graph we actually ran.
# ctx.node() would search the init-built graph (never run here) — don't use it.
print("=== Pattern 1: find 'global' operator by name ===")
found = [None]
def find_global(node):
    if node.name() == "global" and found[0] is None:
        found[0] = node
graph.apply_graph(find_global)

global_op = found[0]
if global_op is not None:
    print(f"  pathname: {global_op.pathname()}")
    for name, slot in global_op.in_slots():
        if slot.has_value():
            print(f"  in  {name} = {slot.value_as_string()}")
    for name, slot in global_op.out_slots():
        if slot.has_value():
            print(f"  out {name} = {slot.value_as_string()}")
    # Expected: dt=1, nsteps=10, timestep=10 (after 10 iterations), ...
else:
    print("  'global' operator not found in graph")

# ── Pattern 2: apply_graph() — walk every node depth-first ─────────────────
print("\n=== Pattern 2: all operators with populated slots ===")
def print_slots(node):
    vals = node.slot_values()   # {} for nodes with no populated slots
    if vals:
        print(f"  {node.pathname()}: {vals}")

graph.apply_graph(print_slots)

pyonika.end(ctx)
