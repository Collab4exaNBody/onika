#!/usr/bin/env python3
"""
Demonstrate pyonika.make_operator(): discover operator slot names and types.

KEY POINT — make_operator() is for slot INTROSPECTION, not value reading.
In onika, input slots are backed by global shared storage that is only
allocated and connected when a full simulation graph is built and run.
Outside a graph, slots are declared and their C++ types are known, but
has_value() returns False and slot_values() returns {} for most operators.

make_operator() already compiles the node internally — do NOT call compile().
"""

import os, sys
import pyonika

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")

# init() populates the factory — make_operator() needs this.
ctx = pyonika.init([sys.argv[0], main_config])

# ── Inspect slot names and types on a unit_system operator ─────────────────
op = pyonika.make_operator("unit_system", {"verbose": True})

print(op)                              # <OperatorNode 'unit_system'>
print(f"compiled: {op.compiled()}")   # True — already done by make_operator

# has_value() is False in standalone context: slots exist but are not backed
# by global shared storage outside a full graph.  Use this loop to discover
# slot names and their C++ types (value_type() returns the mangled type name).
print("input slots:")
for name, slot in op.in_slots():
    val = slot.value_as_string() if slot.has_value() else "<unset>"
    print(f"  {name}: {slot.value_type()} = {val}")
print("output slots:")
for name, slot in op.out_slots():
    val = slot.value_as_string() if slot.has_value() else "<unset>"
    print(f"  {name}: {slot.value_type()} = {val}")
# Expected:
#   input slots:
#     unit_system: N5onika7physics10UnitSystemE = <unset>
#     verbose: b = <unset>
#   output slots:
#     (none, or operator-specific)

# slot_values() returns {} in standalone context for the same reason.
print(f"slot_values(): {op.slot_values()}")

# ── List available operators ────────────────────────────────────────────────
all_ops = pyonika.available_operators()
print(f"\n{len(all_ops)} operators registered")
print(f"'unit_system' present: {'unit_system' in all_ops}")

pyonika.end(ctx)
