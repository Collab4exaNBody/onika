# pyonika — Python bindings for the onika HPC simulation framework

`pyonika` is a pybind11 C++ extension module that exposes the onika operator
graph runtime to Python.  It lets you initialise, configure, inspect, and run
onika simulations entirely from Python, and read operator slot data as numpy
arrays.

---

## Table of contents

1. [Source layout](#source-layout)
2. [Build](#build)
3. [Quick start](#quick-start) — Pattern A (.msp-driven), B (full Python graph), C (patch .msp values)
4. [API reference](#api-reference)
   - [Module-level functions](#module-level-functions)
   - [ApplicationContext](#applicationcontext)
   - [OperatorNode](#operatornode)
   - [OperatorSlotBase](#operatorslotbase)
   - [Exceptions](#exceptions)
5. [.msp → Python mapping](#msp--python-mapping)
6. [Running multiple simulations](#running-multiple-simulations)
7. [numpy buffer protocol](#numpy-buffer-protocol)
8. [YAML / dict conversion](#yaml--dict-conversion)
9. [Python bindings — implementation](#python-bindings--implementation)
   - [module.cpp — entry point and fatal-error mode](#modulecpp--entry-point-and-fatal-error-mode)
   - [bind_app.cpp — ApplicationContext, init / run / end](#bind_appcpp--applicationcontext-init--run--end)
   - [bind_scg.cpp — OperatorNode and OperatorSlotBase](#bind_scgcpp--operatornode-and-operatorslotbase)
   - [bind_factory.cpp — operator factory](#bind_factorycpp--operator-factory)
   - [bind_soatl.cpp — numpy buffer registry](#bind_soatlcpp--numpy-buffer-registry)
   - [yaml_conv.h — Python ↔ YAML conversion](#yaml_convh--python--yaml-conversion)
   - [Ownership model](#ownership-model)
   - [GIL management](#gil-management)
   - [MPI multi-run support](#mpi-multi-run-support)
   - [Simulation alias resolution](#simulation-alias-resolution)

---

## Source layout

```
onika/
├── include/onika/
│   ├── log.h                ← modified: noexcept(false) dtor, enable_python_mode()
│   └── app/api.h            ← init / run / end / build_simulation_graph declarations
├── src/
│   ├── core/log.cpp         ← s_fatal_error_python_mode flag
│   └── core/api.cpp         ← MPI atexit fix for multi-run support
└── python/
    ├── CMakeLists.txt        ← pybind11 auto-fetch, pyonika target
    ├── module.cpp            ← PYBIND11_MODULE entry point
    ├── yaml_conv.h           ← py_to_yaml / yaml_to_py (header-only)
    ├── bind_scg.h / .cpp     ← OperatorNode, OperatorSlotBase bindings
    ├── bind_soatl.h / .cpp   ← numpy buffer registry and slot_as_array
    ├── bind_factory.h / .cpp ← make_operator, available_operators, set/get_operator_defaults
    ├── bind_app.h / .cpp     ← ApplicationContext, init / run / run_node / end / build_simulation_graph
    └── exemples/
        ├── pyonika_dryrun_test_import_pyonika.py      ← import smoke test
        ├── pyonika_run_main_config.py                 ← Pattern A: run + full graph/slot inspection
        ├── pyonika_execute_user_specified_msp_file.py ← Pattern A: interactive .msp launcher
        ├── pyonika_reproduce_print_loop_case.py       ← Pattern B: full Python graph, no .msp
        ├── pyonika_run_simulation.py                  ← multiple sequential init/run/end cycles
        ├── pyonika_make_operator_usage.py             ← make_operator: slot introspection
        └── pyonika_read_slot_values.py                ← reading slot values after run_node()
```

---

## Build

Enable `ONIKA_BUILD_PYTHON` when configuring onika.  pybind11 is fetched
automatically via `FetchContent` if not already installed (requires network
access at configure time):

```bash
cmake -B build-onika \
      -DONIKA_BUILD_PYTHON=ON \
      -DCMAKE_INSTALL_PREFIX=/path/to/onika_install \
      -DPython3_EXECUTABLE=$(which python3) \
      -S /path/to/onika_sources
cmake --build  build-onika -j8
cmake --install build-onika
```

After install, source the generated environment script:

```bash
source /path/to/onika_install/bin/setup-env.sh
```

This sets `PYTHONPATH` (so `import pyonika` works), `ONIKA_CONFIG_PATH`,
`ONIKA_PLUGIN_PATH`, and `ONIKA_DATA_PATH`.

**Install layout:**

```
<install-prefix>/
├── bin/
│   ├── setup-env.sh       ← source this before running Python scripts
│   └── onika-exec
├── lib/
│   └── pyonika.<platform>.so
├── plugins/
└── data/
```

---

## Quick start

```python
import os, sys
import pyonika

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")

# Pattern A — let the .msp file drive everything
ctx = pyonika.init([sys.argv[0], main_config])
if ctx.error_code >= 0:   # ≥0 means early exit (--help, unit tests, …)
    sys.exit(ctx.error_code)

# Inspect the operator graph before running
print("\nSimulation graph:\n")
root = ctx.node("simulation")
root.apply_graph(lambda op: print("\t",op.pathname()))

pyonika.run(ctx)
pyonika.end(ctx)
```

> **Examples:** `exemples/pyonika_dryrun_test_import_pyonika.py` (smoke test),
> `exemples/pyonika_run_main_config.py` (Pattern A + full graph/slot inspection),
> `exemples/pyonika_execute_user_specified_msp_file.py` (interactive .msp launcher).

```python
# Pattern B — build the graph entirely from Python
ctx = pyonika.init([sys.argv[0], main_config])   # bootstrap only

pyonika.set_operator_defaults({
    "print_timestep": {
        "rebind": {"value": "timestep"},
        "body": [
            {"print_int": {"prefix": "Current timestep = ", "suffix": "\n"}}
        ],
    },
    "compute_loop_stop": {
        "rebind": {"end_at": "nsteps", "result": "compute_loop_continue"},
        "body": ["sim_continue"],
    },
    "compute_loop": {
        "loop":      True,
        "condition": "compute_loop_continue",
        "body":      ["print_timestep", "next_time_step", "compute_loop_stop"],
    },
    "global": {
        "dt": 1.0, "nsteps": 20, "timestep": 0, "compute_loop_continue": True,
    },
})

graph = pyonika.build_simulation_graph(ctx, ["global", "mpi_comm_world", "compute_loop"])
pyonika.run_node(ctx, graph)
pyonika.end(ctx)
```

> **Example:** `exemples/pyonika_reproduce_print_loop_case.py` — full Python reproduction of
> `data/exemples/print_loop.msp` without reading any `.msp` file.

```python
# Pattern C — load an .msp file but patch some values before running
#
# init() builds the simulation graph internally from the .msp file, but does
# NOT run it yet.  set_operator_defaults() merges new values into the factory
# defaults (it does not replace the whole block — only the keys you supply are
# overridden).  You must then call build_simulation_graph() to produce a new
# graph that uses those updated defaults, and run it with run_node().
#
# IMPORTANT: run(ctx) would execute the graph that was built internally during
# init(), before set_operator_defaults() was called — your changes would have
# no effect.  Always use run_node(ctx, graph) after build_simulation_graph().

ctx = pyonika.init([sys.argv[0], main_config])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)

# Patch only the keys you want to change — the rest come from the .msp file.
pyonika.set_operator_defaults({
    "global": {
        "nsteps": 50,      # was e.g. 100 in the .msp
        "dt":     2.0,     # was 1.0
    },
})

# Rebuild the graph reusing the simulation: structure stored by init().
# build_simulation_graph(ctx) with no list argument is equivalent to
# build_simulation_graph(ctx, [<simulation-node from the .msp>]).
graph = pyonika.build_simulation_graph(ctx)
pyonika.run_node(ctx, graph)
pyonika.end(ctx)
```

---

## API reference

### Module-level functions

#### `pyonika.init(argv: list[str]) → ApplicationContext`

Initialise onika from a list of command-line arguments.  `argv[0]` is the
program name (used only in usage messages); `argv[1]` is the path to an `.msp`
configuration file.  Additional `--key value` pairs may follow to override
`configuration:` YAML keys at runtime.

```python
ctx = pyonika.init([sys.argv[0], "my_sim.msp"])
ctx = pyonika.init([sys.argv[0], "my_sim.msp", "--omp_num_threads", "4"])
```

**Command-line override format:** each override is two separate list elements
`"--key"` and `"value"` (space-separated, not `--key=value`).  The key is
automatically wrapped under `configuration:` and `-` (dash) in the key is used
as a YAML nesting separator.  So `--omp_num_threads 4` produces
`{ configuration: { omp_num_threads: 4 } }`, and `--omp-num-threads 4`
produces `{ configuration: { omp: { num: { threads: 4 } } } }`.

To override simulation parameters (e.g. `global`, `input_data`, …) use
`set_operator_defaults()` after `init()` as described in Pattern C above — those keys are not reachable via
the command-line override mechanism.

`init()` loads plugins, initialises MPI and OpenMP, parses the `.msp` file,
registers operator defaults, and builds the simulation graph internally (stored
in `ctx`).  It does **not** run the graph.

Raises `pyonika.OnikaError` or `RuntimeError` on failure.

---

#### `pyonika.run(ctx: ApplicationContext)`

Execute the simulation graph that was built internally by `init()`.  Releases
the Python GIL (`py::gil_scoped_release`) so OpenMP threads can run freely.
Use this when you called `init()` with a full `.msp` file and want to run its
graph unchanged.

---

#### `pyonika.run_node(ctx: ApplicationContext, node: OperatorNode)`

Execute an arbitrary graph rooted at `node`, using the profiling settings from
`ctx`.  Releases the GIL identically to `run`.  Use this when you built the
graph yourself with `build_simulation_graph()`, because that graph is not
stored inside `ctx`.

| Situation | Call |
|---|---|
| `init()` with a full `.msp` file | `run(ctx)` |
| Graph built with `build_simulation_graph()` | `run_node(ctx, graph)` |

---

#### `pyonika.end(ctx: ApplicationContext)`

Finalise the simulation: flush profiling traces and free all operator
resources.  MPI is **not** finalised here — it is deferred to an `atexit`
handler registered on the first `init()` call, so that multiple `init`/`end`
cycles in one Python process work correctly.

Must be called once after each `run` or `run_node`.

---

#### `pyonika.make_operator(name: str, config: dict = {}) → OperatorNode`

Instantiate a registered operator by name and optionally configure its input
slots from a Python dict (converted to YAML before being passed to
`yaml_initialize`).

`make_operator()` requires the factory to be populated, so `init()` must be
called first to load plugins.  The returned node is **already compiled** —
the factory calls `compile()` internally after `yaml_initialize`.  Do not call
`compile()` again; it will throw `RuntimeError: OperatorNode cannot be re-compiled`.

Raises `pyonika.OnikaError` if the operator name is not registered.

**What `make_operator` is for — slot introspection, not value reading:**
In onika's slot model, input slots are backed by global shared storage that is
allocated and connected when a full simulation graph is built and run.  Outside
a graph (`make_operator` standalone), the slots are declared and their C++ types
are known, but they are not connected to any storage — so `has_value()` returns
`False` and `slot_values()` returns `{}` for most operators.  The main use of
`make_operator` from Python is therefore to inspect **what slots an operator
has and what their types are**, not to read their values.

```python
import os, sys
import pyonika

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")

# init() populates the factory — make_operator() needs this.
ctx = pyonika.init([sys.argv[0], main_config])

# ── Inspect a unit_system operator ─────────────────────────────────────────
# make_operator() already compiles the node — do NOT call compile() after it.
op = pyonika.make_operator("unit_system", {"verbose": True})

print(op)                              # <OperatorNode 'unit_system'>
print(f"compiled: {op.compiled()}")   # True — already done by make_operator

# Iterate over all input slots to discover their names and C++ types.
# has_value() is False in standalone context (slots are not backed by global
# shared storage outside a full graph), so values appear as "<unset>".
for name, slot in op.in_slots():
    val = slot.value_as_string() if slot.has_value() else "<unset>"
    print(f"  in  {name}: {slot.value_type()} = {val}")
for name, slot in op.out_slots():
    val = slot.value_as_string() if slot.has_value() else "<unset>"
    print(f"  out {name}: {slot.value_type()} = {val}")
# Expected output:
#   in  unit_system: N5onika7physics10UnitSystemE = <unset>
#   in  verbose: b = <unset>
# slot_values() returns {} for the same reason.

# ── List all available operator names ──────────────────────────────────────
all_ops = pyonika.available_operators()
print(f"{len(all_ops)} operators registered")
print("unit_system" in all_ops)   # True after init()

pyonika.end(ctx)
```

To read actual slot values after a simulation has run, use `ctx.node()` or
`apply_graph()` on a graph built with `build_simulation_graph()` — in that
context, slots are connected to global shared storage and `has_value()` returns
`True` for populated slots.

```python
# Build and run a graph, then read slot values from within it.
ctx = pyonika.init([sys.argv[0], main_config])

pyonika.set_operator_defaults({
    "global": {"dt": 1.0, "nsteps": 10, "timestep": 0, "compute_loop_continue": True},
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

# NOTE: ctx.node() searches the graph built internally by init(), not the graph
# returned by build_simulation_graph().  When using run_node(ctx, graph), always
# look up nodes through the `graph` variable.
# ctx.node() is only useful in Pattern A, where run(ctx) executes the init graph.

# ── Find a specific operator by name via apply_graph() ─────────────────────
found = [None]
def find_global(node):
    if node.name() == "global" and found[0] is None:
        found[0] = node
graph.apply_graph(find_global)

global_op = found[0]
if global_op is not None:
    for name, slot in global_op.in_slots():
        if slot.has_value():
            print(f"  {name} = {slot.value_as_string()}")
# e.g.:  dt = 1   nsteps = 10   timestep = 10   compute_loop_continue = 0

# ── Walk every node and collect values ─────────────────────────────────────
def print_slots(node):
    vals = node.slot_values()   # {} for nodes with no populated slots
    if vals:
        print(f"{node.pathname()}: {vals}")

graph.apply_graph(print_slots)

pyonika.end(ctx)
```

> **Examples:** `exemples/pyonika_make_operator_usage.py` (slot introspection with `make_operator`),
> `exemples/pyonika_read_slot_values.py` (reading values via `apply_graph` after `run_node`).

---

#### `pyonika.available_operators() → list[str]`

Return a sorted list of all operator names registered in the factory
(builtins + loaded plugins).

---

#### `pyonika.get_operator_defaults() → dict`

Return the current operator defaults as a Python dict.  The defaults are
loaded by `init()` from the `.msp` file and may be extended via
`set_operator_defaults()`.  Useful for inspecting what named batch sequences
are available before building a graph.

---

#### `pyonika.set_operator_defaults(defaults: dict)`

Merge named operator default definitions into the existing defaults — equivalent
to the top-level non-`simulation`, non-`configuration` keys in an `.msp` file.
**Merges** into the current defaults rather than replacing them, so that batch
definitions loaded by `init()` (e.g. `default_simulation`) remain available
after the call.

```python
pyonika.set_operator_defaults({
    "my_loop": {"loop": True, "condition": "keep_going", "body": ["my_op"]},
    "global":  {"dt": 1.0, "nsteps": 100},
})
```

---

#### `pyonika.build_simulation_graph(ctx: ApplicationContext, simulation: list = None) → OperatorNode`

Build a simulation graph from a Python list of operator specs (the
`simulation:` sequence of an `.msp` file).  Calls `post_graph_build()` on
every node, which is required for slot resource allocation and graph
connection.

**`simulation` is optional.**  When omitted (or `None`), the simulation node
stored in `ctx` during `init()` is used directly — identical to what `init()`
does internally, and the simplest way to re-run the simulation defined in the
`.msp` file:

```python
ctx   = pyonika.init([sys.argv[0], main_config])
graph = pyonika.build_simulation_graph(ctx)         # uses simulation: from the .msp
pyonika.run_node(ctx, graph)
pyonika.end(ctx)
```

The string `"simulation"` in an explicit list is a special alias: if it is
absent from the current operator defaults (always the case after `init()` —
see [Simulation alias resolution](#simulation-alias-resolution)), it is
automatically resolved to `ctx.m_simulation_node` before the graph is built.

```python
graph = pyonika.build_simulation_graph(ctx, [
    "global", "mpi_comm_world",
    {"unit_system": {"verbose": True}},
    "my_loop",
])
pyonika.run_node(ctx, graph)
```

---

#### `pyonika.slot_as_array(slot: OperatorSlotBase) → numpy.ndarray | None`

Module-level variant of `OperatorNode.slot_as_array`.  Returns a numpy array
view of a slot's value, or `None` if the type is not registered or the value
is not yet initialised.  See [numpy buffer protocol](#numpy-buffer-protocol).

---

### ApplicationContext

Returned by `pyonika.init()`.  Holds the full simulation state: MPI
communicators, GPU/CPU counts, loaded configuration, the operator graph, and
profiling data.

| Attribute / method | Type | Description |
|---|---|---|
| `mpi_rank` | `int` | MPI rank of this process |
| `mpi_nprocs` | `int` | Total number of MPI processes |
| `ngpus` | `int` | Number of GPUs available |
| `cpucount` | `int` | Number of CPU cores |
| `error_code` | `int` | `-1` = normal run; `≥ 0` = early-exit code (pass to `sys.exit`) |
| `node(path)` | `OperatorNode` | Look up an operator by its dotted pathname, e.g. `"simulation.unit_system"` |
| `set_multiple_run(bool)` | — | Allow the simulation graph to be executed more than once |

`node()` returns a **non-owning reference** (`return_value_policy::reference`).
Keep `ctx` alive as long as the returned node is in use.

`ApplicationContext` is held by `std::shared_ptr` on both the C++ and Python
sides, so the object is not destroyed until all Python references to it (and to
any `OperatorNode` or `OperatorSlotBase` derived from it) are released.

---

### OperatorNode

Represents a node in the simulation computation graph (SCG).

| Method | Returns | Description |
|---|---|---|
| `name()` | `str` | Local operator name |
| `pathname()` | `str` | Fully qualified dotted path (e.g. `"simulation.compute_loop.my_op"`) |
| `depth()` | `int` | Depth in the graph tree |
| `in_slot_count()` | `int` | Number of input slots |
| `out_slot_count()` | `int` | Number of output slots |
| `in_slots()` | `list[(str, OperatorSlotBase)]` | All input slots as (name, slot) pairs |
| `out_slots()` | `list[(str, OperatorSlotBase)]` | All output slots as (name, slot) pairs |
| `compiled()` | `bool` | Whether slot resources have been allocated |
| `compile()` | — | Allocate slot resources — **not needed after `make_operator()`** (factory already calls it); exposed for advanced use only |
| `yaml_initialize(config: dict)` | — | Configure input/output slots from a Python dict |
| `slot_values()` | `dict[str, str]` | String representations of all initialised slot values |
| `slot_as_array(name: str)` | `ndarray \| None` | numpy view of a named slot (see below) |
| `apply_graph(callback)` | — | Depth-first traversal; calls `callback(node)` for each node |

```python
# Walk the full simulation graph
ctx.node("simulation").apply_graph(lambda op: print(op.pathname()))

# Inspect slots on a specific operator
for name, slot in ctx.node("simulation.unit_system").in_slots():
    print(f"  {name}: {slot.value_type()} = {slot.value_as_string()}")

# Look up a named slot and get its numpy view
arr = ctx.node("simulation.my_extractor").slot_as_array("data")
if arr is not None:
    print(arr.shape, arr.dtype, arr.mean())
```

`slot_as_array(name)` searches both input and output slots (input first).

**Repr:** `<OperatorNode 'simulation.unit_system'>`.

---

### OperatorSlotBase

Represents one input or output slot on an operator.

| Method | Returns | Description |
|---|---|---|
| `name()` | `str` | Slot name |
| `pathname()` | `str` | Fully qualified path |
| `value_type()` | `str` | C++ type name (`typeid(T).name()`, mangled) |
| `documentation()` | `str` | Doc string, if provided by the operator |
| `is_input()` | `bool` | True for `INPUT` or `INPUT_OUTPUT` |
| `is_output()` | `bool` | True for `OUTPUT` or `INPUT_OUTPUT` |
| `is_input_only()` | `bool` | True for `INPUT` only |
| `is_output_only()` | `bool` | True for `OUTPUT` only |
| `has_value()` | `bool` | True if the value has been materialised |
| `is_required()` | `bool` | True if the slot must have a value |
| `value_as_string()` | `str` | String representation of the current value |
| `value_as_bool()` | `bool` | Boolean interpretation of the current value |
| `yaml_initialize(config)` | — | Set the slot value from a Python object |

`OperatorSlotBase` instances are **non-owning references** — they are only
valid while the owning operator (and therefore its `ApplicationContext`) is
alive.

**Repr:** `<Slot 'simulation.my_op.data' [OUT] St6vectorIdSaIdEE>`.

---

### Exceptions

| Exception | Raised when |
|---|---|
| `pyonika.OnikaError` | Unknown operator name in `make_operator`, slot type incompatibility, or any `OperatorCreationException` from the factory |
| `RuntimeError` | Any other `std::exception` escaping an onika call (bad YAML, slot type mismatch, fatal error in Python mode) |

```python
try:
    op = pyonika.make_operator("nonexistent_operator")
except pyonika.OnikaError as e:
    print(f"Operator not found: {e}")
```

**Fatal errors in Python mode:** onika uses an internal `fatal_error()`
mechanism for unrecoverable errors (e.g. incompatible slot types during graph
compilation).  In normal C++ mode these call `std::abort()`.  The Python
module enables *Python mode* at load time (`enable_python_mode(true)`), which
makes `FatalErrorLogStream::~FatalErrorLogStream()` throw `std::runtime_error`
instead — caught by the `ONIKA_PY_CATCH` macro and re-raised as `RuntimeError`.

---

## .msp → Python mapping

| `.msp` construct | Python equivalent |
|---|---|
| Top-level key with a mapping | `set_operator_defaults({"key": {…}})` |
| Top-level key with a scalar | `set_operator_defaults({"key": "value"})` |
| `- operator_name` in simulation sequence | `"operator_name"` string in the list |
| `- op_name: {key: val}` | `{"op_name": {"key": val}}` dict in the list |
| `loop:` / `condition:` / `body:` / `rebind:` | Same keys in the Python dict |
| `simulation: default_simulation` | `build_simulation_graph(ctx, ["default_simulation"])` |

**Full example** — reproducing `print_loop.msp` in pure Python:

```python
import pyonika, sys, os

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
ctx = pyonika.init([sys.argv[0], main_config])   # bootstrap only

pyonika.set_operator_defaults({
    "print_timestep": {
        "rebind": {"value": "timestep"},
        "body": [{"print_int": {"prefix": "Current timestep = ", "suffix": "\n"}}],
    },
    "compute_loop_stop": {
        "rebind": {"end_at": "nsteps", "result": "compute_loop_continue"},
        "body": ["sim_continue"],
    },
    "compute_loop": {
        "loop": True, "condition": "compute_loop_continue",
        "body": ["print_timestep", "next_time_step", "compute_loop_stop"],
    },
    "global": {"dt": 1.0, "nsteps": 20, "timestep": 0, "compute_loop_continue": True},
})

graph = pyonika.build_simulation_graph(ctx, [
    "global", "mpi_comm_world", "init_cuda", "global",
    {"unit_system": {"verbose": True, "unit_system": {"length": "meter", "mass": "kilogram", "time": "second"}}},
    "compute_loop",
])

pyonika.run_node(ctx, graph)
pyonika.end(ctx)
```

> **Example:** `exemples/pyonika_reproduce_print_loop_case.py`.

---

## Running multiple simulations

Each `init()` call reloads YAML, re-reads plugins, and resets the operator
defaults.  Thanks to the MPI `atexit` fix, MPI stays alive across all cycles
and is finalised exactly once at process exit.

```python
import pyonika, sys, os

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")

for nsteps in [10, 50, 100]:
    ctx = pyonika.init([sys.argv[0], main_config])
    pyonika.set_operator_defaults({"global": {"nsteps": nsteps}})
    graph = pyonika.build_simulation_graph(ctx)
    pyonika.run_node(ctx, graph)
    pyonika.end(ctx)
```

> When using `pyexanbody`, MPI is pre-initialised at import time so this works
> out of the box.  When using `pyonika` directly, the first `init()` registers
> the `atexit` handler — subsequent calls are safe as long as you do not call
> `MPI_Finalize()` yourself between cycles.

> **Example:** `exemples/pyonika_run_simulation.py` — runs several named batch
> sequences in sequence within a single Python process.

---

## numpy buffer protocol

`slot_as_array()` returns a **non-owning numpy array** backed by the slot's
in-memory C++ storage.  No data is copied.

**Supported slot types (registered in `bind_soatl.cpp` at module load time):**

| C++ slot type | numpy dtype | shape |
|---|---|---|
| `std::vector<double>` | `float64` | `(N,)` |
| `std::vector<float>` | `float32` | `(N,)` |
| `std::vector<int>` | `int32` | `(N,)` |
| `std::vector<long>` | `int64` | `(N,)` |
| `std::vector<unsigned int>` | `uint32` | `(N,)` |
| `std::vector<unsigned long>` | `uint64` | `(N,)` |
| `std::vector<size_t>` | `uint64` | `(N,)` |
| `std::vector<int8_t>` | `int8` | `(N,)` |
| `std::vector<int16_t>` | `int16` | `(N,)` |
| `std::vector<int32_t>` | `int32` | `(N,)` |
| `std::vector<int64_t>` | `int64` | `(N,)` |
| `std::vector<uint8_t>` | `uint8` | `(N,)` |
| `std::vector<uint16_t>` | `uint16` | `(N,)` |
| `std::vector<uint32_t>` | `uint32` | `(N,)` |
| `std::vector<uint64_t>` | `uint64` | `(N,)` |
| `std::vector<std::array<double,3>>` | `float64` | `(N, 3)` |
| `std::vector<std::array<float,3>>` | `float32` | `(N, 3)` |
| `std::vector<std::array<double,4>>` | `float64` | `(N, 4)` |
| `std::vector<std::array<float,4>>` | `float32` | `(N, 4)` |

Returns `None` for unregistered types or uninitialised slots.

**Lifetime constraint:** the array's memory is owned by the C++ operator slot.
Keep the `ApplicationContext` alive for as long as you use the array.  Assigning
`arr = None` releases the Python reference but does not free the C++ memory.

**Standalone operators:** nodes created with `make_operator()` are already
compiled by the factory — `slot_as_array()` can be called on them immediately.

```python
op = pyonika.make_operator("my_op_with_vector_slot")
arr = op.slot_as_array("input1")   # → np.ndarray or None
```

**Extending the registry:** other pybind11 extension modules can register
additional slot types without linking against `pyonika.so` directly.  The
`_register_slot_extractor_fn` `PyCapsule` holds the raw function pointer:

```cpp
// In another extension module (e.g. _myext.cpp):
#include <pybind11/pybind11.h>
#include <functional>
#include <typeindex>

using SlotExtractorFn = std::function<py::object(onika::scg::OperatorSlotBase&)>;
using RegisterFn      = void(*)(std::type_index, SlotExtractorFn);

namespace py = pybind11;

PYBIND11_MODULE(_myext, m)
{
  py::object pyonika = py::module_::import("pyonika");
  auto capsule = pyonika.attr("_register_slot_extractor_fn");
  auto fn = reinterpret_cast<RegisterFn>(
      PyCapsule_GetPointer(capsule.ptr(), "_register_slot_extractor_fn"));

  fn(typeid(MySlotType), [](onika::scg::OperatorSlotBase& slot) -> py::object {
    auto* typed = static_cast<onika::scg::OperatorSlot<MySlotType>*>(&slot);
    if (!typed->has_value()) return py::none();
    // … build and return a Python object …
  });
}
```

---

## YAML / dict conversion

Operator slots are configured in `.msp` files using YAML.  `pyonika` converts
Python dicts to `YAML::Node` objects via `py_to_yaml()` before passing them to
`OperatorNode::yaml_initialize`, and converts back via `yaml_to_py()` when
reading defaults.

### `py_to_yaml()` — type mapping

| Python type | YAML node type | Notes |
|---|---|---|
| `None` | Null | |
| `bool` | scalar `true`/`false` | checked before `int` (bool ⊂ int in CPython) |
| `int` | scalar integer | serialised as `long long` |
| `float` | scalar string | formatted with 17 significant digits; `.0` appended when no decimal/exponent present (see below) |
| `str` | scalar string | |
| `dict` | Map | keys must be strings |
| `list` / `tuple` | Sequence | |
| anything else | scalar string | `str(obj)` fallback |

### `yaml_to_py()` — type mapping

| YAML scalar | Python type | Detection |
|---|---|---|
| `true`/`True`/`yes`/`on` | `bool` | string equality |
| `false`/`False`/`no`/`off` | `bool` | string equality |
| parseable integer | `int` | `std::stoll` consuming all chars |
| parseable float | `float` | `std::stod` consuming all chars |
| anything else | `str` | |
| Map | `dict` | |
| Sequence | `list` | |
| Null | `None` | |

### Float serialisation fix

`YAML::Node(double)` in yaml-cpp serialises whole-number floats (e.g. `1.0`)
as `"1"` with no decimal point.  yaml-cpp's reader then identifies `"1"` as a
`long`, which breaks slot type matching when the operator expects `double`.

`py_to_yaml()` works around this by formatting Python floats with
`std::setprecision(17)` and unconditionally appending `.0` when none of `.`,
`e`, `E`, `n` (nan), or `i` (inf) is present in the string.  This guarantees:

```
Python 1.0   →  YAML "1.0"   →  C++ double 1.0   ✓
Python 1e10  →  YAML "1e10"  →  C++ double 1e10  ✓  (already has 'e')
Python 1     →  YAML "1"     →  C++ long   1     ✓  (int, not float)
```

---

## Python bindings — implementation

### `module.cpp` — entry point and fatal-error mode

```cpp
PYBIND11_MODULE(pyonika, m)
{
  onika::FatalErrorLogStream::enable_python_mode(true);
  py::register_exception<onika::scg::OperatorCreationException>(m, "OnikaError");

  bind_scg(m);
  bind_soatl(m);
  bind_factory(m);
  bind_app(m);
}
```

Two things happen at module load time:

1. **Python mode** is activated.  `FatalErrorLogStream::~FatalErrorLogStream()`
   is marked `noexcept(false)` and, when Python mode is on, throws
   `std::runtime_error` instead of calling `std::abort()`.  This makes onika
   fatal errors recoverable from Python (they surface as `RuntimeError`).

2. **`OnikaError`** is registered as a proper Python exception class that maps
   to `onika::scg::OperatorCreationException`.  It is raised by `make_operator`
   for unknown operator names and by the factory for slot type incompatibilities.

All other `std::exception` instances that escape any onika call are caught by
the `ONIKA_PY_TRY` / `ONIKA_PY_CATCH` macros in each binding file and
re-raised as `RuntimeError`.

---

### `bind_app.cpp` — ApplicationContext, init / run / end

**`ApplicationContext`** is registered with `std::shared_ptr` as the holder
type so Python shares ownership with the C++ side.  All attributes are
read-only (`def_readonly` / `def_property_readonly`).

**`init`** wraps `onika::app::init(argv)` in a `ONIKA_PY_TRY`/`ONIKA_PY_CATCH`
block.  The returned `shared_ptr<ApplicationContext>` is immediately owned by
Python.

**`run` and `end`** both release the GIL before calling into C++:

```cpp
m.def("run", [](std::shared_ptr<ApplicationContext> ctx) {
  ONIKA_PY_TRY
    py::gil_scoped_release release;
    onika::app::run(ctx);
  ONIKA_PY_CATCH
}, py::arg("ctx"));
```

The GIL release allows OpenMP threads spawned by the simulation kernels to run
without being serialised by the Python interpreter lock.  It is reacquired
automatically when `run()` returns.

**`build_simulation_graph`** converts the Python list to a `YAML::Node` via
`py_to_yaml`, applies the simulation alias resolution (see below), and calls
`onika::app::build_simulation_graph`.  The `simulation` argument defaults to
`py::none()` — when `None`, `ctx->m_simulation_node` is used directly (a
`YAML::Clone` of the node stored during `init()`).

---

### `bind_scg.cpp` — OperatorNode and OperatorSlotBase

**`OperatorSlotBase`** is registered without a holder (raw pointer), with
`return_value_policy::reference` on all methods that return references to
strings.  `yaml_initialize` converts the Python dict to YAML via `py_to_yaml`
and calls the C++ method.

**`OperatorNode`** is registered with `std::shared_ptr<OperatorNode>` as the
holder type.  This is important: factory-created nodes (returned by
`make_operator`) are owned by the `shared_ptr` in Python, while graph-internal
nodes (returned by `ctx.node()` and the `apply_graph` callback) are returned
with `return_value_policy::reference` — Python holds a non-owning view.

**`in_slots` / `out_slots`** build a Python list of `(name, slot)` tuples.
Each slot is cast with `return_value_policy::reference` since `OperatorSlotBase`
instances are owned by their operator.

**`slot_as_array`** searches both input and output slots by name and delegates
to the `slot_as_array(OSB&)` free function from `bind_soatl.cpp`.

**`apply_graph`** wraps the callback in a C++ lambda.  Each visited node is
cast with `return_value_policy::reference` since the graph owns all nodes.

---

### `bind_factory.cpp` — operator factory

**`available_operators`** delegates to `OperatorNodeFactory::instance()->available_operators()`.

**`make_operator`** calls `OperatorNodeFactory::instance()->make_operator(name, yaml)`.
The returned `shared_ptr<OperatorNode>` is owned by Python.

**`set_operator_defaults`** *merges* the Python dict into the existing defaults
using `onika::yaml::merge_nodes`, rather than replacing them.  This ensures
that batch definitions loaded by `init()` from `.msp` files (e.g.
`default_simulation`) survive a subsequent call to `set_operator_defaults`.

**`get_operator_defaults`** returns the current factory defaults as a Python
dict via `yaml_to_py`.

---

### `bind_soatl.cpp` — numpy buffer registry

The registry is a static `std::unordered_map<std::string, SlotToArray>`:

```cpp
using SlotToArray = std::function<py::object(OSB&)>;
static std::unordered_map<std::string, SlotToArray> g_extractors;
```

**Key design choice — string key instead of `std::type_index`:**

Each entry is keyed by `typeid(T).name()` (the ABI-stable mangled name) rather
than `std::type_index`.  This is necessary because of `RTLD_LOCAL`:

When `dlopen` loads a pybind11 extension (e.g. `_exanb_data.so`) with
`RTLD_LOCAL` (the default), its RTTI objects are separate from those in the
main executable and in `pyonika.so`.  A `std::type_index` comparison uses
pointer equality on the internal `std::type_info` object — which would differ
across DSOs for the same logical type (e.g. `OperatorSlot<SimulationStatistics>`).

The mangled name (`typeid(T).name()`) is a plain `const char*` string that is
identical for a given type regardless of which DSO it comes from, so
`unordered_map<string, …>` lookup is cross-DSO-safe.  `slot.value_type()`
returns exactly this mangled name, making the O(1) lookup correct without any
`dynamic_cast`.

**`register_vector<T>`** creates an extractor that:
1. `static_cast`s `OSB&` to `OperatorSlot<vector<T>>*` (safe: the type string already matched)
2. Checks `has_value()`
3. Wraps `vec.data()` in a `py::buffer_info` and returns a zero-copy `py::array_t<T>`

**`register_vector_of_array<T,N>`** does the same for `vector<array<T,N>>`,
producing a 2-D array of shape `(N_elements, N)` with strides `(sizeof(T)*N, sizeof(T))`.

**`slot_as_array(OSB&)`** does the O(1) lookup by `slot.value_type()` and calls
the matching extractor.

**`_register_slot_extractor_fn` PyCapsule:**

```cpp
m.attr("_register_slot_extractor_fn") = py::capsule(
    reinterpret_cast<void*>(&register_slot_extractor),
    "_register_slot_extractor_fn");
```

This exposes the `register_slot_extractor(std::type_index, SlotExtractorFn)`
function pointer as a `PyCapsule` attribute on the `pyonika` module.  Other
`RTLD_LOCAL` extension modules (e.g. `_exanb_data.so`) retrieve it via
`PyCapsule_GetPointer` and call it to register their own extractors without
any link-time dependency on `pyonika.so`.

---

### `yaml_conv.h` — Python ↔ YAML conversion

`py_to_yaml(py::handle)` and `yaml_to_py(const YAML::Node&)` are inline
functions in a header included by all binding files.  See
[YAML / dict conversion](#yaml--dict-conversion) for the full type mapping and
the float serialisation fix.

---

### Ownership model

| Object | pybind11 holder | Who owns the C++ object |
|---|---|---|
| `ApplicationContext` | `std::shared_ptr<ApplicationContext>` | Shared between C++ and Python |
| `OperatorNode` (from `make_operator`) | `std::shared_ptr<OperatorNode>` | Python |
| `OperatorNode` (from `ctx.node()` / `apply_graph`) | `std::shared_ptr<OperatorNode>`, returned as reference | C++ graph; Python has a non-owning view |
| `OperatorSlotBase` | raw pointer, `return_value_policy::reference` | C++ operator; Python has a non-owning view |

The practical implication: **keep `ctx` alive** as long as you hold any
`OperatorNode` reference obtained from `ctx.node()` or `apply_graph`, and as
long as you use any numpy array returned by `slot_as_array()`.

---

### GIL management

`pyonika.run()`, `pyonika.run_node()`, and `pyonika.end()` all release the
Python GIL before entering C++ via `py::gil_scoped_release`.  This allows:

- OpenMP parallel regions inside simulation kernels to run on all cores
  without being serialised by the interpreter lock
- Other Python threads to make progress while a simulation step is running

The GIL is reacquired automatically when the `py::gil_scoped_release` object
goes out of scope (i.e. when the C++ call returns).

---

### MPI multi-run support

The MPI standard forbids calling any MPI routine after `MPI_Finalize()`,
including `MPI_Initialized()`.  When `end()` called `MPI_Finalize()` directly,
any subsequent `init()` in the same process would see `MPI_Initialized() == 1`
(true — MPI was initialised) and enter the "external MPI" branch, which calls
`MPI_Query_thread()` — crashing immediately.

Two changes in `src/core/api.cpp` fix this:

1. **`initialize_mpi`** now also checks `MPI_Finalized()`.  When onika calls
   `MPI_Init` for the first time, it registers a one-shot `atexit` handler
   that calls `MPI_Finalize()` (guarded by a `MPI_Finalized()` check).  On
   subsequent `init()` calls, MPI is already initialised and not yet finalised,
   so the code takes the "external MPI" path safely.

2. **`finalize`** (called by `end()`) no longer calls `MPI_Finalize()` directly.
   The `atexit` handler takes sole responsibility, ensuring `MPI_Finalize` is
   called exactly once when the Python process exits.

This has no observable effect on single-run C++ programs: `MPI_Finalize` is
still called exactly once, just deferred to the natural process-exit sequence.

---

### Simulation alias resolution

`load_yaml_input` (in `src/core/api.cpp`) extracts the `simulation:` key from
the YAML input and removes it from the operator defaults before registering
them with the factory.  This means that if `simulation: default_simulation` is
defined in the `.msp` file, the string `"simulation"` cannot be resolved as a
batch body item — the factory would throw `"Could not find operator factory for 'simulation'"`.

`build_simulation_graph` in `bind_app.cpp` works around this with a pre-pass
over the Python list before calling into C++:

```python
# What the pre-pass does, logically:
if "simulation" not in operator_defaults:
    sim_node = ctx.m_simulation_node   # e.g. scalar "default_simulation"
    resolved = []
    for item in simulation_list:
        if item == "simulation":
            if isinstance(sim_node, str):
                resolved.append(sim_node)          # e.g. "default_simulation"
            elif isinstance(sim_node, list):
                resolved.extend(sim_node)          # inline sequence
        else:
            resolved.append(item)
    simulation_list = resolved
```

This makes `build_simulation_graph(ctx, ["simulation"])` behave identically to
`build_simulation_graph(ctx, ["default_simulation"])` when `main-config.msp`
contains `simulation: default_simulation`.
