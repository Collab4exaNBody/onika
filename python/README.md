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
│   └── core/api.cpp         ← set_deferred_mpi_finalize() for multi-run support
└── python/
    ├── CMakeLists.txt        ← pybind11 auto-fetch, pyonika._core target, pyproject.toml
    ├── pyproject.toml.in     ← pip metadata of the installed package
    ├── pyonika/              ← Python package: __init__, env, mpi, output
    ├── include/onika/python/slot_extractor_api.h ← extractor API for downstream modules
    ├── tests/                ← pytest suite (ctest: pyonika_tests)
    ├── module.cpp            ← PYBIND11_MODULE(_core) entry point
    ├── yaml_conv.h           ← py_to_yaml / yaml_to_py (header-only)
    ├── bind_scg.h / .cpp     ← OperatorNode, OperatorSlotBase bindings
    ├── bind_soatl.h / .cpp   ← numpy buffer registry and slot_as_array
    ├── bind_factory.h / .cpp ← make_operator, available_operators, set/get_operator_defaults
    ├── bind_app.h / .cpp     ← ApplicationContext, init / run / run_node / end / build_simulation_graph
    └── examples/
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

Then install the package into your Python environment (same interpreter as
`Python3_EXECUTABLE`):

```bash
pip install /path/to/onika_install/python
```

The install prefix is baked into the package: `import pyonika` sets
`ONIKA_PLUGIN_PATH`, `ONIKA_CONFIG_PATH` and `ONIKA_DATA_PATH` itself and
initializes MPI.  Values already set by the user (e.g. by sourcing
`bin/setup-env.sh`, which also adds `<prefix>/python` to `PYTHONPATH`) take
precedence.

**Install layout:**

```
<install-prefix>/
├── bin/
│   ├── setup-env.sh       ← optional for Python users
│   └── onika-exec
├── include/onika/python/slot_extractor_api.h
├── python/
│   ├── pyproject.toml     ← pip install <prefix>/python
│   └── pyonika/           ← __init__.py, env.py, mpi.py, output.py, _paths.py, _core.<platform>.so
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

> **Examples:** `examples/pyonika_dryrun_test_import_pyonika.py` (smoke test),
> `examples/pyonika_run_main_config.py` (Pattern A + full graph/slot inspection),
> `examples/pyonika_execute_user_specified_msp_file.py` (interactive .msp launcher).

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

> **Example:** `examples/pyonika_reproduce_print_loop_case.py` — full Python reproduction of
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

#### Helpers (pure Python)

| Function | Description |
|---|---|
| `command_line(msp, *args, program="pyonika", overrides=None, config=None)` | argv for `init()`. `overrides` (nested dict) is merged on top of the input file's operator definitions through `configuration.set`, e.g. `{"global": {"dt": 1e-3}}`; `config` sets `configuration:` options, e.g. `{"omp_num_threads": 4}` |
| `run_file(msp, *args, program="pyonika", overrides=None, config=None)` | `init`, `run`, `end`; returns 0 on success or the early-exit code |
| `describe_operator(name)` | `make_operator(name).describe()`: documentation, slot types, directions, defaults (after `init()`) |
| `capture_output(stream=None, stderr=True)` | context manager sending output written by C++ during the block to `stream` (default `sys.stdout`; use an `io.StringIO` to keep it) |
| `setup_env(prefixes)` | add install prefixes (plugins, config, data); done at import for onika, by applications for their own prefix |
| `ensure_mpi_initialized()` | done at import |


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

Boolean flags can be passed as a single element — the parser defaults the
value to `true` when no following value is present:

```python
ctx = pyonika.init([sys.argv[0], "my_sim.msp", "--nogpu"])
# → { configuration: { nogpu: true } }  — disables GPU even if one is present
```

Note: `--no-gpu` would NOT work — the dash splits into nested YAML
`{ no: { gpu: true } }` which matches nothing.  The correct key is `nogpu`
(no dash), as declared in `onika/include/onika/app/default_app_config.h`.

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
resources.  MPI is **not** finalised here — importing `pyonika` turns on
deferred MPI finalisation, and an `atexit` handler registered on the first
`init()` call finalises MPI, so that multiple `init`/`end` cycles in one Python
process work correctly.

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

> **Examples:** `examples/pyonika_make_operator_usage.py` (slot introspection with `make_operator`),
> `examples/pyonika_read_slot_values.py` (reading values via `apply_graph` after `run_node`).

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
copy of a slot's value, or `None` if the type is not registered or the value
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
| `graph` | `OperatorNode` | Root of the graph built by `init()` (`None` after an early exit or `end()`) |
| `set_multiple_run(bool)` | — | Allow the simulation graph to be executed more than once |

Nodes returned by `node()`, `apply_graph()` callbacks and slots returned by
`in_slots()`/`out_slots()` keep their owner (the graph root, the traversed
node, the operator) alive, so they never dangle.  After `end()` their
resources are freed: values read from them are `None`.

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
| `documentation()` | `str` | Operator description |
| `describe()` | `dict` | `name`, `pathname`, `documentation`, `slots` (each as `OperatorSlotBase.describe()`) |
| `is_terminal()` | `bool` | True if the node has no sub-operators |
| `depth()` | `int` | Depth in the graph tree |
| `in_slot_count()` | `int` | Number of input slots |
| `out_slot_count()` | `int` | Number of output slots |
| `in_slots()` | `list[(str, OperatorSlotBase)]` | All input slots as (name, slot) pairs |
| `out_slots()` | `list[(str, OperatorSlotBase)]` | All output slots as (name, slot) pairs |
| `compiled()` | `bool` | Whether slot resources have been allocated |
| `compile()` | — | Allocate slot resources — **not needed after `make_operator()`** (factory already calls it); exposed for advanced use only |
| `yaml_initialize(config: dict)` | — | Configure input/output slots from a Python dict |
| `slot_values(typed=False)` | `dict` | Values of all initialised slots: strings, or parsed as YAML with `typed=True` |
| `slot_as_array(name: str)` | `ndarray \| None` | numpy copy of a named slot (see below) |
| `apply_graph(callback)` | — | Depth-first traversal; calls `callback(node)` for each node |

```python
# Walk the full simulation graph
ctx.node("simulation").apply_graph(lambda op: print(op.pathname()))

# Inspect slots on a specific operator
for name, slot in ctx.node("simulation.unit_system").in_slots():
    print(f"  {name}: {slot.value_type()} = {slot.value_as_string()}")

# Look up a named slot and get a numpy copy of its value
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
| `value_type()` | `str` | Mangled C++ type name (`typeid(T).name()`, extractor registry key) |
| `type_name()` | `str` | Readable C++ type name |
| `direction()` | `str` | `"in"`, `"out"` or `"inout"` |
| `documentation()` | `str` | Doc string, if provided by the operator |
| `is_input()` | `bool` | True for `INPUT` or `INPUT_OUTPUT` |
| `is_output()` | `bool` | True for `OUTPUT` or `INPUT_OUTPUT` |
| `is_input_only()` | `bool` | True for `INPUT` only |
| `is_output_only()` | `bool` | True for `OUTPUT` only |
| `has_value()` | `bool` | True if the value has been materialised |
| `is_required()` | `bool` | True if the slot must have a value |
| `value()` | object | Current value parsed as YAML (number, list, dict…), or `None` |
| `value_as_string()` | `str` | String representation of the current value |
| `value_as_bool()` | `bool` | Boolean interpretation of the current value |
| `describe()` | `dict` | `name`, `type`, `direction`, `required`, `documentation`, `value` |
| `yaml_initialize(config)` | — | Set the slot value from a Python object |

A slot keeps its operator alive (see `ApplicationContext.node`).

**Repr:** `<Slot 'simulation.my_op.data' [out] std::vector<double>>`.

---

### Exceptions

| Exception | Raised when |
|---|---|
| `pyonika.OnikaError` (subclass of `RuntimeError`) | Unknown operator name in `make_operator`, slot type incompatibility (`OperatorCreationException`), and any `fatal_error()` raised in Python mode: missing or malformed input file, invalid operator parameters, ... |
| `RuntimeError` and other builtins | Any other `std::exception` escaping an onika call. pybind11 maps a few standard exceptions to more specific builtins (e.g. `std::out_of_range` → `IndexError`, `std::invalid_argument` → `ValueError`) |

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
makes `FatalErrorLogStream::~FatalErrorLogStream()` throw `onika::FatalError`
instead, which is raised as `pyonika.OnikaError`.

Throwing is only done when it is safe; otherwise the old behaviour is kept:

- inside an OpenMP parallel region (e.g. task-parallel operators), an exception
  cannot leave the region, so the process aborts;
- if another exception is already being propagated, the process aborts;
- with more than one MPI process, `MPI_Abort` is called so that the other ranks
  do not wait forever for the rank that failed.

After a `RuntimeError` raised from `run()`/`run_node()`, the simulation graph is
left half-executed: call `end(ctx)` and start again with a new `init()`.

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

> **Example:** `examples/pyonika_reproduce_print_loop_case.py`.

---

## Running multiple simulations

Each `init()` call reloads YAML, re-reads plugins, and resets the operator
defaults.  Because `pyonika` defers MPI finalisation, MPI stays alive across
all cycles and is finalised exactly once at process exit.

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

> `import pyonika` initializes MPI (`pyonika.mpi.ensure_mpi_initialized()`)
> and registers its finalization at process exit, so this works out of the box
> as long as you do not call `MPI_Finalize()` yourself between cycles.

> **Example:** `examples/pyonika_run_simulation.py` — two init/run/end cycles
> within a single Python process.

---

## numpy buffer protocol

`slot_as_array()` returns a numpy array holding a **copy** of the slot's
in-memory C++ storage (pybind11 copies a raw buffer when no owner object is
given).  Writing to the array does not modify the slot.

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

**Lifetime:** since the data is copied, the array stays valid after `end(ctx)`
or after the operator is destroyed.  The cost is one copy of the vector per call.

**Standalone operators:** nodes created with `make_operator()` are already
compiled by the factory — `slot_as_array()` can be called on them immediately.

```python
op = pyonika.make_operator("my_op_with_vector_slot")
arr = op.slot_as_array("input1")   # → np.ndarray or None
```

**Extending the registry:** other pybind11 extension modules register
converters for their own types with `onika/python/slot_extractor_api.h`
(installed with onika), without linking against pyonika:

```cpp
#include <onika/python/slot_extractor_api.h>
namespace py = pybind11;

PYBIND11_MODULE(_myext, m)
{
  onika::python::register_slot_extractor<MySlotType>( [](const MySlotType& v) -> py::object {
    py::dict d;
    d["value"] = v.m_value;
    return d;
  } );
}
```

The registration function is published in the capsule
`pyonika._slot_extractor_api` with an API version and the pybind11 internals
version; a module built against an incompatible pyonika fails at import with
an `ImportError` instead of crashing.  (`_register_slot_extractor_fn`, the
former unversioned capsule, is kept for compatibility.)

---

## YAML / dict conversion

Operator slots are configured in `.msp` files using YAML.  `pyonika` converts
Python dicts to `YAML::Node` objects via `py_to_yaml()` before passing them to
`OperatorNode::yaml_initialize`, and converts back via `yaml_to_py()` when
reading defaults.

### `py_to_yaml()` — type mapping

| Python type | YAML node | Notes |
|---|---|---|
| `None` | Null | |
| `bool` | `true` / `false` | checked before `int` (bool ⊂ int in CPython) |
| `int` | integer scalar | exact, any size |
| `float` | float scalar | 17 significant digits, `.0` appended to whole numbers so they stay floats; `nan`/`inf` → `.nan`/`.inf` |
| `str`, `bytes` | string scalar | |
| `dict` | Map | keys converted with `str()` |
| numpy scalar / 0-d array | as its Python value | `np.int64(3)` → `3` |
| list, tuple, other sequences (numpy arrays) | Sequence | |
| anything else | string scalar | `str(obj)` |

### `yaml_to_py()` — type mapping

Plain (unquoted) scalars follow the YAML 1.2 core schema; quoted scalars are
always strings:

| YAML scalar | Python |
|---|---|
| `true` / `True` / `TRUE`, `false` / ... | `bool` |
| decimal integer (`-12`, `0`, no leading zeros) | `int` (any size) |
| decimal float (`1.5`, `1e-3`, `.5`), `.inf`, `.nan` | `float` |
| anything else (`yes`, `007`, `1.0 ang`, `3.0e-3 ps`) | `str` |
| Null (`null`, `~`, empty) | `None` |

A dict read with `get_operator_defaults()` can therefore be modified and passed
back to `set_operator_defaults()` without changing the meaning of values
(e.g. `yes` or `007` are not turned into `True` or `7`).

---

## Python bindings — implementation

### Package layout

`pyonika` is a Python package around the compiled module `pyonika._core`:

| File | Role |
|---|---|
| `module.cpp` | `PYBIND11_MODULE(_core)`: enables Python mode for `fatal_error()`, deferred MPI finalization, registers `OnikaError`, `_flush_output()` |
| `bind_app.cpp` | `ApplicationContext`, `init`, `run`, `run_node`, `end`, `build_simulation_graph` |
| `bind_scg.cpp` | `OperatorNode`, `OperatorSlotBase` (`describe()`, typed values) |
| `bind_factory.cpp` | `make_operator`, `available_operators`, `get/set_operator_defaults` |
| `bind_soatl.cpp` | slot extractor registry, `slot_as_array`, `_slot_extractor_api` capsule |
| `yaml_conv.h` | Python ↔ YAML conversion |
| `pyonika/__init__.py` | re-exports, `run_file`, `command_line`, `describe_operator`; sets up env and MPI at import |
| `pyonika/env.py` | `setup_env(prefixes)`: `ONIKA_PLUGIN_PATH`, `ONIKA_CONFIG_PATH`, `ONIKA_DATA_PATH` from install prefixes |
| `pyonika/mpi.py` | `ensure_mpi_initialized()`: locates the libmpi onika is linked with (`dladdr`) and calls `MPI_Init_thread` once |
| `pyonika/output.py` | `capture_output()`: redirects file descriptors 1/2 written by C++ to a Python stream |

### Errors

`fatal_error()` normally aborts.  In Python mode it throws `onika::FatalError`
when that is safe (outside OpenMP parallel regions, no exception in flight,
single MPI process), otherwise it still aborts (`MPI_Abort` with several
ranks).  `FatalError` and `OperatorCreationException` are translated to
`pyonika.OnikaError` (subclass of `RuntimeError`).  `init()` without input
file or with an unreadable/malformed file, and `run()` or `end()` after an
early exit (`--help`), raise instead of exiting or crashing.

### Ownership model

| Object | Held by Python as | Keeps alive |
|---|---|---|
| `ApplicationContext` | `shared_ptr` | its graph until `end()` |
| graph root (`ctx.graph`, `make_operator`, `build_simulation_graph`) | `shared_ptr` | the whole graph |
| node from `ctx.node()` | reference | the graph root |
| node passed to an `apply_graph` callback | reference | the node `apply_graph` was called on |
| slot from `in_slots()` / `out_slots()` | reference | its operator |

`end()` frees slot resources and drops the context's reference to the graph;
nodes still referenced from Python stay valid but hold no values.
`slot_as_array()` and the extractors always return copies.

### GIL management

`init`, `run`, `run_node`, `end`, `build_simulation_graph` and `make_operator`
release the GIL while onika runs (MPI/OpenMP/GPU setup, plugin loading, graph
execution).  Callbacks of `apply_graph` run with the GIL held.

### MPI multi-run support

`import pyonika` initializes MPI (`MPI_THREAD_MULTIPLE`) and registers
`MPI_Finalize` at exit.  onika then sees MPI as externally initialized and
never finalizes it in `end()`, so `init()/run()/end()` can be repeated.
`set_deferred_mpi_finalize(true)` additionally covers the case where MPI was
not initialized beforehand.

With several MPI processes (`mpirun -n N python script.py`), `import pyonika`
also installs a `sys.excepthook` (`pyonika.mpi.install_abort_on_exception`)
that calls `MPI_Abort` after printing the traceback of an uncaught exception,
so that the other processes do not wait forever in a collective operation.

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
