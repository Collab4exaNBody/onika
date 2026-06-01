# pyonika — Python bindings for the onika HPC simulation framework

`pyonika` is a PyBind11 extension module that exposes the onika C++ API to Python. It lets you initialise, inspect, configure, and run onika simulations from Python, and read operator slot data as numpy arrays.

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Build and install](#build-and-install)
3. [Quick start](#quick-start)
4. [Reproducing an .msp file in Python](#reproducing-an-msp-file-in-python)
5. [Example scripts](#example-scripts)
6. [API reference](#api-reference)
   - [Module-level functions](#module-level-functions)
   - [ApplicationContext](#applicationcontext)
   - [OperatorNode](#operatornode)
   - [OperatorSlotBase](#operatorslotbase)
   - [Exceptions](#exceptions)
7. [YAML / dict configuration](#yaml--dict-configuration)
8. [Numpy buffer protocol](#numpy-buffer-protocol)
9. [Implementation notes](#implementation-notes)
10. [Source files](#source-files)

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.26 | Required by onika |
| C++ compiler | C++20 | GCC ≥ 11 or Clang ≥ 14 |
| Python 3 | ≥ 3.8 | Header package required |
| python3-dev | any | Provides `Python.h` |
| numpy | ≥ 1.20 | Required at import time |
| pybind11 | ≥ 2.13 | Fetched automatically by CMake |

Install system packages on Debian/Ubuntu:
```bash
sudo apt-get install python3-dev python3-pip python3-numpy
```

Or use a virtual environment:
```bash
python3 -m venv ~/onika-env
source ~/onika-env/bin/activate
pip install numpy
```

---

## Build and install

Enable the `ONIKA_BUILD_PYTHON` CMake option. Point CMake at the correct Python interpreter if needed.

```bash
cmake -B build-onika \
      -DONIKA_BUILD_PYTHON=ON \
      -DCMAKE_INSTALL_PREFIX=/path/to/install \
      -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build build-onika --target onika pyonika
cmake --install build-onika
```

After install, source the generated environment script to put `pyonika` on `PYTHONPATH` and set all onika runtime variables (`ONIKA_CONFIG_PATH`, `ONIKA_PLUGIN_PATH`, `ONIKA_DATA_PATH`):

```bash
source /path/to/install/bin/setup-env.sh
python exemples/pyonika_reproduce_print_loop_case.py
```

The installed layout is:

```
<install-prefix>/
├── bin/
│   ├── setup-env.sh    ← source this before running Python scripts
│   └── onikarun
├── lib/
│   └── pyonika.<platform>.so
├── plugins/
└── data/
```

---

## Quick start

After sourcing `setup-env.sh` (see [Build and install](#build-and-install)):

```python
import os, sys
import pyonika

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")

# Initialise — equivalent to running onika-exec with the given config file
ctx = pyonika.init([sys.argv[0], main_config])

if ctx.error_code >= 0:   # early exit (help, unit tests, …)
    sys.exit(ctx.error_code)

# Inspect the operator graph
root = ctx.node("simulation")
root.apply_graph(lambda op: print(op.pathname()))

# Run the simulation
pyonika.run(ctx)
pyonika.end(ctx)
```

---

## Reproducing an .msp file in Python

An `.msp` file has two logical sections that map directly to Python calls:

| `.msp` section | Python equivalent |
|---|---|
| Top-level named keys (`compute_loop`, `global`, …) | `pyonika.set_operator_defaults({…})` |
| `simulation:` sequence | `pyonika.build_simulation_graph(ctx, […])` |
| `- operator_name` | `"operator_name"` string in the list |
| `- op: {key: val}` | `{"op": {"key": val}}` dict in the list |
| `loop:` / `condition:` / `body:` / `rebind:` | same keys in the dict |

**Full example** — `exemples/pyonika_reproduce_print_loop_case.py` reproduces `data/exemples/print_loop.msp`:

```python
import pyonika

# 1. Bootstrap the framework (MPI, plugins, logging) — do NOT run its graph.
ctx = pyonika.init([argv0, "data/config/main-config.msp"])

# 2. Register named operator defaults (top-level .msp keys).
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

# 3. Build the simulation graph (the simulation: sequence).
graph = pyonika.build_simulation_graph(ctx, [
    "global", "mpi_comm_world", "init_cuda", "global",
    {"unit_system": {"verbose": True, "unit_system": {"length": "meter", ...}}},
    "compute_loop",
])

# 4. Run, then finalise.
pyonika.run_node(ctx, graph)
pyonika.end(ctx)
```

> **Float values in dicts:** Python floats like `1.0` are always serialised with a
> decimal point (`"1.0"`) so yaml-cpp unambiguously treats them as `double`, not `long`.

---

## Example scripts

The `exemples/` directory contains five Python scripts that demonstrate the API. All of them assume `setup-env.sh` has been sourced.

### `pyonika_dryrun_test_import_pyonika.py`

Minimal smoke test — imports `pyonika` and prints a confirmation message. Useful for verifying that the install is correct before going further.

```bash
python exemples/pyonika_dryrun_test_import_pyonika.py
# Onika successfully imported!!!
```

### `pyonika_run_main_config.py`

Runs a simulation from `main-config.msp`, then demonstrates the factory API:

- calls `pyonika.init()` with `main-config.msp` — onika builds the simulation graph internally
- traverses the graph with `ctx.node("simulation").apply_graph(...)` and prints all operators and their slots
- lists all registered operators via `pyonika.available_operators()`
- creates two standalone operators with `pyonika.make_operator()` and inspects their slots
- executes the graph with `pyonika.run(ctx)` and finalises with `pyonika.end(ctx)`

Use this script when you already have a working `.msp` file and want to inspect or extend it from Python.

### `pyonika_reproduce_print_loop_case.py`

Full Python reproduction of `data/exemples/print_loop.msp` without using the `.msp` file at all:

- calls `pyonika.init()` with `main-config.msp` only to bootstrap MPI, OpenMP, plugins and logging — the graph from that file is **not** executed
- registers custom named operator definitions with `pyonika.set_operator_defaults()`
- builds the simulation graph entirely from Python with `pyonika.build_simulation_graph()`
- executes the Python-built graph with `pyonika.run_node(ctx, graph)` and finalises

Use this script as a template when you want to drive a simulation entirely from Python without writing any `.msp` file.

### `pyonika_execute_user_specified_msp_file.py`

Interactive launcher that prompts the user for an `.msp` file path at runtime, with readline-based tab-completion for file paths:

- sets up a tab-completer (compatible with both GNU readline and macOS libedit)
- prompts the user to enter the path to any `.msp` input file
- calls `pyonika.init()` with the supplied file, runs the graph, and finalises

Use this script when you want to quickly run an arbitrary `.msp` file without editing the source.

### `pyonika_run_simulation.py`

Demonstrates running **multiple simulations sequentially** from a single Python process using named batch operators already defined in `main-config.msp`:

```python
import os, sys, pyonika

main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")

# 1 — run the "simulation_test" batch by its explicit name
ctx = pyonika.init([sys.argv[0], main_config])
graph = pyonika.build_simulation_graph(ctx, ["simulation_test"])
pyonika.run_node(ctx, graph)
pyonika.end(ctx)

# 2 — run "default_simulation" by its explicit name
ctx = pyonika.init([sys.argv[0], main_config])
graph = pyonika.build_simulation_graph(ctx, ["default_simulation"])
pyonika.run_node(ctx, graph)
pyonika.end(ctx)

# 3 — use the symbolic name "simulation" (alias resolved via ctx.m_simulation_node)
ctx = pyonika.init([sys.argv[0], main_config])
graph = pyonika.build_simulation_graph(ctx, ["simulation"])
pyonika.run_node(ctx, graph)
pyonika.end(ctx)

# 4 — omit the argument entirely; defaults to ctx.m_simulation_node from init()
ctx = pyonika.init([sys.argv[0], main_config])
graph = pyonika.build_simulation_graph(ctx)
pyonika.run_node(ctx, graph)
pyonika.end(ctx)
```

`simulation_test` and `default_simulation` are top-level named batch sequences in `main-config.msp`. `build_simulation_graph(ctx, ["name"])` looks up that batch via the operator defaults loaded by `init()` and builds the corresponding graph.

> **"simulation" as a symbolic alias.**  In `main-config.msp`, `simulation: default_simulation` declares `simulation` as an alias rather than a concrete batch sequence.  `load_yaml_input` (in `api.cpp`) extracts this key and removes it from the operator defaults before registering them with the factory — so the factory cannot resolve `"simulation"` when it appears as a body item.  The Python binding's `build_simulation_graph` pre-resolves any `"simulation"` string in the list by substituting it with `ctx.m_simulation_node` (the stored value of the `simulation:` key) before passing the list to C++. See [Simulation alias resolution](#simulation-alias-resolution).

> **No argument — use the config's simulation directly.**  `build_simulation_graph(ctx)` (no `simulation` argument) passes `ctx.m_simulation_node` straight to the C++ factory, exactly replicating the path taken by `init()` when it builds and stores the graph internally. This is the most direct way to re-execute whatever `simulation:` is set to in the `.msp` file.

> **Multiple init/end cycles.** Each `init()` call reloads YAML, re-reads plugins, and resets the operator defaults. Thanks to the MPI `atexit` fix (see [MPI multi-run support](#mpi-multi-run-support)), MPI stays alive across all cycles and is finalised once at process exit. This replaces the previous behaviour where `end()` called `MPI_Finalize()` directly, which prevented any subsequent `init()` call from using MPI.

Use this script as a template for parameter sweeps, sensitivity studies, or any workflow that needs to run several independent simulations in one Python session.

---

## API reference

### Module-level functions

#### `pyonika.init(argv: list[str]) -> ApplicationContext`

Initialise onika from a list of command-line arguments. `argv[0]` is the program name (used only in usage messages); `argv[1]` is the path to the `.msp` configuration file. Additional `--key=value` pairs are accepted.

```python
ctx = pyonika.init([sys.argv[0], "my_sim.msp"])
ctx = pyonika.init([sys.argv[0], "my_sim.msp", "--configuration.omp_num_threads=4"])
```

Raises `pyonika.OnikaError` or `RuntimeError` on failure.

#### `pyonika.run(ctx: ApplicationContext)`

Execute the simulation graph that was built by `init()` from the `.msp` file. Releases the Python GIL so OpenMP threads can run freely.

#### `pyonika.run_node(ctx: ApplicationContext, node: OperatorNode)`

Execute an arbitrary graph rooted at `node`, using the profiling settings from `ctx`. Releases the GIL in the same way as `run`.

**When to use which:**

| Situation | Call |
|---|---|
| You called `init()` with a full `.msp` file and want to run its graph | `run(ctx)` |
| You built the graph yourself with `build_simulation_graph()` | `run_node(ctx, graph)` |

The difference is which node is executed: `run` always runs `ctx`'s internal simulation graph (set during `init`); `run_node` runs the node you pass in. A graph built with `build_simulation_graph()` is not stored inside `ctx`, so `run` would not reach it — use `run_node` instead.

```python
# Pattern A — .msp-driven
ctx = pyonika.init([sys.argv[0], "my_sim.msp"])
pyonika.run(ctx)      # runs the graph from my_sim.msp
pyonika.end(ctx)

# Pattern B — Python-driven
ctx = pyonika.init([sys.argv[0], "main-config.msp"])  # bootstrap only
graph = pyonika.build_simulation_graph(ctx, [...])
pyonika.run_node(ctx, graph)   # runs the Python-built graph
pyonika.end(ctx)
```

#### `pyonika.end(ctx: ApplicationContext)`

Finalise the simulation: write profiling traces and free all operator resources. MPI is **not** finalized here — it is deferred to an `atexit` handler registered on the first `init()` call, so that multiple `init`/`end` cycles within a single Python process work correctly. Must be called once after each `run` or `run_node`.

#### `pyonika.make_operator(name: str, config: dict = {}) -> OperatorNode`

Instantiate a registered operator by name and optionally configure its input slots from a Python dict. The dict is converted to a YAML node before being passed to `yaml_initialize`.

```python
op = pyonika.make_operator("unit_system", {"verbose": True})
op = pyonika.make_operator("message", {"mesg": "Hello from Python"})
```

After creating a standalone operator, call `op.compile()` before accessing slot values.

Raises `pyonika.OnikaError` if the operator name is not registered.

#### `pyonika.available_operators() -> list[str]`

Return a sorted list of all operator names registered in the factory (builtins + loaded plugins).

#### `pyonika.set_operator_defaults(defaults: dict)`

Register named operator default definitions — equivalent to the top-level non-`simulation`, non-`configuration` keys in an `.msp` file. Must be called before `build_simulation_graph` if the simulation list references custom named operators.

```python
pyonika.set_operator_defaults({
    "my_loop": {"loop": True, "condition": "keep_going", "body": ["my_op"]},
    "global":  {"dt": 1.0, "nsteps": 100},
})
```

#### `pyonika.build_simulation_graph(ctx: ApplicationContext, simulation: list = None) -> OperatorNode`

Build a simulation graph from a Python list of operator specs (the `simulation:` sequence of an `.msp` file). Calls `post_graph_build()` on every node, which is required for slot resource allocation and graph connections.

**`simulation` is optional.** When omitted (or `None`), the simulation node stored in `ctx` during `init()` is used directly — identical to what `init()` does internally. This is the simplest way to re-run the simulation defined in the `.msp` file:

```python
ctx = pyonika.init([sys.argv[0], "main-config.msp"])
graph = pyonika.build_simulation_graph(ctx)   # uses "simulation:" from main-config.msp
pyonika.run_node(ctx, graph)
pyonika.end(ctx)
```

If an item in an explicit `simulation` list is the string `"simulation"` and that key is not present in the current operator defaults (always the case after `init()` — it is extracted and stored in `ctx` separately), it is automatically resolved to `ctx.m_simulation_node` before the graph is built. This lets you write `["simulation"]` as a shorthand for "whatever `simulation:` points to in the config file". See [Simulation alias resolution](#simulation-alias-resolution).

Returns the root `OperatorNode`. Pass it to `run_node(ctx, graph)` to execute it.

```python
graph = pyonika.build_simulation_graph(ctx, [
    "global", "mpi_comm_world",
    {"unit_system": {"verbose": True}},
    "my_loop",
])
pyonika.run_node(ctx, graph)
```

#### `pyonika.slot_as_array(slot: OperatorSlotBase) -> numpy.ndarray | None`

Return a numpy array view of a slot's value. Returns `None` if the slot type is not registered or the value is not yet initialised. See [Numpy buffer protocol](#numpy-buffer-protocol).

---

### ApplicationContext

Returned by `pyonika.init()`. Holds the full simulation state.

| Attribute / method | Type | Description |
|---|---|---|
| `mpi_rank` | `int` | MPI rank of this process |
| `mpi_nprocs` | `int` | Total number of MPI processes |
| `ngpus` | `int` | Number of GPUs available |
| `cpucount` | `int` | Number of CPU cores |
| `error_code` | `int` | `-1` = normal; `≥ 0` = early exit code |
| `node(path)` | `OperatorNode` | Look up an operator by its dotted pathname (e.g. `"simulation.unit_system"`) |
| `set_multiple_run(bool)` | — | Allow the simulation graph to be run more than once |

`node()` returns a **non-owning reference**. Keep `ctx` alive as long as the returned node is in use.

---

### OperatorNode

Represents a node in the simulation computation graph (SCG).

| Method | Returns | Description |
|---|---|---|
| `name()` | `str` | Local operator name |
| `pathname()` | `str` | Fully qualified dotted path |
| `depth()` | `int` | Depth in the graph tree |
| `in_slot_count()` | `int` | Number of input slots |
| `out_slot_count()` | `int` | Number of output slots |
| `in_slots()` | `list[(str, OperatorSlotBase)]` | All input slots as (name, slot) pairs |
| `out_slots()` | `list[(str, OperatorSlotBase)]` | All output slots as (name, slot) pairs |
| `compiled()` | `bool` | Whether slot resources have been allocated |
| `compile()` | — | Allocate slot resources (needed for standalone `make_operator` nodes) |
| `yaml_initialize(config: dict)` | — | Configure input/output slots from a Python dict |
| `slot_values()` | `dict[str, str]` | String representations of all initialised slot values |
| `slot_as_array(name: str)` | `ndarray \| None` | numpy view of a named slot (see below) |
| `apply_graph(callback)` | — | Depth-first traversal; calls `callback(node)` for each node |

```python
# Configure an operator already in the graph
ctx.node("simulation.unit_system").yaml_initialize({"verbose": True})

# Inspect all slots
for name, slot in ctx.node("simulation.my_op").in_slots():
    print(name, slot.value_type(), slot.value_as_string())
```

---

### OperatorSlotBase

Represents one input or output slot on an operator.

| Method | Returns | Description |
|---|---|---|
| `name()` | `str` | Slot name |
| `pathname()` | `str` | Fully qualified path |
| `value_type()` | `str` | C++ type name (`typeid(T).name()`) |
| `documentation()` | `str` | Doc string (if provided by the operator) |
| `is_input()` | `bool` | True for INPUT or INPUT_OUTPUT |
| `is_output()` | `bool` | True for OUTPUT or INPUT_OUTPUT |
| `is_input_only()` | `bool` | True for INPUT only |
| `is_output_only()` | `bool` | True for OUTPUT only |
| `has_value()` | `bool` | True if the value has been materialised |
| `is_required()` | `bool` | True if the slot must have a value |
| `value_as_string()` | `str` | String representation of the value |
| `value_as_bool()` | `bool` | Boolean interpretation of the value |
| `yaml_initialize(config)` | — | Set the slot value from a Python object |

`OperatorSlotBase` instances are **non-owning references**. They are valid as long as the owning operator (and therefore its `ApplicationContext`) is alive.

---

### Exceptions

| Exception | Raised when |
|---|---|
| `pyonika.OnikaError` | Unknown operator name passed to `make_operator`, slot type incompatibility, or any `OperatorCreationException` from the factory |
| `RuntimeError` | Any other `std::exception` escaping an onika call (slot type mismatch, bad YAML, etc.) |

```python
try:
    op = pyonika.make_operator("nonexistent_operator")
except pyonika.OnikaError as e:
    print(f"Operator not found: {e}")
```

> **Note on fatal errors:** onika uses an internal `fatal_error()` mechanism for unrecoverable errors (e.g. incompatible slot types during graph compilation). In Python mode these throw `std::runtime_error` and are caught as `RuntimeError`. In non-Python mode they call `std::abort()`.

---

## YAML / dict configuration

Operator slots are configured in `.msp` files using YAML. `pyonika` allows the same configuration from Python dicts, which are converted to YAML nodes before being passed to `OperatorNode::yaml_initialize`.

**YAML → Python type mapping:**

| YAML | Python |
|---|---|
| mapping | `dict` |
| sequence | `list` |
| integer scalar | `int` |
| float scalar | `float` |
| bool scalar (`true`/`false`) | `bool` |
| null | `None` |
| other scalar | `str` |

**Example — replicate an `.msp` stanza in Python:**

```yaml
# In an .msp file:
- unit_system:
    verbose: true
    unit_system:
      length: meter
      mass: kilogram
      time: second
```

```python
# Equivalent in Python:
op = pyonika.make_operator("unit_system", {
    "verbose": True,
    "unit_system": {
        "length": "meter",
        "mass": "kilogram",
        "time": "second",
    }
})
```

---

## Numpy buffer protocol

`slot_as_array()` returns a **non-owning numpy array** backed by the slot's in-memory storage. No data is copied.

**Supported slot types:**

| C++ slot type | numpy dtype | shape |
|---|---|---|
| `std::vector<double>` | `float64` | `(N,)` |
| `std::vector<float>` | `float32` | `(N,)` |
| `std::vector<int>` / `int32_t` | `int32` | `(N,)` |
| `std::vector<long>` / `int64_t` | `int64` | `(N,)` |
| `std::vector<uint32_t>` | `uint32` | `(N,)` |
| `std::vector<uint64_t>` / `size_t` | `uint64` | `(N,)` |
| `std::vector<std::array<double,3>>` | `float64` | `(N, 3)` |
| `std::vector<std::array<float,3>>` | `float32` | `(N, 3)` |
| `std::vector<std::array<double,4>>` | `float64` | `(N, 4)` |
| `std::vector<std::array<float,4>>` | `float32` | `(N, 4)` |

Returns `None` for unregistered types or uninitialised slots.

**Usage:**

```python
arr = ctx.node("simulation.my_op").slot_as_array("my_vector")
if arr is not None:
    print(arr.shape, arr.dtype, arr.mean())

# Module-level variant:
arr = pyonika.slot_as_array(slot_obj)
```

**Lifetime:** The array's memory is owned by the operator slot. Keep the `ApplicationContext` alive for as long as you use the array. Assigning `arr = None` releases the Python reference but does not free the C++ memory.

**Standalone operators:** For operators created with `make_operator()` (not part of a simulation graph), call `op.compile()` before accessing slot arrays — this triggers slot resource allocation.

```python
op = pyonika.make_operator("default_slot_value_from_ctor_args")
op.compile()
arr = op.slot_as_array("input1")   # → np.ndarray
```

---

## Implementation notes

### What was modified in onika

**`include/onika/log.h` and `src/core/log.cpp`**

`FatalErrorLogStream::~FatalErrorLogStream()` was marked `noexcept(false)` and a static `enable_python_mode(bool)` function was added. When Python mode is active (enabled at module load), the destructor throws `std::runtime_error` instead of calling `std::abort()`. This makes onika fatal errors recoverable from Python.

### Simulation alias resolution

**`python/bind_app.cpp` — `build_simulation_graph`**

`load_yaml_input` (in `src/core/api.cpp`) extracts the `simulation:` key from the YAML input and removes it from the operator defaults before registering them with the factory. This means that if `simulation: default_simulation` is defined in the `.msp` file, the string `"simulation"` cannot be resolved as a batch body item — the factory throws "Could not find a operator factory for 'simulation'".

The Python binding works around this at the call site: before forwarding the simulation list to C++, it checks each scalar item against the string `"simulation"`. If found and `"simulation"` is absent from the current operator defaults (the normal case after `init()`), it substitutes the item with the content of `ctx.m_simulation_node`:

- scalar `ctx.m_simulation_node` (e.g. `"default_simulation"`) → the item `"simulation"` is replaced with `"default_simulation"`, so the factory resolves it through the normal batch lookup.
- sequence `ctx.m_simulation_node` (inline simulation list) → the items from the sequence are inlined in place of `"simulation"`.

This makes `build_simulation_graph(ctx, ["simulation"])` behave exactly like `build_simulation_graph(ctx, ["default_simulation"])` when the config contains `simulation: default_simulation`.

### MPI multi-run support

**`src/core/api.cpp` — `initialize_mpi` and `finalize`**

The MPI standard forbids calling any MPI routine after `MPI_Finalize()`, including `MPI_Initialized()`. When `end()` called `MPI_Finalize()` directly, any subsequent `init()` in the same process would see `MPI_Initialized() == 1` (true — MPI was initialised) and enter the "external MPI" branch, which calls `MPI_Query_thread()` — crashing immediately.

The fix introduces two changes:

1. **`initialize_mpi`** now also checks `MPI_Finalized()`. When onika calls `MPI_Init` for the first time, it registers a one-shot `atexit` handler that calls `MPI_Finalize()` (guarded by a `MPI_Finalized()` check). On subsequent `init()` calls, MPI is already initialised and not yet finalised, so the code takes the "external MPI" path safely.

2. **`finalize`** no longer calls `MPI_Finalize()` directly. The `atexit` handler takes over that responsibility, ensuring MPI is finalised exactly once when the Python (or C++) process exits.

This has no observable effect on single-run C++ programs: `MPI_Finalize` is still called exactly once, just deferred to the natural process-exit sequence.

### CMake integration

The Python module is opt-in via `-DONIKA_BUILD_PYTHON=ON`. pybind11 is fetched automatically via `FetchContent` if not already installed:

```cmake
option(ONIKA_BUILD_PYTHON "Build Python (PyBind11) bindings" OFF)
if(ONIKA_BUILD_PYTHON)
  add_subdirectory(python)
endif()
```

### GIL management

`pyonika.run()` and `pyonika.end()` release the Python GIL (`py::gil_scoped_release`) so that OpenMP threads spawned by onika are not blocked. The GIL is reacquired automatically when control returns to Python.

### OperatorNode ownership model

`OperatorNode` is registered with `std::shared_ptr<OperatorNode>` as the pybind11 holder type.

- **Factory-created nodes** (`make_operator`) → Python holds shared ownership.
- **Graph-internal nodes** (`ctx.node(...)`, `apply_graph` callback) → returned with `return_value_policy::reference`; Python holds a non-owning view.

### Python float → YAML conversion

`YAML::Node(double)` in yaml-cpp serialises whole-number floats (e.g. `1.0`) as `"1"` with no decimal point. yaml-cpp then re-reads `"1"` as `long`, which breaks slot type matching when an operator expects `double`.

`yaml_conv.h` works around this by formatting Python floats with `std::setprecision(17)` and appending `.0` when no decimal point, `e`, or special value (`nan`/`inf`) is present. This guarantees `1.0` → `"1.0"` and yaml-cpp correctly identifies it as a float.

### SOATL numpy dispatch

`bind_soatl.cpp` maintains a static registry of `std::type_index → extractor` functions. Each extractor uses `dynamic_cast<OperatorSlot<T>*>` to verify the slot type at runtime, then wraps the underlying `std::vector<T>::data()` pointer in a `py::buffer_info` without copying. New slot types can be supported by adding entries to the registry.

---

## Source files

```
onika/
├── include/onika/log.h          ← modified: noexcept(false), enable_python_mode()
├── src/core/log.cpp             ← modified: s_fatal_error_python_mode flag
├── src/core/api.cpp             ← modified: MPI atexit fix for multi-run support
├── python/
│   ├── CMakeLists.txt           ← pybind11 FetchContent, pyonika target
│   ├── module.cpp               ← PYBIND11_MODULE entry point
│   ├── yaml_conv.h              ← py_to_yaml / yaml_to_py utilities
│   ├── bind_scg.h / .cpp        ← OperatorNode, OperatorSlotBase
│   ├── bind_soatl.h / .cpp      ← numpy buffer protocol
│   ├── bind_factory.h / .cpp    ← make_operator, available_operators
│   ├── bind_app.h / .cpp        ← ApplicationContext, init/run/end
│   └── README.md                ← this file
└── exemples/
    ├── pyonika_dryrun_test_import_pyonika.py      ← import smoke test
    ├── pyonika_run_main_config.py                 ← .msp-driven run + factory/slot inspection
    ├── pyonika_reproduce_print_loop_case.py       ← full Python reproduction of print_loop.msp
    ├── pyonika_execute_user_specified_msp_file.py ← interactive launcher for any .msp file
    └── pyonika_run_simulation.py                  ← multiple sequential simulations in one process
```
