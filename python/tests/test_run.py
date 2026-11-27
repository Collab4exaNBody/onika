"""Running a graph from Python: slot values after a run, and errors raised while it runs."""

import os

import numpy as np
import pytest

import pyonika


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """Context of main-config.msp, with plugins loaded."""
    monkeypatch.chdir(tmp_path)
    main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
    c = pyonika.init(["pyonika", main_config])
    assert c.error_code < 0
    yield c
    pyonika.end(c)


def find_node(graph, name):
    found = []
    graph.apply_graph(lambda node: found.append(node) if node.name() == name else None)
    assert found, f"no node named '{name}' in {graph}"
    return found[0]


def run_graph(ctx, simulation):
    graph = pyonika.build_simulation_graph(ctx, simulation)
    pyonika.run_node(ctx, graph)
    return graph


# default_slot_value_from_ctor_args (onikaBasicTutorial plugin) has three std::vector<double>
# input slots with default values: input1 = 10 x 5.0, input2 = [2..6], input3 = [1..7].
VECTOR_OP = "default_slot_value_from_ctor_args"


def test_scalar_slots_after_loop(ctx):
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
    graph = run_graph(ctx, ["global", "mpi_comm_world", "compute_loop"])
    slots = dict(find_node(graph, "global").out_slots())
    # sim_continue goes on while timestep <= nsteps, so next_time_step runs nsteps+1 times
    assert int(slots["timestep"].value_as_string()) == 11
    assert slots["compute_loop_continue"].value_as_bool() is False


def test_slot_as_array(ctx):
    graph = run_graph(ctx, ["global", {VECTOR_OP: {"input1": (10.0, 20.0, 30.0)}}])
    op = find_node(graph, VECTOR_OP)

    input1 = op.slot_as_array("input1")
    assert isinstance(input1, np.ndarray) and input1.dtype == np.float64
    assert input1.tolist() == [10.0, 20.0, 30.0]  # set from a Python tuple
    assert op.slot_as_array("input2").tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert op.slot_as_array("input3").tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    # module level function, from the slot object
    input3_slot = dict(op.in_slots())["input3"]
    assert pyonika.slot_as_array(input3_slot).tolist() == op.slot_as_array("input3").tolist()

    # unknown slot, or slot type without a registered extractor
    assert op.slot_as_array("no_such_slot") is None
    assert find_node(graph, "global").slot_as_array("timestep") is None


def test_slot_as_array_is_a_copy(ctx):
    graph = run_graph(ctx, ["global", VECTOR_OP])
    op = find_node(graph, VECTOR_OP)
    values = op.slot_as_array("input3")
    values[:] = -1.0
    assert op.slot_as_array("input3")[0] == 1.0


def test_recompile_raises(ctx):
    op = pyonika.make_operator("unit_system")
    with pytest.raises(pyonika.OnikaError, match="cannot be re-compiled"):
        op.compile()  # make_operator() already compiled it


def test_bad_slot_value_raises_while_running(ctx):
    # the conversion error is raised when the slot is allocated, inside an OpenMP
    # critical section: it must reach Python instead of terminating the process
    graph = pyonika.build_simulation_graph(ctx, ["global", {VECTOR_OP: {"input1": "not a list"}}])
    with pytest.raises(pyonika.OnikaError, match="could not convert key input1"):
        pyonika.run_node(ctx, graph)

    # the process can still run a valid graph
    graph = run_graph(ctx, ["global", {VECTOR_OP: {"input1": [1.0, 2.0]}}])
    assert find_node(graph, VECTOR_OP).slot_as_array("input1").tolist() == [1.0, 2.0]


def test_several_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
    for _ in range(3):
        assert pyonika.run_file(main_config) == 0


def test_no_value_after_end_of_a_run(tmp_path, monkeypatch):
    """end() frees the slot values of the graph built by init(), also after it has run:
    they read as None, not from released memory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "loop.msp").write_text(
        "global: { dt: 1.0, nsteps: 2, timestep: 0, compute_loop_continue: true }\n"
        "compute_loop_stop: { rebind: { end_at: nsteps, result: compute_loop_continue }, body: [ sim_continue ] }\n"
        "compute_loop: { loop: true, condition: compute_loop_continue, body: [ next_time_step, compute_loop_stop ] }\n"
        f"simulation: [ global, mpi_comm_world, compute_loop, {VECTOR_OP} ]\n")
    c = pyonika.init(["pyonika", "loop.msp"])
    pyonika.run(c)
    timestep = dict(c.node("simulation.global").out_slots())["timestep"]
    vector_node = c.node(f"simulation.{VECTOR_OP}")
    vector = dict(vector_node.in_slots())["input1"]
    assert timestep.value() == 3 and vector.has_value()
    pyonika.end(c)
    for s in (timestep, vector):
        assert not s.has_value()
        assert s.value() is None
    assert vector_node.slot_as_array("input1") is None
