"""pyonika API: introspection, YAML conversion, object lifetimes, output capture."""

import gc
import io
import os

import numpy as np
import pytest

import pyonika


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """Context of main-config.msp: plugins loaded, graph built, not run."""
    monkeypatch.chdir(tmp_path)
    main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
    c = pyonika.init(["pyonika", main_config])
    assert c.error_code < 0
    yield c
    pyonika.end(c)


def test_describe_operator(ctx):
    d = pyonika.describe_operator("unit_system")
    assert d["name"] == "unit_system"
    slots = {s["name"]: s for s in d["slots"]}
    assert slots["verbose"]["type"] == "bool"
    assert slots["verbose"]["direction"] == "in"
    for s in d["slots"]:
        assert set(s) == {"name", "type", "direction", "required", "documentation", "value"}


def test_yaml_round_trip(ctx):
    values = {
        "s_yes": "yes", "s_zeros": "007", "s_unit": "1.0 ang", "s_true": "true",
        "i": 3, "big": 2**70, "neg": -12, "f": 1.0, "fe": 1.5e-3,
        "np_i": np.int64(5), "np_f": np.float32(0.5), "arr": np.arange(3),
        "b": True, "none": None, "nested": {"l": [1, "two", 3.0]},
    }
    pyonika.set_operator_defaults({"roundtrip_test": values})
    got = pyonika.get_operator_defaults()["roundtrip_test"]
    assert got["s_yes"] == "yes" and got["s_zeros"] == "007" and got["s_unit"] == "1.0 ang"
    assert got["i"] == 3 and got["big"] == 2**70 and got["neg"] == -12
    assert isinstance(got["f"], float) and got["f"] == 1.0 and got["fe"] == 1.5e-3
    assert got["np_i"] == 5 and got["np_f"] == 0.5 and got["arr"] == [0, 1, 2]
    assert got["b"] is True and got["none"] is None
    assert got["nested"] == {"l": [1, "two", 3.0]}


def test_lifetimes_after_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main_config = os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")
    c = pyonika.init(["pyonika", main_config])
    node = c.node("simulation")
    collected = []
    node.apply_graph(collected.append)
    slots = [s for n in collected for _, s in n.in_slots()]
    pyonika.end(c)
    del c, node
    gc.collect()
    # graph nodes and slots kept alive by Python references: still usable
    for n in collected:
        n.pathname()
        n.describe()
    for s in slots:
        assert s.name()
        assert not s.has_value()


def test_slot_values_typed(ctx):
    op = pyonika.make_operator("unit_system", {"verbose": True})
    op.slot_values(typed=True)  # must not raise, values parsed as YAML when possible
    assert op.describe()["name"] == "unit_system"


def test_capture_output(ctx):
    buf = io.StringIO()
    with pyonika.capture_output(buf):
        pyonika.run(ctx)
    assert len(buf.getvalue()) > 0


def test_command_line():
    argv = pyonika.command_line("in.msp", program="p",
                                overrides={"global": {"dt": 0.5, "flag": True}},
                                config={"omp_num_threads": 2})
    assert argv == ["p", "in.msp", "--set-global-dt", "0.5", "--set-global-flag", "true",
                    "--omp_num_threads", "2"]
