"""Errors in onika must raise Python exceptions, never kill the interpreter."""

import pytest

import pyonika


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # plugins.db is written in the working directory


def test_onika_error_is_runtime_error():
    assert issubclass(pyonika.OnikaError, RuntimeError)


def test_init_without_input_file():
    with pytest.raises(pyonika.OnikaError, match="missing input file"):
        pyonika.init(["pyonika"])


def test_init_with_missing_file():
    with pytest.raises(pyonika.OnikaError, match="Unable to open"):
        pyonika.init(["pyonika", "/nonexistent/file.msp"])


def test_init_with_malformed_yaml(tmp_path):
    bad = tmp_path / "bad.msp"
    bad.write_text("simulation: [ unclosed\n")
    with pytest.raises(pyonika.OnikaError, match="Error reading YAML"):
        pyonika.init(["pyonika", str(bad)])


def test_unknown_operator():
    pyonika.init(["pyonika", "--help"])  # loads plugins
    with pytest.raises(pyonika.OnikaError):
        pyonika.make_operator("no_such_operator")


def test_end_after_early_exit():
    ctx = pyonika.init(["pyonika", "--help"])
    assert ctx.error_code == 0
    assert ctx.graph is None
    assert ctx.node("simulation") is None
    pyonika.end(ctx)
    with pytest.raises(pyonika.OnikaError, match="no simulation graph"):
        pyonika.run(ctx)
