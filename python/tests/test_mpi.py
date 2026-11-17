"""MPI runs. Also run under mpirun by the pyonika_tests_mpi_np2 ctest entry,
which sets PYONIKA_TEST_NPROCS to the number of processes."""

import os

import pytest

import pyonika

EXPECTED_NPROCS = int(os.environ.get("PYONIKA_TEST_NPROCS", "1"))


@pytest.fixture
def main_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return os.path.join(os.environ["ONIKA_CONFIG_PATH"], "main-config.msp")


def test_mpi_size_and_rank(main_config):
    for _ in range(2):  # MPI stays initialized across init/end cycles
        ctx = pyonika.init(["pyonika", main_config])
        assert ctx.error_code < 0
        assert ctx.mpi_nprocs == EXPECTED_NPROCS
        assert 0 <= ctx.mpi_rank < ctx.mpi_nprocs
        pyonika.run(ctx)
        pyonika.end(ctx)
