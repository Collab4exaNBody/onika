"""
Local simulation runs started by an MCP server, in the background.

A job runs ``<python> -c "<package>.run_file(input)"`` (under ``mpirun`` when
more than one process is asked for) in the input's directory, with its output
in ``<input>.log``. Jobs live as long as the server process.
"""

import itertools
import os
import shutil
import subprocess
import sys
import time

__all__ = ["JobManager"]

_RUN = "import sys, {package} as app; sys.exit(app.run_file(sys.argv[1], overrides={overrides!r}))"


class JobManager:
    def __init__(self, package):
        self.package = package
        self._jobs = {}
        self._ids = itertools.count(1)

    def start(self, input_path, nprocs=1, overrides=None, omp_threads=None):
        input_path = os.path.abspath(input_path)
        argv = [sys.executable, "-c", _RUN.format(package=self.package, overrides=overrides), input_path]
        if nprocs > 1:
            mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
            if not mpirun:
                raise RuntimeError("nprocs > 1 needs mpirun or mpiexec in PATH")
            argv = [mpirun, "-n", str(nprocs)] + argv
        env = dict(os.environ)
        if omp_threads:
            env["OMP_NUM_THREADS"] = str(omp_threads)
        log_path = input_path + ".log"
        log = open(log_path, "w")
        proc = subprocess.Popen(argv, cwd=os.path.dirname(input_path), env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        job_id = str(next(self._ids))
        self._jobs[job_id] = {"proc": proc, "log": log, "log_path": log_path, "input": input_path,
                              "nprocs": nprocs, "start": time.time()}
        return job_id

    def status(self, job_id, tail=40):
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"no job '{job_id}' (known: {', '.join(self._jobs) or 'none'})")
        code = job["proc"].poll()
        if code is not None and not job["log"].closed:
            job["log"].close()
        with open(job["log_path"], errors="replace") as f:
            lines = f.read().splitlines()
        state = "running" if code is None else ("finished" if code == 0 else "failed")
        return {"job_id": job_id, "state": state, "exit_code": code, "input": job["input"],
                "log": job["log_path"], "elapsed": round(time.time() - job["start"], 1),
                "log_tail": "\n".join(lines[-tail:])}

    def stop(self, job_id):
        job = self._jobs[job_id]
        if job["proc"].poll() is None:
            job["proc"].terminate()
            try:
                job["proc"].wait(10)
            except subprocess.TimeoutExpired:
                job["proc"].kill()
        return self.status(job_id)

    def list(self):
        return [self.status(j, tail=0) for j in self._jobs]
