"""
Building blocks for MCP servers of onika-based applications.

This package is pure Python and never imports ``pyonika`` in the server
process (no C++ library, no MPI initialization there): work that runs onika
goes through ``worker.run_worker`` (a child process that imports the
application package), so that a crash or an abort in a simulation never
takes the server down.

* ``units``   -- static checking of quantities ("5.0 ang") and their dimension
* ``worker``  -- run catalog / check / smoke commands in a child process
* ``catalog`` -- cached operator catalog (documentation and slots)
* ``smoke``   -- build check and short run of an input in a fresh run directory
* ``jobs``    -- background local runs
"""
