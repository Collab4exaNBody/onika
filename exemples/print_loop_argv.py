import os
import sys
import pyonika

# run using :
# python3 print_loop_argv.py print_loop.msp

ctx = pyonika.init([sys.argv[0], sys.argv[1]])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)
pyonika.run(ctx)
pyonika.end(ctx)
