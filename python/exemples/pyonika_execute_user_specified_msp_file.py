import os
import sys
import glob
import readline
import pyonika

# --- Enable tab-completion for file paths at the input() prompt ----------
def _path_completer(text, state):
    """Return the `state`-th match for `text` (filenames / directories)."""
    # Expand ~ and environment variables so completion works on them too
    expanded = os.path.expanduser(os.path.expandvars(text))
    matches = glob.glob(expanded + '*')
    # Add a trailing slash to directories so you can keep tabbing into them
    matches = [m + os.sep if os.path.isdir(m) else m for m in matches]
    try:
        return matches[state]
    except IndexError:
        return None

readline.set_completer_delims(' \t\n;')

# macOS ships with libedit instead of GNU readline; the bind syntax differs.
if 'libedit' in readline.__doc__:
    readline.parse_and_bind('bind ^I rl_complete')
else:
    readline.parse_and_bind('tab: complete')

readline.set_completer(_path_completer)
# -------------------------------------------------------------------------

# Ask the user for the input file name (Tab completes paths)
input_file = input("Please enter the name of the input file to run: ").strip()

if not os.path.isfile(input_file):
    print(f"Error: file '{input_file}' not found.")
    sys.exit(1)

ctx = pyonika.init([sys.argv[0], input_file])
if ctx.error_code >= 0:
    sys.exit(ctx.error_code)
pyonika.run(ctx)
pyonika.end(ctx)
