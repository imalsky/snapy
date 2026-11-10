"""Require the selected nonexistent Gloo interface to fail on both ranks."""
import subprocess
import sys

p = subprocess.run(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                   text=True, errors="replace", timeout=45)
n = p.stdout.count("Unable to find address for: snapy_no_such_interface")
if p.returncode != 0 and n == 2:
    raise SystemExit(0)
print(p.stdout)
raise SystemExit(f"expected two interface errors; exit={p.returncode}, matches={n}")
