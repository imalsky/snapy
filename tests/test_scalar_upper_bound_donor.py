#!/usr/bin/env python3
"""upper-bound must not drive a donor negative: the bound's flux blend ran after the positivity limiter.
python test_scalar_upper_bound_donor.py [--device cuda]"""
import math, sys, torch, yaml
from pathlib import Path
import test_tracer_dry_convention as J
from snapy import kIPR

DEV = "cuda" if "cuda" in sys.argv else "cpu"
if DEV == "cuda" and not torch.cuda.is_available(): sys.exit(125)  # no visible GPU: skip
c = yaml.safe_load(open(Path(__file__).resolve().parent / "test_tracer_dry_convention.yaml"))
c["dynamics"]["equation-of-state"]["limiter"] = True
c["scalar"]["upper-bound"] = 1.0e-3
c["integration"]["implicit-scheme"] = 1  # the implicit solve allows a dt past mass Courant 1
c["geometry"]["bounds"]["x2max"] = 4.0e4
b = J.make_block(c, DEV); w, r = J.initial_state(b, c, moving=False); ng = 3
w[kIPR][..., ng + 14] *= 2.0; r.zero_(); r[..., ng + 15] = 1.0e-3  # tracer at the bound in cell 15 only
v, _ = b.initialize({"hydro_w": w, "scalar_r": r}); I = J.interior(c); smin = math.inf
for _ in range(3):
    dt = b.max_time_step(v)
    for st in range(len(b.intg.stages)):
        b.forward(v, dt, st); smin = min(smin, float(v["scalar_s"][I].min()))
    b.check_redo(v)
print("upper-bound donor: min s", smin)  # main 2d9fa419: -1.83e-5
sys.exit(0 if smin >= 0.0 else 1)
