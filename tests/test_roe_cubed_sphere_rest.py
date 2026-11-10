#!/usr/bin/env python3
"""A resting atmosphere on the six-panel cubed sphere must stay at rest with the Roe solver.

Without the face-local projection in roe.cpp (prim2local / flux2global, as in lmars.cpp) the
x2/x3 pressure flux is p instead of p sin(theta) and the cube corners spin up (~2 m/s here).
"""
import os
import tempfile
from pathlib import Path

import yaml
from snapy import Mesh, MeshOptions, kIDN, kIPR, kIV2, kIV3

cfg = yaml.safe_load(Path(__file__).with_name("test_flux_positivity_cubedsphere.yaml").read_text())
cfg.pop("scalar")
cfg["dynamics"]["riemann-solver"]["type"] = "roe"
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
    yaml.safe_dump(cfg, f)
options = MeshOptions.from_yaml(f.name)
os.unlink(f.name)
options.set_local_horizontal_cells(6, 6)
mesh = Mesh(options)
state = []
for block in mesh.blocks:
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    w[kIDN], w[kIPR] = 1.0, 1.0e5
    state.append({"hydro_w": w})
state, _ = mesh.initialize(state)
redo = 0
for cycle in (1, 2):
    mesh.set_cycle(cycle)
    for stage in range(len(mesh.blocks[0].module("intg").stages)):
        mesh.forward(state, 300.0, stage)
    redo += mesh.check_redo(state)
vmax = max(v["hydro_w"][kIV2 : kIV3 + 1].abs().max().item() for v in state)
assert redo == 0 and vmax < 1.0e-9, f"roe: max|v|={vmax:.3e} m/s, redo={redo}"
