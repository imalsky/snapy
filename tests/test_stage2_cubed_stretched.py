#!/usr/bin/env python3
"""C24 n5 full sphere on a stretched x1: a hydrostatic rest with a uniform tracer stays put (#280)."""
import argparse
import math
from pathlib import Path
import torch
import snapy
from snapy import kIDN, kIPR

P0, H0, G = 1.e6, 6.e4, 22.

def check(device):
    mesh = snapy.Mesh(snapy.MeshOptions.from_yaml(str(Path(__file__).with_suffix(".yaml"))))
    blocks = list(mesh.blocks)
    f = blocks[0].module("coord").buffer("x1f").double()[3:34]
    p = P0 * torch.exp(-((f[1:] + f[:-1]) / 2 - f[0]) / H0)
    col = torch.stack([p / (G * H0)] + [0 * p] * 3 + [p]).view(5, 1, 1, 30)
    wb, _, _ = snapy.balance_column(col, (f[1:] - f[:-1]).contiguous(), G, rtol=1.e-13, stretched=True)
    wb = torch.cat([wb[..., :1]] * 3 + [wb] + [wb[..., -1:]] * 3, -1)
    state = []
    for b in blocks:
        b.to(torch.device(device))
        bufs = dict(b.named_buffers())
        w = bufs["hydro.D"].clone().zero_()
        w[kIDN], w[kIPR] = wb[kIDN, 0, 0].to(w), wb[kIPR, 0, 0].to(w)
        state.append({"hydro_w": w, "scalar_r": bufs["scalar.D"].clone().fill_(.01)})
    state, _ = mesh.initialize(state)
    for cycle in range(1, 101):
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(state)
        for stage in range(len(blocks[0].module("intg").stages)):
            mesh.forward(state, dt, stage)
        assert mesh.check_redo(state) == 0, f"rejected cycle {cycle}"
    for b, s in zip(blocks, state):
        sl = tuple(b.part((0, 0, 0), False)[1:])
        rho = s["hydro_u"][kIDN][sl]
        v = (s["hydro_u"][1:4][(slice(None),) + sl] / rho).abs().max().item() / math.sqrt(1.4 * G * H0)
        q = (s["scalar_s"][0][sl] / rho - .01).abs().max().item()
        assert v <= 2.e-12 and q <= 2.e-13, f"max|v|/c_s {v}, tracer ratio error {q}"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(125)
    check(a.device)
