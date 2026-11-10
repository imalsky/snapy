#!/usr/bin/env python3
"""#280: x1 WENO5/cp5 on r^2-dr cell averages is fifth order on a spherical-polar deck."""
import math, os, sys, tempfile
import numpy as np, torch, yaml, snapy

DEV = "cuda" if "--device" in sys.argv and "cuda" in sys.argv else "cpu"
if DEV == "cuda" and not torch.cuda.is_available(): sys.exit(125)
GX, GW = np.polynomial.legendre.leggauss(12)
f = lambda r: 1. + 0.5 * np.sin(4. * r) * np.exp(-0.5 * r)
PLAIN = "--x1-centroid" in sys.argv  # the switch on a stretched x1: plain means in
if PLAIN: os.environ["SNAP_X1_CENTROID_EXACT"] = "1"


def error(kind, n):  # max face error of the deck's recon1 on exact r^2 averages
    rc = {"type": kind, "scale": False, "shock": False}
    cfg = {"reference-state": {"Tref": 300., "Pref": 1.e5}, "species": [{"name": "dry", "composition": {"H": 1.5, "He": .15}, "cv_R": 2.5}],
           "geometry": {"type": "spherical-polar", "bounds": {"x1min": 1., "x1max": 2., "x2min": 1., "x2max": 2., "x3min": 0., "x3max": 1.},
                        "cells": {"nx1": n, "nx2": 1, "nx3": 1, "nghost": 3, **({"x1rat": 1.058 ** (32. / n)} if PLAIN else {})}},
           "dynamics": {"equation-of-state": {"type": "ideal-gas"}, "reconstruct": {"vertical": rc, "horizontal": rc}, "riemann-solver": {"type": "lmars"}},
           "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting"}}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fp:
        yaml.safe_dump(cfg, fp)
    b = snapy.Mesh(snapy.MeshOptions.from_yaml(fp.name)).blocks[0]
    os.unlink(fp.name)
    b.to(torch.device(DEV), torch.float64)
    xf = b.buffer("coord.x1f").cpu().double().numpy()
    r = 0.5 * (xf[:-1] + xf[1:])[:, None] + 0.5 * np.diff(xf)[:, None] * GX[None, :]
    wr = GW[None, :] * (1. if PLAIN else r * r)
    w = torch.tensor((wr * f(r)).sum(1) / wr.sum(1)).view(1, 1, 1, -1).repeat(5, 1, 1, 1).to(DEV)
    out, i = b.module("hydro.recon1").forward(w, 3, False).cpu(), np.arange(6, n + 1)  # three faces off each wall
    return max(np.abs(out[s][0, 0, 0].numpy()[i] - f(xf[i])).max() for s in (0, 1))


fails = []
for kind in ("weno5", "cp5"):
    e = [error(kind, n) for n in (32, 64, 128)]
    o = [math.log2(e[0] / e[1]), math.log2(e[1] / e[2])]
    print(f"{kind}: err {e[0]:.2e} {e[1]:.2e} {e[2]:.2e} orders {o[0]:.2f} {o[1]:.2f}")
    fails += [f"{kind} order {o[1]:.2f} < {4. if PLAIN else 4.5}"] * (o[1] < (4. if PLAIN else 4.5))
print("FAIL:", fails) if fails else print("### radial order test passed. ###")
sys.exit(1 if fails else 0)
