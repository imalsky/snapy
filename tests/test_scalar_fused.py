#!/usr/bin/env python3
"""The fused passive-scalar path (SNAP_SCALAR_FUSED=1, CUDA, float64) gives the same bits as the default path.

Each case runs NSTEP rk3 steps twice, once per switch value, in child processes (the switch is read once
per process), and compares a hash of every block's hydro_u and scalar_s:
  sphere       six gnomonic cubed-sphere panels in one process: WENO5, implicit-scheme 9, eos limiter,
               face-flux covariance with its centroid part, tracers over many decades
  sphere_cov0  the same with SNAP_FLUX_COVARIANCE=0 (no covariance row)
  sphere_ub    the same with an upper bound on the tracers (its second limiter stays in libtorch)
  sphere_plm   PLM tracer reconstruction (the WENO5 kernel is not used; the other kernels are)
  box          a Cartesian 3-D box, periodic sides (covariance without a centroid part)

  python test_scalar_fused.py
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile

import torch
import yaml

NSTEP, NVAR = 4, 6
CASES = ("sphere", "sphere_cov0", "sphere_ub", "sphere_plm", "box")


def config(case):
    sphere = case.startswith("sphere")
    if sphere:
        geometry = {"type": "gnomonic-equiangle",
                    "bounds": {"x1min": 1.6628e7, "x1max": 1.7128e7, "x2min_pi": -0.25, "x2max_pi": 0.25,
                               "x3min_pi": -0.25, "x3max_pi": 0.25},
                    "cells": {"nx1": 12, "nx2": 8, "nx3": 8, "nghost": 3}}
        lateral = "custom"
    else:
        geometry = {"type": "cartesian",
                    "bounds": {"x1min": 0., "x1max": 1.e4, "x2min": 0., "x2max": 2.e4, "x3min": 0., "x3max": 2.e4},
                    "cells": {"nx1": 12, "nx2": 10, "nx3": 8, "nghost": 3}}
        lateral = "periodic"
    cfg = {
        "reference-state": {"Tref": 0.0, "Pref": 1.0e5},
        "species": [{"name": "dry", "composition": {"H": 2}, "cv_R": 2.5}],
        "geometry": geometry,
        "dynamics": {"equation-of-state": {"type": "ideal-moist", "density-floor": 1.0e-10,
                                           "pressure-floor": 1.0e-10, "limiter": True, "max-iter": 30,
                                           "ftol": 1.0e-6},
                     "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": True},
                                     "horizontal": {"type": "weno5", "scale": True, "shock": True}},
                     "riemann-solver": {"type": "hllc"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": lateral, "x2-outer": lateral,
                                            "x3-inner": lateral, "x3-outer": lateral}},
        "integration": {"type": "rk3", "cfl": 0.45, "implicit-scheme": 9 if sphere else 0, "nlim": -1,
                        "tlim": 1.0e12, "ncycle_out": 0},
        "forcing": {"const-gravity": {"grav1": -13.5}},
        "scalar": {"nvar": NVAR, "riemann-solver": {"type": "upwind"}},
    }
    if sphere:
        cfg["distribute"] = {"layout": "cubed-sphere", "nb2": 1, "nb3": 1, "blocks_per_process": 6,
                             "verbose": False}
        cfg["forcing"]["coriolis"] = {"type": "xyz", "omega1": 2.2e-6}
    if case == "sphere_ub":
        cfg["scalar"]["upper-bound"] = 1.0
    if case == "sphere_plm":
        cfg["scalar"]["reconstruct"] = {"type": "plm", "scale": False, "shock": True}
    return cfg


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def run(case):
    import snapy
    from snapy import Mesh, MeshOptions, kIDN, kIPR, kIV1, kIV2, kIV3
    cfg = config(case)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        options = MeshOptions.from_yaml(tmp)
    finally:
        os.unlink(tmp)
    sphere = case.startswith("sphere")
    if sphere:
        options.blocks_per_process(6)
        options.set_local_horizontal_cells(8, 8)
    mesh = Mesh(options)
    blocks = list(mesh.blocks)
    dev = torch.device("cuda")
    for b in blocks:
        b.to(dev)
    mu = blocks[0].module("hydro.eos").species_weight()
    mu = float(mu if isinstance(mu, float) else mu[0])
    Rd, T0, g = 8.314462618 / mu, 600.0, 13.5
    x1min = cfg["geometry"]["bounds"]["x1min"]
    variables = []
    for ib, b in enumerate(blocks):
        coord = b.module("coord")
        x1v = coord.buffer("x1v")
        beta, alpha = torch.meshgrid(coord.buffer("x3v"), coord.buffer("x2v"), indexing="ij")
        bufs = dict(b.named_buffers())
        w = bufs["hydro.D"].clone().zero_()
        P = 1.0e6 * torch.exp(-(x1v - x1min) * g / (Rd * T0))
        w[kIDN] = (P / (Rd * T0)).expand_as(w[kIDN])
        w[kIPR] = P.expand_as(w[kIPR])
        gen = torch.Generator(device="cpu").manual_seed(7 + ib)
        w[kIV1] = 1.0e-3 * torch.randn(tuple(w[kIV1].shape), generator=gen, dtype=torch.float64).to(dev)
        if sphere:
            face_id = int(b.get_layout().loc_of(ib)[2])
            lon, lat = snapy.coord.cs_ab_to_lonlat(snapy.coord.get_cs_face_name(face_id), alpha, beta)
            vel = torch.zeros((3,) + tuple(alpha.shape), dtype=torch.float64, device=dev)
            vel[2] = 50.0 * torch.cos(lat)
            snapy.coord.cs_sph_to_contra_(vel, alpha, beta, face_id)
            w[kIV2] = vel[1].unsqueeze(-1)
            w[kIV3] = vel[2].unsqueeze(-1)
            shape = (1.0 + 0.5 * torch.sin(3 * lon) * torch.cos(lat)).unsqueeze(-1)
        else:
            w[kIV2] = 20.0
            w[kIV3] = -15.0
            shape = (1.0 + 0.5 * torch.sin(alpha / 3.0e3) * torch.cos(beta / 4.0e3)).unsqueeze(-1)
        r = bufs["scalar.D"].clone().zero_()
        for k in range(NVAR):
            # mixing ratios over many decades, one tracer exactly zero, one with a sharp front
            r[k] = 10.0 ** (-6 * k) * shape * (1.0 + 0.3 * torch.sin(x1v / 7.0e4 + k))
        r[2] = 0.0
        r[3] = torch.where(x1v > x1v[x1v.numel() // 2], 1.0e-9, 1.0e-30).expand_as(r[3])
        variables.append({"hydro_w": w, "scalar_r": r})
    variables, _ = mesh.initialize(variables)
    stages = len(blocks[0].module("intg").stages)
    dt = 0.9 * float(mesh.max_time_step(variables))
    for c in range(1, NSTEP + 1):
        mesh.set_cycle(c)
        for st in range(stages):
            mesh.forward(variables, dt, st)
        err = mesh.check_redo(variables)
        assert err == 0, f"redo request {err}"
    return {"case": case, "fused": os.environ.get("SNAP_SCALAR_FUSED", "0"),
            "hydro_u": [digest(v["hydro_u"]) for v in variables],
            "scalar_s": [digest(v["scalar_s"]) for v in variables],
            "finite": bool(all(torch.isfinite(v["scalar_s"]).all() for v in variables))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None, choices=CASES)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available")
        sys.exit(125)
    torch.set_default_dtype(torch.float64)
    if a.child:
        print(json.dumps(run(a.child)), flush=True)
        return
    failures = []
    for case in CASES:
        res = {}
        for fused in ("0", "1"):
            env = dict(os.environ)
            env["SNAP_SCALAR_FUSED"] = fused
            if case == "sphere_cov0":
                env["SNAP_FLUX_COVARIANCE"] = "0"
            else:
                env.pop("SNAP_FLUX_COVARIANCE", None)
            out = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", case], env=env,
                                 capture_output=True, text=True)
            rows = [l for l in out.stdout.splitlines() if l.startswith("{")]
            if out.returncode or not rows:
                print(out.stderr[-3000:], flush=True)
                failures.append(f"{case} fused={fused}: child exit {out.returncode}")
                continue
            res[fused] = json.loads(rows[-1])
        if len(res) < 2:
            continue
        same = res["0"]["hydro_u"] == res["1"]["hydro_u"] and res["0"]["scalar_s"] == res["1"]["scalar_s"]
        print(f"{case:12s} fused vs default: {'bitwise identical' if same else 'DIFFERENT'}", flush=True)
        if not same:
            failures.append(f"{case}: fused and default paths differ")
        if not (res["0"]["finite"] and res["1"]["finite"]):
            failures.append(f"{case}: non-finite tracers")
    for failure in failures:
        print("FAIL", failure)
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
