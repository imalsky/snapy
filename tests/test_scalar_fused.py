#!/usr/bin/env python3
"""The fused passive-scalar path (SNAP_SCALAR_FUSED=1, CUDA, float64) against the default path.

Each case runs NSTEP rk3 steps twice, once per switch value, in child processes (the switch is read once
per process). The flow does not see the scalars, so every block's hydro_u must be bit for bit the same. The
fused kernels reproduce the default path's arithmetic except the WENO5 weights, which take fewer divisions:
with WENO5 each scalar must agree to TOL relative to the largest value within RAD cells along each axis (round-off
is relative to the values a cell's stencils combine: next to a sharp front a cell far below the front's height is a
near-cancellation of front-sized terms); with PLM (no WENO5 kernel, every other fused kernel) bit for bit. Cases:
  sphere       six gnomonic cubed-sphere panels in one process: WENO5, implicit-scheme 9, eos limiter,
               face-flux covariance with its centroid part, tracers over many decades
  sphere_cov0  the same with SNAP_FLUX_COVARIANCE=0 (no covariance row)
  sphere_ub    the same with an upper bound on the tracers (its second limiter stays in libtorch)
  sphere_plm   PLM tracer reconstruction (the WENO5 kernel is not used; the other kernels are)
  sphere_cov0_plm  PLM without the covariance row
  box          a Cartesian 3-D box, periodic sides (covariance without a centroid part)
  box_plm      the box with PLM
  box_dry_plm  the box with PLM and a stage forcing that adds and removes dry mass (both arms of the
               scalars' dry-source carry)
  box_x1rat    the box on a stretched x1 grid (x1rat 1.1): x1 has per-face rows, which the fused kernel does not
               read, so x1 must take the default path
The positivity limiter must act (cells with theta < 1) in the PLM cases, the same cells on both paths.

  python test_scalar_fused.py
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict

import numpy as np
import torch
import yaml


class _DrySource(torch.nn.Module):
    """Adds 1e-4 of the dry density in even x1 columns and removes it in odd ones."""

    def forward(self, variables: Dict[str, torch.Tensor], dt: float, stage: int) -> Dict[str, torch.Tensor]:
        rho = variables["hydro_u"][0]
        i = torch.arange(rho.size(-1), dtype=rho.dtype, device=rho.device)
        sign = 1.0 - 2.0 * torch.remainder(i, 2.0)
        du = torch.zeros_like(variables["hydro_u"])
        du[0] = 1.0e-4 * rho * sign
        return {"hydro_du": du}


NSTEP, NVAR, TOL, RAD = 4, 6, 1.0e-12, 4
CASES = (
    "sphere",
    "sphere_cov0",
    "sphere_ub",
    "sphere_plm",
    "sphere_cov0_plm",
    "box",
    "box_plm",
    "box_dry_plm",
    "box_x1rat",
)


def config(case):
    sphere = case.startswith("sphere")
    if sphere:
        geometry = {
            "type": "gnomonic-equiangle",
            "bounds": {
                "x1min": 1.6628e7,
                "x1max": 1.7128e7,
                "x2min_pi": -0.25,
                "x2max_pi": 0.25,
                "x3min_pi": -0.25,
                "x3max_pi": 0.25,
            },
            "cells": {"nx1": 12, "nx2": 8, "nx3": 8, "nghost": 3},
        }
        lateral = "custom"
    else:
        geometry = {
            "type": "cartesian",
            "bounds": {"x1min": 0.0, "x1max": 1.0e4, "x2min": 0.0, "x2max": 2.0e4, "x3min": 0.0, "x3max": 2.0e4},
            "cells": {"nx1": 12, "nx2": 10, "nx3": 8, "nghost": 3},
        }
        lateral = "periodic"
    cfg = {
        "reference-state": {"Tref": 0.0, "Pref": 1.0e5},
        "species": [{"name": "dry", "composition": {"H": 2}, "cv_R": 2.5}],
        "geometry": geometry,
        "dynamics": {
            "equation-of-state": {
                "type": "ideal-moist",
                "density-floor": 1.0e-10,
                "pressure-floor": 1.0e-10,
                "limiter": True,
                "max-iter": 30,
                "ftol": 1.0e-6,
            },
            "reconstruct": {
                "vertical": {"type": "weno5", "scale": True, "shock": True},
                "horizontal": {"type": "weno5", "scale": True, "shock": True},
            },
            "riemann-solver": {"type": "hllc"},
        },
        "boundary-condition": {
            "external": {
                "x1-inner": "reflecting",
                "x1-outer": "reflecting",
                "x2-inner": lateral,
                "x2-outer": lateral,
                "x3-inner": lateral,
                "x3-outer": lateral,
            }
        },
        "integration": {
            "type": "rk3",
            "cfl": 0.45,
            "implicit-scheme": 9 if sphere else 0,
            "nlim": -1,
            "tlim": 1.0e12,
            "ncycle_out": 0,
        },
        "forcing": {"const-gravity": {"grav1": -13.5}},
        "scalar": {"nvar": NVAR, "riemann-solver": {"type": "upwind"}},
    }
    if sphere:
        cfg["distribute"] = {"layout": "cubed-sphere", "nb2": 1, "nb3": 1, "blocks_per_process": 6, "verbose": False}
        cfg["forcing"]["coriolis"] = {"type": "xyz", "omega1": 2.2e-6}
    if case == "sphere_ub":
        cfg["scalar"]["upper-bound"] = 1.0
    if case == "box_x1rat":
        cfg["geometry"]["cells"]["x1rat"] = 1.1
    if case.endswith("_plm"):
        cfg["scalar"]["reconstruct"] = {"type": "plm", "scale": False, "shock": True}
    return cfg


def local_max(x, r=RAD):
    """max |x| over a box of +-r cells along each spatial axis of (block, nc3, nc2, nc1), cut at the edges"""
    m = np.abs(x)
    for ax in (1, 2, 3):
        out = m.copy()
        n = m.shape[ax]
        for s in range(1, min(r, n - 1) + 1):
            lo = [slice(None)] * 4
            hi = [slice(None)] * 4
            lo[ax], hi[ax] = slice(0, n - s), slice(s, n)
            out[tuple(lo)] = np.maximum(out[tuple(lo)], m[tuple(hi)])
            out[tuple(hi)] = np.maximum(out[tuple(hi)], m[tuple(lo)])
        m = out
    return m


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def run(case, save):
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
    if "_dry" in case:
        path = os.path.join(os.path.dirname(os.path.abspath(save)), f"{case}_dry.pt")
        torch.jit.script(_DrySource().eval()).save(path)
        mesh.set_user_stage_forcings([path])
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
    np.save(save, torch.stack([v["scalar_s"] for v in variables]).cpu().numpy())
    hits = sum(int(dict(b.named_buffers())["scalar.positivity_hits"].sum()) for b in blocks)
    return {
        "case": case,
        "fused": os.environ.get("SNAP_SCALAR_FUSED", "0"),
        "positivity_hits": hits,
        "hydro_u": [digest(v["hydro_u"]) for v in variables],
        "finite": bool(all(torch.isfinite(v["scalar_s"]).all() for v in variables)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None, choices=CASES)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available")
        sys.exit(125)
    torch.set_default_dtype(torch.float64)
    if a.child:
        print(json.dumps(run(a.child, a.save)), flush=True)
        return
    failures = []
    limited = 0  # positivity-limited cells in the bit-for-bit cases
    tmpdir = tempfile.mkdtemp(dir=os.getcwd())
    for case in CASES:
        res, arr = {}, {}
        for fused in ("0", "1"):
            env = dict(os.environ)
            env["SNAP_SCALAR_FUSED"] = fused
            if case.startswith("sphere_cov0"):
                env["SNAP_FLUX_COVARIANCE"] = "0"
            else:
                env.pop("SNAP_FLUX_COVARIANCE", None)
            save = os.path.join(tmpdir, f"{case}_{fused}.npy")
            out = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--child", case, "--save", save],
                env=env,
                capture_output=True,
                text=True,
            )
            rows = [line for line in out.stdout.splitlines() if line.startswith("{")]
            if out.returncode or not rows:
                print(out.stderr[-3000:], flush=True)
                failures.append(f"{case} fused={fused}: child exit {out.returncode}")
                continue
            res[fused] = json.loads(rows[-1])
            arr[fused] = np.load(save)
            os.unlink(save)
        if len(res) < 2:
            continue
        if res["0"]["hydro_u"] != res["1"]["hydro_u"]:
            failures.append(f"{case}: the flow differs (hydro_u)")
        if not (res["0"]["finite"] and res["1"]["finite"]):
            failures.append(f"{case}: non-finite tracers")
        a0, a1 = arr["0"], arr["1"]
        h0, h1 = res["0"]["positivity_hits"], res["1"]["positivity_hits"]
        if case.endswith("_plm"):
            same = a0.shape == a1.shape and np.array_equal(a0.view(np.uint64), a1.view(np.uint64))
            print(
                f"{case:16s} fused vs default: {'bitwise identical' if same else 'DIFFERENT'}, "
                f"positivity-limited cells {h0} / {h1}",
                flush=True,
            )
            if not same:
                failures.append(f"{case}: fused and default paths differ")
            if h0 != h1:
                failures.append(f"{case}: positivity-limited cells differ ({h0} / {h1})")
            limited += h1
            continue
        worst = 0.0
        for k in range(NVAR):
            x0, x1 = a0[:, k], a1[:, k]
            err = float((np.abs(x1 - x0) / np.maximum(local_max(x0), 1.0e-320)).max())
            if not err <= TOL:
                failures.append(f"{case}: tracer {k} differs by {err:.3g} of the local largest value (> {TOL})")
            worst = max(worst, err) if err == err else float("nan")
        print(
            f"{case:16s} fused vs default: hydro_u bitwise {res['0']['hydro_u'] == res['1']['hydro_u']}, "
            f"scalars within {worst:.3g} of the local largest value, positivity-limited cells {h0} / {h1}",
            flush=True,
        )
    shutil.rmtree(tmpdir)
    if not limited:
        failures.append("the positivity limiter never acted in a bit-for-bit case, so it was not tested")
    for failure in failures:
        print("FAIL", failure)
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
