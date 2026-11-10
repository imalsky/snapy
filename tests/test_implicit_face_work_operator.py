#!/usr/bin/env python3
"""gravity-work: face with an implicit scheme keeps the #283 tall column at rest and E+PE closed.

The column of test_implicit_gravity_tall_column.py (45 cells of dz = 29.9 km, 11.3
scale heights, reflecting x1 walls, rk3 + weno5 + lmars), put into the scheme's own
discrete hydrostatic balance with snapy.balance_column, run with gravity-work: face
(no fixer) for NSTEP fixed steps at acoustic Courant 65.6, 197 and 657, with vic-full
(implicit-scheme 9) and vic-partial (1), at rest and moving (w = w0 sin(pi z / H),
w0 = 10 m/s, 1 m/s at Courant 657 to keep the advective Courant number below 1):
  every run finite with |E+PE drift| < EPE_TOL (E+PE the interior sum of E + rho g z);
  at rest also max |w| < W_TOL.
With SNAP_GRAVITY_WORK_RADIAL_EXACT on (the default with face work, radial_exact()) the
conserved energy is E + P, P the corrected PE of
docs/derivations/curved-gravity-work-weight.md sec 7, and the drift is measured on it
(E+PE then drifts by the O(dz^2) error of PE itself, ~1e-10 here); the same runs blew up
at Courant 197 and 657 before that switch's work was coupled into the operator (#296).

Face work booked after the implicit solve, outside the operator that linearises the
cell work, blows the column up at Courant 65.6 by step 27 (chengcli/snapy#283). With
the face work inside the operator the column stays near 5e-9 m/s and E+PE drifts by
~1e-14.

  python test_implicit_face_work_operator.py [--device cpu] [--nstep 40]
"""
import argparse
import functools
import math
import os
import sys
import tempfile

import torch
import yaml

GRAV, GAMMA, RD, T0, PS = 23.1, 1.403461, 3515.0, 786.0, 1.0e7
NZ, DZ, NX2 = 45, 29946.8085106, 8
# (dt, w0 of the moving run): acoustic Courant 65.6, 197, 657
RUNGS = ((997.0, 10.0), (3000.0, 10.0), (10000.0, 1.0))
SCHEMES = ((9, "vic-full"), (1, "vic-partial"))
W_TOL = 1.0e-7  # m/s
EPE_TOL = 1.0e-11


def config(scheme, integ="rk3"):
    rgas = 8.31446261815324
    return {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": NZ * DZ, "x2min": 0.0, "x2max": 31415926.53589793,
                                "x3min": 0.0, "x3max": 3926990.8169872416},
                     "cells": {"nx1": NZ, "nx2": NX2, "nx3": 1, "nghost": 3}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": GAMMA, "weight": rgas / RD,
                                  "density-floor": 3.4682e-18, "pressure-floor": 3.4682e-18,
                                  "temperature-floor": 1.e-6, "limiter": True},
            "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                            "horizontal": {"type": "weno5", "scale": True, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": integ, "cfl": 0.5, "implicit-scheme": scheme, "nlim": -1, "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -GRAV, "gravity-work": "face"}},
    }


@functools.lru_cache(maxsize=None)
def radial_exact():
    """SNAP_GRAVITY_WORK_RADIAL_EXACT as hydro.cpp reads it: on unless 0/false/off/no"""
    v = os.environ.get("SNAP_GRAVITY_WORK_RADIAL_EXACT", "1").lower()
    return v not in ("0", "false", "off", "no")


def corrected_pe(rho, x1f, x1v, grav1, spherical=False, centroid=False):
    """per-cell -grav1 sigma^2 s[rho] (times the cell volume by the caller): what P adds
    to PE_d with SNAP_GRAVITY_WORK_RADIAL_EXACT on, rho along the last dimension;
    centroid (gnomonic-equiangle, where x1v is not the r^2 centroid r_c): also
    -grav1 (r_c - x1v) rho (derivation sec 12)"""
    from test_gravity_work_radial_exact import centroid_offset, slope, variance
    pe = -grav1 * variance(x1f, spherical).to(rho) * slope(rho, x1v.to(rho))
    if centroid:
        pe = pe - grav1 * centroid_offset(x1f, x1v).to(rho) * rho
    return pe


def run(scheme, dt, w0, nstep, device, integ="rk3"):
    """Returns (finite, steps run, max |w|, relative E+PE drift)."""
    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(scheme, integ), f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))

    ng = 3
    z = (torch.arange(NZ, dtype=torch.float64) + 0.5) * DZ
    p = PS * torch.exp(-GRAV * z / (RD * T0))
    col = torch.zeros(kIPR + 1, 1, 1, NZ, dtype=torch.float64)
    col[kIDN, 0, 0] = p / (RD * T0)
    col[kIPR, 0, 0] = p
    wb, _, _ = snapy.balance_column(col, torch.full((NZ,), DZ, dtype=torch.float64), GRAV)
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + NZ] = wb[c, 0, 0].to(w)
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ:] = w[c][..., ng + NZ - 1:ng + NZ]
    w[kIV1][..., ng:ng + NZ] = (w0 * torch.sin(math.pi * z / (NZ * DZ))).to(w)
    block_vars, _ = block.initialize({"hydro_w": w})

    interior = (Ellipsis, slice(ng, ng + NX2), slice(ng, ng + NZ))
    zz = z.to(w)

    x1f = torch.arange(NZ + 1, dtype=torch.float64) * DZ
    exact = radial_exact()

    def epe():
        u = block_vars["hydro_u"][interior]
        e = u[kIPR] + u[kIDN] * GRAV * zz
        if exact:
            e = e + corrected_pe(u[kIDN], x1f, zz, -GRAV)
        return e.sum().item()

    e0 = epe()
    nstage = len(block.module("intg").stages)
    wmax = 0.0
    for n in range(1, nstep + 1):
        for stage in range(nstage):
            block.forward(block_vars, dt, stage)
        u = block_vars["hydro_u"][interior]
        if not torch.isfinite(u).all():
            return False, n, wmax, float("nan"), None
        wmax = max(wmax, (u[kIV1] / u[kIDN]).abs().max().item())
    return True, nstep, wmax, (epe() - e0) / abs(e0), u[kIV1] / u[kIDN]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nstep", type=int, default=40)
    args = ap.parse_args()
    if args.device.startswith("cuda") and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 0
    torch.set_default_dtype(torch.float64)

    cs = math.sqrt(GAMMA * RD * T0)
    failures = []
    print("SNAP_GRAVITY_WORK_RADIAL_EXACT %s: E+PE is E + %s" % (
        ("on", "P") if radial_exact() else ("off", "PE_d")), flush=True)
    for scheme, sname in SCHEMES:
        for dt, w0_moving in RUNGS:
            for w0 in (0.0, w0_moving):
                finite, n, wmax, drift, _ = run(scheme, dt, w0, args.nstep, args.device)
                case = "%-11s Courant %5.1f w0 %4.1f" % (sname, cs * dt / DZ, w0)
                print("%s  steps %d  max|w| %.3e m/s  E+PE drift %.2e" % (case, n, wmax, drift))
                if not finite:
                    failures.append("%s: state non-finite at step %d" % (case, n))
                elif w0 == 0. and not wmax < W_TOL:
                    failures.append("%s: max|w| = %.3e m/s (tol %.0e)" % (case, wmax, W_TOL))
                elif not abs(drift) < EPE_TOL:
                    failures.append("%s: E+PE drift %.2e (tol %.0e)" % (case, drift, EPE_TOL))
    # rk3s4 weights the implicit solve by its own stages: no less accurate than rk3
    w = {(g, k): run(9, DZ / cs / k, 1.e-3, 16 * k, args.device, g)[4] for g, k in
         (("rk3", 32), ("rk3", 4), ("rk3", 8), ("rk3s4", 4), ("rk3s4", 8))}
    e = {gk: float((x - w["rk3", 32]).norm() / w["rk3", 32].norm()) for gk, x in w.items()}
    print("w error vs rk3 at Courant 1/32: rk3 %.4e %.4e, rk3s4 %.4e %.4e (Courant 1/4, 1/8)"
          % (e["rk3", 4], e["rk3", 8], e["rk3s4", 4], e["rk3s4", 8]))
    if math.log2(e["rk3s4", 4] / e["rk3s4", 8]) < math.log2(e["rk3", 4] / e["rk3", 8]):
        failures.append("rk3s4 converges slower than rk3 with implicit-scheme 9")
    for f in failures:
        print("FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
