#!/usr/bin/env python3
"""Tracer and species ratios under the species positivity limiter (#338).
A passive tracer r = s / rho_dry and a vapour fraction q that start uniform must stay uniform where
the limiter fires (eos limiter on, or the covariance rows with it off), and neither may go negative
where the dry air is positive. The limiter used to scale the tracer and species flux by the donor's
theta and leave the unlimited dry mass flux alone; the relative rule now lets them drain as far as
the dry air, but past their content only as far as the poorer neighbour repays.
Arms (the value each failed at, main fb003523 unless noted):
  uniform nx2=4, nx2=nx3=4     8-cell stretched column, bump, x1 Courant 15: max|r/r0-1| 1.068
  uniform nx2=1                the same column alone (no limiter): round-off
  one-cell 5 / one-hole 6      non-negativity beside an empty neighbour
  limiter C30 / C100           nx1 = 32, eos limiter on, bump in one column: 4.401e-02 / 7.055e-03
  limiter one-cell / one-hole  non-negativity on that deck
  settling, limiter off / on   a cell settling drains exports at most its content (629a3850: +1.6e-03)
  sideways x2 / x3 drain       non-negative where rho_dry > 0 (629a3850: tracer -7.9e-04)
python test_tracer_ratio.py [--device cuda] [--scheme 0|9]"""
import math, os, sys, tempfile, torch, yaml
from pathlib import Path
os.environ["SNAP_FLUX_COVARIANCE"] = "1"  # the nx2 > 1 arms need the covariance rows
import snapy
from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR, kIV1

DEV = "cuda" if "cuda" in sys.argv else "cpu"
SCHEME = int(sys.argv[sys.argv.index("--scheme") + 1]) if "--scheme" in sys.argv else 1
ZTOP, T0, RG, G, R0 = 3.2e4, 320., 3900., 10., 1e-3


def build(card):
    with tempfile.TemporaryDirectory() as wd:
        p = os.path.join(wd, "card.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(card, f)
        block = MeshBlock(MeshBlockOptions.from_yaml(p))
    block.to(torch.device(DEV))
    return block


def column(nx1, nx2, nx3=1, limiter=False, x2max=4.e6, amp=1., onecol=False, courant=15., q=1e-4, r=R0):
    """Isothermal column on a stretched x1 grid (snapy.balance_column), a Gaussian pressure bump
    (in every column, or in the first only), ratios q and r per x1 cell; returns block, state, dt"""
    weno = {"type": "weno5", "scale": True, "shock": True}
    block = build({
        "reference-state": {"Tref": 0., "Pref": 1.e5},
        "species": [{"name": "dry", "composition": {"H": 1.6667, "He": 0.16667}, "cv_R": 2.5},
                    {"name": "vapor", "composition": {"H": 2, "O": 1}, "cv_R": 2.5, "u0_R": 0.},
                    {"name": "cloud", "composition": {"H": 2, "O": 1}, "cv_R": 9.0, "u0_R": -3430.}],
        "reactions": [{"equation": "vapor <=> cloud", "type": "nucleation",
                       "rate-constant": {"formula": "h2o_ideal"}}],
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0., "x1max": ZTOP, "x2min": 0., "x2max": x2max,
                                "x3min": 0., "x3max": 4.e6 if nx3 > 1 else 1.e3},
                     "cells": {"nx1": nx1, "nx2": nx2, "nx3": nx3, "nghost": 3,
                               "x1rat": 4. ** (1. / (nx1 - 1))}},
        "distribute": {"layout": "slab", "nb2": 1, "nb3": 1, "verbose": False},
        "dynamics": {"equation-of-state": {"type": "ideal-moist", "density-floor": 1.e-10,
                                           "pressure-floor": 1.e-10, "limiter": limiter},
                     "reconstruct": {"vertical": weno, "horizontal": weno},
                     "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                               **{f"x{d}-{s}": "periodic" for d in (2, 3) for s in ("inner", "outer")}}},
        "integration": {"type": "rk3", "cfl": 0.9, "implicit-scheme": SCHEME, "nlim": -1,
                        "tlim": 1.e9, "ncycle_out": 0},
        "forcing": {"const-gravity": {"grav1": -G}},
        "scalar": {"nvar": 1, "names": ["tracer"], "reconstruct": {"type": "weno5", "shock": False}},
    })
    f = block.module("coord").buffer("x1f").double().cpu()[3:3 + nx1 + 1]
    h, hs = f[1:] - f[:-1], RG * T0 / G
    pbar = 1.e5 * hs * (torch.exp(-f[:-1] / hs) - torch.exp(-f[1:] / hs)) / h  # exact cell means
    col = torch.zeros((5, 1, 1, nx1))
    col[0, 0, 0], col[4, 0, 0] = pbar / (RG * T0), pbar
    wb, _, _ = snapy.balance_column(col, h, G, wall_clamp=True, rtol=1.e-13, max_iter=400, stretched=True)
    wb = wb.to(DEV)
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    for k, src in ((kIDN, 0), (kIPR, 4)):
        w[k][..., 3:3 + nx1] = wb[src]
        w[k][..., :3] = wb[src][..., :1]
        w[k][..., 3 + nx1:] = wb[src][..., -1:]
    ext = lambda p: torch.cat([p[:1].repeat(3), p, p[-1:].repeat(3)])  # ghosts take the edge values
    prof = lambda a: ext(torch.as_tensor(a, dtype=w.dtype, device=w.device).reshape(-1).expand(nx1))
    w[kICY] = prof(q)
    mask = torch.zeros(w.shape[2], dtype=w.dtype, device=w.device)
    mask[3 if onecol else slice(None)] = 1.
    z = block.module("coord").buffer("x1v").double()
    w[kIPR] *= 1. + amp * mask.view(1, -1, 1) * torch.exp(-((z - 0.5 * ZTOP) / 2.e3) ** 2 / 2).view(1, 1, -1)
    v, _ = block.initialize({"hydro_w": w, "scalar_r": torch.zeros_like(w[:1]) + prof(r)})
    return block, v, courant * float(h.min()) / math.sqrt(1.4 * RG * T0)


def step(block, v, dt, q0):  # one rk3 step: max|r/r0-1|, max|q/q0-1|, min s, min rho_v, stage-0 faces
    nx2, nx3 = v["hydro_u"].shape[2], v["hydro_u"].shape[1]
    I = (slice(3, -3) if nx3 > 1 else slice(None), slice(3, -3) if nx2 > 1 else slice(None), slice(3, -3))
    B, dr, dq, ls, lq, face = dict(block.named_buffers()), 0., 0., math.inf, math.inf, None
    for stage in range(3):
        block.forward(v, dt, stage)
        if stage == 0:
            Fr, Fs = B["hydro.F1"][kIDN][I[:2]][..., 3:-2], B["scalar.F1"][0][I[:2]][..., 3:-2]
            face = Fs[Fr != 0] / (R0 * Fr[Fr != 0])
        u, s = v["hydro_u"][(slice(None),) + I], v["scalar_s"][0][I]
        dr = max(dr, float((s / u[kIDN] / R0 - 1).abs().max()))
        dq = max(dq, float((u[kICY] / (u[kIDN] + u[kICY:].sum(0)) / q0 - 1).abs().max()))
        ls, lq = min(ls, float(s.min())), min(lq, float(u[kICY].min()))
    return dr, dq, ls, lq, float((face - 1).abs().max())


def one(n, k, hole):  # one cell k filled in an empty column, or one empty cell k in a full one
    p = torch.full((n,), 1. if hole else 0., dtype=torch.float64)
    p[k] = 0. if hole else 1.
    return p


def uniform(nx2, nx3=1):
    dr, dq, _, _, df = step(*column(8, nx2, nx3), 1e-4)
    return f"max|r/r0-1| = {dr:.3e}, max|q/q0-1| = {dq:.3e}, stage-0 x1 F_s/(r0 F_rho) off 1 by {df:.1e}", \
        max(dr, dq, df) <= 1e-12


def holes(k, hole, nx1=8, q0=1e-4, **kw):
    p = one(nx1, k, hole)
    _, _, ls, lq, _ = step(*column(nx1, 4, q=q0 * p, r=R0 * p, **kw), q0)
    return f"min s = {ls:.3e}, min rho_v = {lq:.3e}", ls >= -1e-12 * R0 and lq >= -1e-12 * q0


LIM = dict(limiter=True, x2max=1.6e6, onecol=True, amp=0.55, courant=30.)  # the C30 arm's deck


def limiter(courant, amp):
    block, v, dt = column(32, 4, limiter=True, x2max=1.6e6, amp=amp, onecol=True, courant=courant, q=5e-3)
    dr, dq, _, _, _ = step(block, v, dt, 5e-3)
    redo = block.check_redo(v)
    return f"max|r/r0-1| = {dr:.3e}, max|q/q0-1| = {dq:.3e}, redo = {redo}", max(dr, dq) <= 1e-12 and redo == 0


def carry(edit):  # the six-cell column of test_flux_positivity_carry.yaml, nx2 = 4
    card = yaml.safe_load(open(Path(__file__).resolve().parent / "test_flux_positivity_carry.yaml"))
    card["geometry"]["cells"]["nx2"], card["geometry"]["bounds"]["x2max"] = 4, 4.
    card["boundary-condition"]["external"].update(
        {f"x{d}-{s}": "periodic" for d in (2, 3) for s in ("inner", "outer")})
    edit(card)
    block = build(card)
    return block, dict(block.named_buffers())["hydro.D"].clone().zero_()


def settling(lim):  # 8 cells rising at 2 m/s, the cloud settling at 1 m/s and falling off 0.9 per cell
    def edit(card):
        card["geometry"]["bounds"]["x1max"], card["geometry"]["cells"]["nx1"] = 8., 8
        card["dynamics"]["equation-of-state"]["limiter"] = lim
        card["sedimentation"] = {"radius": {}, "density": {}, "const-vsed": {"cloud": -1.}}
    block, w = carry(edit)
    w[kIDN], w[kIPR], w[kIV1], w[kICY] = 1., 1.e5, 2., 0.01
    for c in range(8):
        w[kICY + 1][..., 2 + c] = 0.02 * 0.9 ** c
    v, _ = block.initialize({"hydro_w": w})
    u = v["hydro_u"].clone()
    du = block.module("hydro").forward(1., v["hydro_u"], v)  # dt = dx = dx2 = 1
    B = dict(block.named_buffers())
    F1, F2 = B["hydro.F1"][kICY + 1], B["hydro.F2"][kICY + 1]
    I, J = slice(3, 10), slice(2, -2)  # every cell above the bottom one: settling leaves its lower face
    out = (F1[..., 4:11].clamp_min(0.) + (-F1[..., I]).clamp_min(0.))[:, J]
    out = out + F2[:, 3:-1, I].clamp_min(0.) + (-F2[:, J, I]).clamp_min(0.)
    excess, low = float((out - u[kICY + 1][:, J, I]).max()), float((u + du)[kICY + 1][:, J, I].min())
    return excess, low, float(F1[:, J, 2:11].sum())


def sideways(nx3, lim):  # ratios alternate 1, 0.1 along x2 (x3), wind along it at dt = dx
    def edit(card):
        card["geometry"]["bounds"]["x3max"], card["geometry"]["cells"]["nx3"] = (4., 4) if nx3 > 1 else (1., 1)
        card["dynamics"]["equation-of-state"]["limiter"] = lim
        card["scalar"] = {"nvar": 1, "names": ["tracer"], "reconstruct": {"type": "dc"}}
    block, w = carry(edit)
    w[kIDN], w[kIPR], w[kIV1 + (2 if nx3 > 1 else 1)] = 1., 1.e5, 2.
    alt = torch.tensor([0.1, 1.] * 4, dtype=w.dtype, device=w.device)
    prof = alt.view(-1, 1, 1) if nx3 > 1 else alt.view(1, -1, 1)
    w[kICY], w[kICY + 1] = 0.01 * prof, 0.
    v, _ = block.initialize({"hydro_w": w, "scalar_r": (1e-3 * prof).expand(w.shape[1:]).unsqueeze(0).clone()})
    block.forward(v, 1., 0)
    I = (slice(2, -2) if nx3 > 1 else slice(None), slice(2, -2), slice(2, -2))
    ok = v["hydro_u"][kIDN][I] > 0
    s, y = float(v["scalar_s"][0][I][ok].min()), float(v["hydro_u"][kICY][I][ok].min())
    return f"min tracer {s:+.3e}, min vapour {y:+.3e} (where rho_dry > 0)", s >= 0. and y >= 0.


if __name__ == "__main__":
    if DEV == "cuda" and not torch.cuda.is_available(): sys.exit(125)  # no visible GPU: skip
    arms = [("uniform nx2=4", lambda: uniform(4)), ("uniform nx2=nx3=4", lambda: uniform(4, 4)),
            ("uniform nx2=1", lambda: uniform(1)),
            ("one-cell 5", lambda: holes(5, False)), ("one-hole 6", lambda: holes(6, True)),
            ("limiter C30", lambda: limiter(30., 0.55)), ("limiter C100", lambda: limiter(100., 0.18)),
            ("limiter one-cell 22", lambda: holes(22, False, 32, 5e-3, **LIM)),
            ("limiter one-hole 22", lambda: holes(22, True, 32, 5e-3, **LIM))]
    bad = False
    for name, arm in arms:
        msg, ok = arm()
        print(f"{DEV} scheme {SCHEME} {name}: {msg}{'' if ok else '  FAIL'}")
        bad |= not ok
    (e0, l0, f0), (e1, l1, f1) = settling(False), settling(True)
    ok = max(e0, e1) <= 1e-12 * 0.02 and min(l0, l1) >= 0. and abs(f0 - f1) <= 1e-14 * abs(f1)
    print(f"{DEV} settling: max(cloud export - content) {e0:+.3e} (limiter off) {e1:+.3e} (on), "
          f"min update {min(l0, l1):+.3e}, cloud x1 flux sum {f0:.17g} / {f1:.17g}{'' if ok else '  FAIL'}")
    bad |= not ok
    for nx3 in (1, 4):
        for lim in (False, True):
            msg, ok = sideways(nx3, lim)
            print(f"{DEV} sideways {'x3' if nx3 > 1 else 'x2'} drain, limiter {'on ' if lim else 'off'}: {msg}"
                  f"{'' if ok else '  FAIL'}")
            bad |= not ok
    sys.exit(1 if bad else 0)
