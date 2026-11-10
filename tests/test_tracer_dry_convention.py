#!/usr/bin/env python3
"""Passive tracers are per DRY air: scalar_s = hydro_u[IDN] * r, transported by the dry mass
flux. A tracer that starts uniform must therefore stay uniform (to round-off) under every
operator that moves or converts dry air. Three arms on a warm unsaturated moist column:

  init     MeshBlock.initialize() builds scalar_s from the DRY density, not the total.
  implicit the vertical implicit correction moves tracers with the dry face transfer, not
           the total one (it over-moved them by 1/(1 - sum y)).
  relax    relax-bot-comp converts dry air to vapour in the bottom cell; the tracer riding
           that dry air goes with it (r was left to drift).

Each arm also checks that its operator actually did something large (an overestimate of the old
error >= 1000x the tolerance; the measured old error was ~40x smaller than that estimate), so a pass
is never vacuous. A combined arm runs production's scheme 9 with relax-bot-comp and the limiter on.

Limit: a UNIFORM tracer checks that tracer and dry air stay consistent; it cannot see a wrong donor
or upwind choice in tracer transport.
  donor    one tracer cell under a pressure bump: the implicit transfer must not overdraw the empty donor.
  through  a uniform tracer in one column, dt x10: the solve moves over a cell mass through a cell,
           so a donor limit must credit the inflow from below (r stays uniform).

  python test_tracer_dry_convention.py [--device cuda] [--yaml PATH]
"""
import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

R0 = 1.0e-3
TOL = 1.0e-12


def make_block(config, device):
    from snapy import MeshBlock, MeshBlockOptions

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))
    return block


def initial_state(block, config, moving):
    from snapy import kICY, kIDN, kIPR, kIV1

    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    ng = config["geometry"]["cells"]["nghost"]
    nx1 = config["geometry"]["cells"]["nx1"]
    x1max = float(config["geometry"]["bounds"]["x1max"])
    dz = x1max / nx1
    z = (torch.arange(w.size(-1), dtype=torch.float64, device=w.device) - ng + 0.5) * dz
    T, R = 320.0, 3900.0
    p = 1.0e5 * torch.exp(-z * 10.0 / (R * T))
    w[kIPR] = p.view(1, 1, -1)
    w[kIDN] = (p / (R * T)).view(1, 1, -1)
    w[kICY] = 0.02  # vapour mass fraction of the total; cloud stays 0
    if moving:
        w[kIV1] = (5.0 * torch.sin(math.pi * z / x1max)).view(1, 1, -1)
        # sheared horizontal wind: the full (scheme 9) VIC couples it, the partial one does not
        w[kIV1 + 1] = (10.0 * torch.cos(math.pi * z / x1max)).view(1, 1, -1)
    r = torch.full((1,) + tuple(w.shape[1:]), R0, dtype=w.dtype, device=w.device)
    return w, r


def interior(config):
    ng = config["geometry"]["cells"]["nghost"]
    return (slice(None), slice(None), slice(ng, -ng), slice(ng, -ng))


def run(config, device, moving, nlim):
    from snapy import kIDN

    block = make_block(config, device)
    w, r = initial_state(block, config, moving)
    v, _ = block.initialize({"hydro_w": w, "scalar_r": r})
    I = interior(config)
    bufs = dict(block.named_buffers())
    mass_corr = next((b for k, b in bufs.items() if k.endswith("mass_corr")), None)
    # largest relative over-transport the old implicit tracer update (total M instead of
    # the dry transfer) would have applied in one stage, summed over stages
    implicit_bug_scale = 0.0
    # departures are measured from the post-initialisation r, so each arm tests only its
    # own operator (not the initialisation, which the init arm covers)
    r_init = v["scalar_s"][I] / v["hydro_u"][kIDN][I[1:]]
    for _ in range(nlim):
        dt = block.max_time_step(v)
        for stage in range(len(block.intg.stages)):
            block.forward(v, dt, stage)
            if mass_corr is not None and mass_corr.numel() > 0:
                from snapy import kIV1
                g = config["geometry"]
                vol = (float(g["bounds"]["x1max"]) / g["cells"]["nx1"]) * (
                    float(g["bounds"]["x2max"]) / g["cells"]["nx2"]) * (
                    float(g["bounds"]["x3max"]) / g["cells"]["nx3"])
                excess = (mass_corr[kIV1] - mass_corr[kIV1 + 1]).abs()[I[1:]]
                implicit_bug_scale += float((excess / (v["hydro_u"][kIDN][I[1:]] * vol)).max())
    dev = float((v["scalar_s"][I] / (v["hydro_u"][kIDN][I[1:]] * r_init) - 1.0).abs().max())
    # the primitive the runners read must agree with the conserved pair
    cached = float((v["scalar_r"][I] / r_init - 1.0).abs().max())
    dev = max(dev, cached)
    return v, dev, implicit_bug_scale


def main():
    from snapy import kIDN

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--yaml", default=str(Path(__file__).resolve().parent / "test_tracer_dry_convention.yaml"))
    args = parser.parse_args()
    with open(args.yaml) as f:
        base = yaml.safe_load(f)
    failures = []

    # ---- init: scalar_s from dry density ----
    cfg = yaml.safe_load(yaml.safe_dump(base))
    block = make_block(cfg, args.device)
    w, r = initial_state(block, cfg, moving=False)
    v, _ = block.initialize({"hydro_w": w, "scalar_r": r})
    I = interior(cfg)
    s = v["scalar_s"][I]
    dry_err = float((s / (v["hydro_u"][kIDN][I[1:]] * R0) - 1).abs().max())
    tot_err = float((s / (v["hydro_w"][kIDN][I[1:]] * R0) - 1).abs().max())
    loading = float((v["hydro_w"][kIDN][I[1:]] / v["hydro_u"][kIDN][I[1:]] - 1).abs().max())
    print(f"init    : |s/(rho_dry r)-1|={dry_err:.3e}  |s/(rho_tot r)-1|={tot_err:.3e}  "
          f"rho_tot/rho_dry-1={loading:.3e}")
    if not dry_err < TOL:
        failures.append(f"init: scalar_s is not rho_dry*r (err {dry_err:.3e})")
    if not loading > 1000 * TOL:
        failures.append(
            f"init: dry and total density indistinguishable ({loading:.3e}); test cannot bite")

    # ---- implicit: tracer moved with the dry face transfer (partial and full VIC) ----
    finals = {}
    for scheme in (1, 9):
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["integration"]["implicit-scheme"] = scheme
        vs, dev, bug_scale = run(cfg, args.device, moving=True, nlim=cfg["integration"]["nlim"])
        finals[scheme] = vs["hydro_u"].clone()
        print(f"implicit{scheme}: max|r/r_init-1|={dev:.3e}"
              f"  old-code over-transport estimate={bug_scale:.3e}")
        if not dev < TOL:
            failures.append(f"implicit scheme {scheme}: uniform tracer departed by {dev:.3e}")
        if not bug_scale > 1000 * TOL:
            failures.append(
                f"implicit scheme {scheme}: the solve moved too little mass ({bug_scale:.3e}) "
                "or no mass_corr buffer was found; test cannot bite")

    if torch.equal(finals[1], finals[9]):
        failures.append(
            "implicit: schemes 1 and 9 gave bitwise-identical states; scheme 9 not exercised")

    # ---- relax: relax-bot-comp carries the tracer with the dry air it converts, both ways ----
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["integration"]["implicit-scheme"] = 0
    nlim = cfg["integration"]["nlim"]
    v0, _, _ = run(cfg, args.device, moving=False, nlim=nlim)
    I = interior(cfg)
    for label, xfrac in (("moisten", 0.2), ("dry", 0.0)):
        c = yaml.safe_load(yaml.safe_dump(cfg))
        c["forcing"]["relax-bot-comp"] = {"tau": 10.0, "species": ["vapor"], "xfrac": [xfrac]}
        v1, dev, _ = run(c, args.device, moving=False, nlim=nlim)
        forced = float(
            ((v1["hydro_u"][kIDN] - v0["hydro_u"][kIDN])[I[1:]] / v0["hydro_u"][kIDN][I[1:]])
            .abs()
            .max()
        )
        print(f"relax-{label}: max|r/r_init-1|={dev:.3e}"
              f"  dry-density change due to the forcing={forced:.3e}")
        if not dev < TOL:
            failures.append(f"relax ({label}): uniform tracer departed by {dev:.3e}")
        if not forced > 1000 * TOL:
            failures.append(
                f"relax ({label}): forcing converted too little dry air ({forced:.3e});"
                f" test cannot bite")

    # ---- combined, production-like: scheme 9 + relax-bot-comp + EOS limiter ----
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["integration"]["implicit-scheme"] = 9
    cfg["dynamics"]["equation-of-state"]["limiter"] = True
    cfg["forcing"]["relax-bot-comp"] = {"tau": 10.0, "species": ["vapor"], "xfrac": [0.2]}
    _, dev, bug_scale = run(cfg, args.device, moving=True, nlim=cfg["integration"]["nlim"])
    print(f"combined: max|r/r_init-1|={dev:.3e}  old-code over-transport estimate={bug_scale:.3e}")
    if not dev < TOL:
        failures.append(
            f"combined (scheme 9 + relax + limiter): uniform tracer departed by {dev:.3e}")

    # ---- donor: the implicit transfer never overdraws a near-empty tracer donor ----
    from snapy import kIPR
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["integration"]["implicit-scheme"] = 1
    cfg["geometry"]["bounds"]["x2max"] = 4.0e4  # a longer dt: up to 0.886 cell masses move
    block = make_block(cfg, args.device)
    w, r = initial_state(block, cfg, moving=False)
    w[kIPR][..., 3 + 14] *= 2.0  # the bump pushes air up through cell 13, which holds no tracer
    r.zero_()
    r[..., 3 + 12] = R0
    v, _ = block.initialize({"hydro_w": w, "scalar_r": r})
    I, s_min = interior(cfg), math.inf
    total0 = float(v["scalar_s"][I].sum())
    for _ in range(2):
        dt = block.max_time_step(v)
        for stage in range(len(block.intg.stages)):
            block.forward(v, dt, stage)
            s_min = min(s_min, float(v["scalar_s"][I].min()))
    total_dev = abs(float(v["scalar_s"][I].sum()) / total0 - 1.0)
    print(f"donor   : min s={s_min:.3e}  |total tracer change|={total_dev:.3e}")
    if not (s_min >= 0.0 and total_dev < TOL):
        failures.append(f"donor: min s {s_min:.3e}, total tracer change {total_dev:.3e}")

    # ---- through: a uniform tracer stays uniform when over a cell mass passes through a cell ----
    from snapy import kIV1
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["integration"]["implicit-scheme"] = 1
    cfg["geometry"]["cells"]["nx2"], cfg["geometry"]["bounds"]["x2max"] = 1, 1.0e4
    block = make_block(cfg, args.device)
    w, r = initial_state(block, cfg, moving=False)
    dz = float(cfg["geometry"]["bounds"]["x1max"]) / cfg["geometry"]["cells"]["nx1"]
    z = (torch.arange(w.size(-1), dtype=torch.float64, device=w.device) - 3 + 0.5) * dz
    w[kIPR] *= (1.0 + 2.0 * torch.exp(-(((z - 16.5 * dz) / (4.0 * dz)) ** 2) / 2)).view(1, 1, -1)
    v, _ = block.initialize({"hydro_w": w, "scalar_r": r})
    I = (slice(None), slice(None), slice(None), slice(3, -3))  # x1 is the only ghosted axis
    M = next(b for k, b in dict(block.named_buffers()).items() if k.endswith("mass_corr"))
    vol = dz * 1.0e4 * float(cfg["geometry"]["bounds"]["x3max"]) / cfg["geometry"]["cells"]["nx3"]
    r0, dt = v["scalar_s"][I] / v["hydro_u"][kIDN][I[1:]], block.max_time_step(v) * 10.0
    dev, moved = 0.0, 0.0
    for stage in range(len(block.intg.stages)):
        cell = v["hydro_u"][kIDN][I[1:]] * vol
        block.forward(v, dt, stage)
        moved = max(moved, float((M[kIV1 + 1][I[1:]].abs() / cell).max()))
        dev = max(dev, float((v["scalar_s"][I] / (v["hydro_u"][kIDN][I[1:]] * r0) - 1).abs().max()))
    print(f"through : max|r/r0-1|={dev:.3e}  max implicit transfer / cell mass={moved:.3e}")
    if not (dev <= 1.0e-14 and moved > 1.0):
        failures.append(f"through: r departed by {dev:.3e} with {moved:.3e} cell masses moved")

    if failures:
        for msg in failures:
            print("FAIL:", msg)
        sys.exit(1)
    print("### tracer dry-convention test passed. ###")


if __name__ == "__main__":
    main()
