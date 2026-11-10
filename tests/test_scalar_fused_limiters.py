#!/usr/bin/env python3
"""The fused passive-scalar path (SNAP_SCALAR_FUSED=1) against the default path on main's tracer-limit decks.

Each arm runs twice in child processes (the switch is read once per process). The flow does not see the
scalars, so hydro_u must be bit for bit the same. With a donor-cell (dc) tracer reconstruction no WENO5
kernel runs and the scalars must be bit for bit the same; with WENO5 they must agree to TOL of the largest
value within RAD cells (test_scalar_fused.py). The stretched ratio column runs WENO5 only. Each fused child must launch the fused kernels its arm uses
(torch profiler) and give exactly the fallback warnings it expects; a default child must launch none. Each
arm must bind what it is there for, on both paths, observed per stage cell by cell (StageLog), and those
records must be the same on both paths (with dc also every stage's fluxes and state, bit for bit):
  ratio      test_tracer_ratio.py's C30 column (32 stretched x1 cells, nx2 4, eos limiter, x1 Courant 30,
             implicit scheme 1): limited cells, and cells whose tracer drains past its content, which only
             the donor-share correction allows; x1 is stretched, so it reconstructs on the default path
  upper      test_scalar_upper_bound_donor.py's column: a tracer at the bound in one cell; from the same
             start with and without the bound, the flux faces the bound's blend moved at the first stage
             where the two differ (all earlier states equal)
  donor      test_tracer_dry_convention.py's donor arm: one tracer cell under a pressure bump; faces where
             the implicit transfer's donor cap binds (WENO5; with dc the cap does not bind on this deck),
             a non-negative tracer and a fixed tracer total
  through    its through arm: over a cell mass moves through a cell in one stage; capped faces, and a
             uniform tracer stays uniform (the cap credits the inflow from below)
  stratified test_scalar_stratified.py's one Scalar.forward on an exponential column (WENO5, 40 and 80
             cells): the fused faces keep its order above 4.5 (dc is not an order test there)

  python test_scalar_fused_limiters.py
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import yaml
from test_scalar_fused import FUSED_KERNEL, digest, local_max, warnings_in

HERE = os.path.dirname(os.path.abspath(__file__))

TOL, ARMS, RECONS = 1.0e-12, ("ratio", "upper", "donor", "through", "stratified"), ("dc", "weno5")
# a stretched x1 reconstructs only with weno5 or cp5, so the ratio column has no dc arm
SKIP = {("ratio", "dc")}
CORE = {"scalar_divergence_kernel", "scalar_average3_kernel"}
TRANSFER = {"scalar_transfer_faces_kernel", "scalar_transfer_add_kernel"}
LIMITER = {"scalar_theta_kernel", "scalar_theta_scale_kernel"}


def expected(arm, recon):
    """the fused kernels an arm must launch, and the fallback warnings it must give"""
    kernels = {
        "ratio": CORE | TRANSFER | LIMITER | {"scalar_lateral_flux_kernel"},
        "upper": CORE | TRANSFER | LIMITER | {"scalar_lateral_flux_kernel"},
        "donor": CORE | TRANSFER | {"scalar_lateral_flux_kernel"},
        "through": CORE | TRANSFER,
        "stratified": {"scalar_divergence_kernel"},
    }[arm]
    if recon == "weno5":
        kernels = kernels | {"scalar_weno5_faces_kernel"}
    warns = {"not WENO5"} if recon == "dc" else ({"x1 rows"} if arm == "ratio" else set())
    return kernels, warns


def profiled(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    names = sorted({m.group(1) for e in prof.key_averages() for m in [FUSED_KERNEL.search(e.key)] if m})
    return out, names


class StageLog:
    """Per-stage observations of a Cartesian block, cell by cell, from its state before and its buffers after
    each stage:
      limited  interior cells whose final tracer outflow reaches their content (theta < 1)
      drained  cells whose tracer outflow exceeds their content: theta alone cannot allow it, only the
               donor-share correction raises theta past it
      capped   x1 faces where the implicit transfer's donor cap binds: P = (M > 0 ? r_below : r) M against
               held = max(s - dt D, 0) vol, swept up each column as in at::native::donor_cap
      fluxes, state  the stage's limited scalar fluxes and scalar_s
    limited and drained mean that only where the positivity limiter runs (eos limiter or covariance rows):
    every arm but through, whose single column has neither."""

    KINDS = ("limited", "drained", "capped", "fluxes", "state")

    def __init__(self, block, v):
        self.block, self.v = block, v
        self.B = dict(block.named_buffers())
        coord = block.module("coord")
        self.dx = {3: coord.buffer("dx1f"), 2: coord.buffer("dx2f"), 1: coord.buffer("dx3f")}
        self.M = next((b for k, b in self.B.items() if k.endswith("mass_corr")), None)
        self.rows = {k: [] for k in self.KINDS}

    def step(self, dt, stage):
        from snapy import kIV1

        s0, r0 = self.v["scalar_s"].clone(), self.v["scalar_r"].clone()
        self.block.forward(self.v, dt, stage)
        I = tuple([slice(None)] + [slice(3, n - 3) if n > 1 else slice(None) for n in s0.shape[1:]])
        out, fluxes = torch.zeros_like(s0), []
        for dim, name in ((3, "scalar.F1"), (2, "scalar.F2"), (1, "scalar.F3")):
            F, n = self.B.get(name), s0.size(dim)
            if F is None or F.numel() == 0 or n == 1:
                continue
            shape = [1, 1, 1, 1]
            shape[dim] = n - 1
            per_volume = (F.narrow(dim, 1, n - 1).clamp_min(0) + (-F.narrow(dim, 0, n - 1)).clamp_min(0)) / self.dx[
                dim
            ].narrow(0, 0, n - 1).view(shape)
            out.narrow(dim, 0, n - 1).add_(per_volume)
            fluxes.append(F.clone().reshape(-1))
        drain, content = (dt * out)[I], s0[I]
        self.rows["limited"].append(((drain > 0) & (drain >= content * (1 - 1e-10))).cpu().numpy())
        self.rows["drained"].append((drain > content * (1 + 1e-9)).cpu().numpy())
        self.rows["fluxes"].append(torch.cat(fluxes).cpu().numpy())
        self.rows["state"].append(self.v["scalar_s"].cpu().numpy())
        capped = np.zeros(0, dtype=bool)
        if self.M is not None and self.M.numel() > 0:
            ds = torch.zeros_like(s0)
            ds[I] = -dt * self.B["scalar.D"][I]
            vol = self.dx[3].view(1, 1, 1, -1) * self.dx[2].view(1, 1, -1, 1) * self.dx[1].view(1, -1, 1, 1)
            held = (s0 + ds).clamp_min(0) * vol
            M = self.M[kIV1 + 1]
            r_below = torch.zeros_like(r0)
            r_below[..., 1:] = r0[..., :-1]
            P = torch.where(M > 0, r_below, r0) * M
            bind = torch.zeros_like(P, dtype=torch.bool)
            avail = held[..., 3]
            for i in range(4, s0.size(-1) - 3):
                q = P[..., i]
                kept = torch.maximum(torch.minimum(q, avail), -held[..., i])
                bind[..., i] = kept != q
                avail = held[..., i] + kept
            capped = bind.cpu().numpy()
        self.rows["capped"].append(capped)

    def counts(self):
        return {k: int(sum(int(a.sum()) for a in self.rows[k])) for k in ("limited", "drained", "capped")}


def arm_ratio(recon):
    import test_tracer_ratio as T

    build = T.build

    def build_recon(card):
        card["scalar"]["reconstruct"]["type"] = recon
        return build(card)

    T.build = build_recon
    block, v, dt = T.column(32, 4, limiter=True, x2max=1.6e6, amp=0.55, onecol=True, courant=30.0, q=5e-3)
    log = StageLog(block, v)

    def run():
        for stage in range(3):
            log.step(dt, stage)
        return v

    v, kernels = profiled(run)
    info = log.counts()
    return v, kernels, info, info["drained"] > 0 and info["limited"] > 0, log


def dry_convention_yaml():
    with open(os.path.join(HERE, "test_tracer_dry_convention.yaml")) as f:
        return yaml.safe_load(f)


def dry_convention_deck(recon):
    import test_tracer_dry_convention as J

    c = dry_convention_yaml()
    c["scalar"]["reconstruct"]["type"] = recon
    return J, c


def arm_upper(recon):
    from snapy import kIPR

    def one(bound):
        J, c = dry_convention_deck(recon)
        c["dynamics"]["equation-of-state"]["limiter"] = True
        if bound:
            c["scalar"]["upper-bound"] = 1.0e-3
        c["integration"]["implicit-scheme"] = 1
        c["geometry"]["bounds"]["x2max"] = 4.0e4
        b = J.make_block(c, "cuda")
        w, r = J.initial_state(b, c, moving=False)
        ng = 3
        w[kIPR][..., ng + 14] *= 2.0
        r.zero_()
        r[..., ng + 15] = 1.0e-3
        v, _ = b.initialize({"hydro_w": w, "scalar_r": r})
        I, smin, log = J.interior(c), math.inf, StageLog(b, v)
        for _ in range(3):
            dt = b.max_time_step(v)
            for st in range(len(b.intg.stages)):
                log.step(dt, st)
                smin = min(smin, float(v["scalar_s"][I].min()))
            b.check_redo(v)
        return v, log, smin

    (v, log, smin), kernels = profiled(lambda: one(True))
    _, log0, _ = one(False)
    # the same start with and without the bound: at the first stage whose fluxes differ in value (the blend
    # also flips signed zeros), every earlier state being equal in value, the faces the bound's blend moved
    changed, first = 0, -1
    for n, (a, b) in enumerate(zip(log.rows["fluxes"], log0.rows["fluxes"])):
        if (a != b).any():
            same_before = n == 0 or np.array_equal(log.rows["state"][n - 1], log0.rows["state"][n - 1])
            changed, first = (int((a != b).sum()) if same_before else 0), n
            break
    info = dict(log.counts(), **{"faces the bound moved": changed, "at stage": first, "min s": smin})
    return v, kernels, info, changed > 0 and smin >= 0.0, log


def arm_donor(recon):
    from snapy import kIPR

    J, c = dry_convention_deck(recon)
    c["integration"]["implicit-scheme"] = 1
    c["geometry"]["bounds"]["x2max"] = 4.0e4
    block = J.make_block(c, "cuda")
    w, r = J.initial_state(block, c, moving=False)
    w[kIPR][..., 3 + 14] *= 2.0
    r.zero_()
    r[..., 3 + 12] = J.R0
    v, _ = block.initialize({"hydro_w": w, "scalar_r": r})
    I, smin, log = J.interior(c), math.inf, StageLog(block, v)
    total0 = float(v["scalar_s"][I].sum())

    def run():
        nonlocal smin
        for _ in range(2):
            dt = block.max_time_step(v)
            for stage in range(len(block.intg.stages)):
                log.step(dt, stage)
                smin = min(smin, float(v["scalar_s"][I].min()))
        return v

    v, kernels = profiled(run)
    change = abs(float(v["scalar_s"][I].sum()) / total0 - 1.0)
    info = dict(log.counts(), **{"min s": smin, "total change": change})
    # measured: with dc no face of this deck reaches the cap; the through arm binds it with dc
    capped = info["capped"] > 0 or recon == "dc"
    return v, kernels, info, capped and smin >= 0.0 and change < 1e-12, log


def arm_through(recon):
    from snapy import kIDN, kIPR, kIV1

    J, c = dry_convention_deck(recon)
    c["integration"]["implicit-scheme"] = 1
    c["geometry"]["cells"]["nx2"], c["geometry"]["bounds"]["x2max"] = 1, 1.0e4
    block = J.make_block(c, "cuda")
    w, r = J.initial_state(block, c, moving=False)
    dz = float(c["geometry"]["bounds"]["x1max"]) / c["geometry"]["cells"]["nx1"]
    z = (torch.arange(w.size(-1), dtype=torch.float64, device=w.device) - 3 + 0.5) * dz
    w[kIPR] *= (1.0 + 2.0 * torch.exp(-(((z - 16.5 * dz) / (4.0 * dz)) ** 2) / 2)).view(1, 1, -1)
    v, _ = block.initialize({"hydro_w": w, "scalar_r": r})
    I = (slice(None), slice(None), slice(None), slice(3, -3))
    M = next(b for k, b in dict(block.named_buffers()).items() if k.endswith("mass_corr"))
    vol = dz * 1.0e4 * float(c["geometry"]["bounds"]["x3max"]) / c["geometry"]["cells"]["nx3"]
    r0, dt = v["scalar_s"][I] / v["hydro_u"][kIDN][I[1:]], block.max_time_step(v) * 10.0
    dev, moved, log = 0.0, 0.0, StageLog(block, v)

    def run():
        nonlocal dev, moved
        for stage in range(len(block.intg.stages)):
            cell = v["hydro_u"][kIDN][I[1:]] * vol
            log.step(dt, stage)
            moved = max(moved, float((M[kIV1 + 1][I[1:]].abs() / cell).max()))
            dev = max(dev, float((v["scalar_s"][I] / (v["hydro_u"][kIDN][I[1:]] * r0) - 1).abs().max()))
        return v

    v, kernels = profiled(run)
    info = dict(log.counts(), **{"max|r/r0-1|": dev, "cell masses moved": moved})
    return v, kernels, info, info["capped"] > 0 and dev <= 1e-14 and moved > 1.0, log


def arm_stratified(recon):
    import snapy

    def rho(z):
        return np.exp(-z / 3.2e4)

    def ratio(z):
        return 1 + 0.5 * np.sin(2 * np.pi * z / 1.6e4)

    outs = []

    def ds(n):  # test_scalar_stratified.py's ds() on the GPU
        c = dry_convention_yaml()
        c["geometry"]["cells"].update(nx1=n, nx2=1)
        c["scalar"]["reconstruct"] = {"type": recon}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", dir=os.getcwd()) as f:
            yaml.safe_dump(c, f)
            f.flush()
            b = snapy.MeshBlock(snapy.MeshBlockOptions.from_yaml(f.name))
        b.to(torch.device("cuda"))
        b.buffer("hydro.F1")[0].fill_(1.0)
        xf, (g, w) = b.buffer("coord.x1f").cpu().numpy(), np.polynomial.legendre.leggauss(8)

        def avg(f):
            pts = (xf[:-1, None] + xf[1:, None] + np.diff(xf)[:, None] * g) / 2
            return torch.tensor((w * f(pts)).sum(1) / 2).view(1, 1, 1, -1).cuda()

        u = b.buffer("hydro.D").clone().zero_()
        u[0] = avg(rho)
        s = b.module("scalar").forward(1.0, avg(lambda z: rho(z) * ratio(z)), {"hydro_u": u})
        outs.append(s.clone())
        s = s[0, 0, 0, 3:-3].cpu().numpy()
        return abs(s - (-np.diff(ratio(xf[3:-3])) / np.diff(xf[3:-3]))).sum() / n

    order, kernels = profiled(lambda: [math.log2(ds(m) / ds(2 * m)) for m in (40, 80)])
    v = {"scalar_s": torch.cat([o.reshape(-1) for o in outs]).view(1, 1, 1, -1)}
    ok = recon == "dc" or min(order) > 4.5
    return v, kernels, {"orders": order}, ok, None


def child(arm, recon, save):
    torch.set_default_dtype(torch.float64)
    v, kernels, info, bound, log = globals()[f"arm_{arm}"](recon)
    np.save(save, v["scalar_s"].cpu().numpy())
    if log is not None:
        rows = {f"{k}{n}": a for k in StageLog.KINDS for n, a in enumerate(log.rows[k])}
        np.savez(save[: -len(".npy")] + "_stages.npz", **rows)
    hydro = v.get("hydro_u")
    return {
        "kernels": kernels,
        "info": info,
        "bound": bool(bound),
        "hydro_u": None if hydro is None else digest(hydro),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None, choices=ARMS)
    ap.add_argument("--recon", default="weno5", choices=RECONS)
    ap.add_argument("--save", default=None)
    ap.add_argument("--device", default="cuda")  # main's builders read "cuda" from argv
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available")
        sys.exit(125)
    if a.child:
        print(json.dumps(child(a.child, a.recon, a.save)), flush=True)
        return
    failures = []
    tmpdir = tempfile.mkdtemp(dir=os.getcwd())
    for arm in ARMS:
        for recon in RECONS:
            if (arm, recon) in SKIP:
                continue
            res, arr, tag = {}, {}, f"{arm}/{recon}"
            for fused in ("0", "1"):
                env = dict(os.environ, SNAP_SCALAR_FUSED=fused)
                save = os.path.join(tmpdir, f"{arm}_{recon}_{fused}.npy")
                cmd = [sys.executable, os.path.abspath(__file__), "--child", arm, "--recon", recon]
                out = subprocess.run(
                    cmd + ["--save", save, "--device", "cuda"], env=env, capture_output=True, text=True, cwd=tmpdir
                )
                rows = [line for line in out.stdout.splitlines() if line.startswith("{")]
                if out.returncode or not rows:
                    print(out.stderr[-3000:], flush=True)
                    failures.append(f"{tag} fused={fused}: child exit {out.returncode}")
                    continue
                res[fused] = json.loads(rows[-1])
                res[fused]["warnings"] = warnings_in(out.stderr)
                arr[fused] = np.load(save)
                stages = save[: -len(".npy")] + "_stages.npz"
                arr[fused + "stages"] = dict(np.load(stages)) if os.path.exists(stages) else {}
            if len(res) < 2:
                continue
            # per stage, cell by cell: the limiter's cells, the donor-share cells and the capped faces on
            # both paths; with dc also every stage's limited fluxes and state, bit for bit
            s0, s1 = arr["0stages"], arr["1stages"]
            kinds = ("limited", "drained", "capped") + (("fluxes", "state") if recon == "dc" else ())
            if s0.keys() != s1.keys():
                failures.append(f"{tag}: the paths logged different stages")
            for key in sorted(s0.keys() & s1.keys()):
                if key.rstrip("0123456789") not in kinds:
                    continue
                a, b = s0[key], s1[key]
                same = a.shape == b.shape and (
                    np.array_equal(a, b) if a.dtype == bool else np.array_equal(a.view(np.uint64), b.view(np.uint64))
                )
                if not same:
                    failures.append(f"{tag}: stage record {key} differs between the paths")
            kernels, warns = expected(arm, recon)
            if res["0"]["kernels"] or res["0"]["warnings"]:
                failures.append(f"{tag}: the default path ran {res['0']['kernels']}, warned {res['0']['warnings']}")
            if not kernels <= set(res["1"]["kernels"]):
                failures.append(f"{tag}: fused kernels {sorted(kernels - set(res['1']['kernels']))} did not run")
            if res["1"]["warnings"] != warns:
                failures.append(f"{tag}: fallback warnings {sorted(res['1']['warnings'])}, expected {sorted(warns)}")
            if res["0"]["hydro_u"] != res["1"]["hydro_u"]:
                failures.append(f"{tag}: the flow differs (hydro_u)")
            for fused in ("0", "1"):
                if not res[fused]["bound"]:
                    failures.append(f"{tag} fused={fused}: the arm did not bind: {res[fused]['info']}")
            a0, a1 = arr["0"], arr["1"]
            if recon == "dc":
                same = a0.shape == a1.shape and np.array_equal(a0.view(np.uint64), a1.view(np.uint64))
                verdict = "bitwise identical" if same else "DIFFERENT"
                if not same:
                    failures.append(f"{tag}: fused and default paths differ")
            else:
                err = float((np.abs(a1 - a0) / np.maximum(local_max(a0), 1.0e-320)).max())
                verdict = f"within {err:.3g} of the local largest value"
                if not err <= TOL:
                    failures.append(f"{tag}: scalars differ by {err:.3g} of the local largest value (> {TOL})")
            print(
                f"{tag:18s} fused vs default: {verdict}; default {res['0']['info']}, fused {res['1']['info']}; "
                f"fused kernels {', '.join(res['1']['kernels'])}",
                flush=True,
            )
    shutil.rmtree(tmpdir)
    for failure in failures:
        print("FAIL", failure)
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
