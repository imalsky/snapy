"""run_hydro's initial column on a stretched x1 (#280).

The example drivers build their initial column by stepping an adiabat from one
cell centre to the next. The step must be the centre spacing (dx1v), not the
cell width (dx1f): on a stretched x1 the two differ, and with dx1f the
pressure at the top of a 64-cell column is off by ~35%.

A dry 300 km column (one species, cv_R 2.5, g = 22) is run for zero steps on a
uniform grid and on x1rat 1/1.058 and 1.058. The initial pressure at every
centre is compared with the analytic dry adiabat, and the stretched error must
stay within a few times the uniform error of the same driver.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

DECK = """reference-state: {{Tref: 0., Pref: 1.e5}}
species:
  - name: dry
    composition: {{H: 1.5, He: 0.15}}
    cv_R: 2.5
geometry:
  type: cartesian
  bounds: {{x1min: 0., x1max: 300.e3, x2min: 0., x2max: 1.e3, x3min: 0., x3max: 1.e3}}
  cells: {{nx1: {n}, nx2: 1, nx3: 1, nghost: 3{rat}}}
dynamics:
  equation-of-state: {{type: ideal-moist, density-floor: 1.e-10, pressure-floor: 1.e-10, limiter: true, max-iter: 10, ftol: 1.e-6}}
  reconstruct:
    vertical: {{type: weno5, scale: true, shock: true}}
    horizontal: {{type: weno5, scale: true, shock: true}}
  riemann-solver: {{type: hllc}}
boundary-condition:
  external: {{x1-inner: reflecting, x1-outer: reflecting, x2-inner: periodic, x2-outer: periodic, x3-inner: periodic, x3-outer: periodic}}
integration: {{type: rk3, cfl: 0.45, implicit-scheme: 0, nlim: 0, tlim: 1.e9, ncycle_out: 1}}
outputs:
  - type: restart
    dt: 1.e30
forcing:
  const-gravity: {{grav1: -{g}}}
problem: {{Ps: {ps}, Ts: {ts}, Tmin: 1.}}
"""

N, G, PS, TS = 64, 22.0, 1.0e7, 600.0
RGAS = 8.31446  # kintera::constants::Rgas


def column(exe, workdir, q):
    import snapy

    name = "u" if q is None else f"q{q}"
    d = os.path.join(workdir, name)
    os.makedirs(d)
    for res in ("nasa9.dat",):
        src = os.path.join(os.getcwd(), res)
        if os.path.exists(src):
            os.symlink(src, os.path.join(d, res))
    rat = "" if q is None else f", x1rat: {q!r}"
    with open(os.path.join(d, "col.yaml"), "w") as f:
        f.write(DECK.format(n=N, rat=rat, g=G, ps=PS, ts=TS))
    subprocess.run([exe, "-i", "col.yaml"], cwd=d, check=True,
                   stdout=subprocess.DEVNULL)
    w = snapy.load_restart(os.path.join(d, "col.00000.restart"), 0)["hydro_w"]
    op = snapy.MeshBlockOptions.from_yaml(os.path.join(d, "col.yaml"))
    x1v = snapy.MeshBlock(op).module("coord").buffer("x1v")
    ng = op.coord().nghost()
    return x1v[ng:ng + N].double().numpy(), w[4, 0, 0, ng:ng + N].double().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    args = parser.parse_args()
    args.exe = os.path.abspath(args.exe)

    if not os.path.exists(args.exe):
        print(f"{args.exe} was not built")
        sys.exit(125)

    import snapy
    print(f"snapy.__file__ = {snapy.__file__}", flush=True)

    import kintera

    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "thermo.yaml"), "w") as f:
            f.write(DECK.format(n=N, rat="", g=G, ps=PS, ts=TS))
        kintera.ThermoOptions.from_yaml(os.path.join(tmp, "thermo.yaml"))
        rd = RGAS / kintera.species_weights()[0]
    cp = 3.5 * rd

    def error(z, p):
        t = TS - G * z / cp
        return np.abs(p / (PS * (t / TS) ** (cp / rd)) - 1.0).max()

    with tempfile.TemporaryDirectory() as tmp:
        e_u = error(*column(args.exe, tmp, None))
        print(f"uniform N={N}: max |p/p_adiabat - 1| = {e_u:.3e}")
        ok = e_u < 0.05
        for q in (1.0 / 1.058, 1.058):
            e = error(*column(args.exe, tmp, q))
            print(f"x1rat {q:.6f} N={N}: max |p/p_adiabat - 1| = {e:.3e} "
                  f"({e / e_u:.2f}x uniform)")
            ok &= e < 6.0 * e_u
    print("PASS" if ok else "FAIL: stretched IC error exceeds 6x the uniform")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
