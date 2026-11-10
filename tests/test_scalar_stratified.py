import math, os, sys, tempfile, numpy as np, torch, yaml, snapy
def ds(n, recon, rho, r):  # one Scalar.forward with mass flux 1 and exact cell averages of rho and rho r
    c = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "test_tracer_dry_convention.yaml")))
    c["geometry"]["cells"].update(nx1=n, nx2=1); c["scalar"]["reconstruct"] = {"type": recon}
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml"); yaml.safe_dump(c, f); f.flush()
    b = snapy.MeshBlock(snapy.MeshBlockOptions.from_yaml(f.name)); b.buffer("hydro.F1")[0].fill_(1.)
    xf, (g, w) = b.buffer("coord.x1f").numpy(), np.polynomial.legendre.leggauss(8)
    avg = lambda f: torch.tensor((w * f((xf[:-1, None] + xf[1:, None] + np.diff(xf)[:, None] * g) / 2)).sum(1) / 2).view(1, 1, 1, -1)
    u = b.buffer("hydro.D").clone().zero_(); u[0] = avg(rho)
    s = b.module("scalar").forward(1., avg(lambda z: rho(z) * r(z)), {"hydro_u": u})[0, 0, 0, 3:-3].numpy()
    return s, -np.diff(r(xf[3:-3])) / np.diff(xf[3:-3])
rho, r = lambda z: np.exp(-z / 3.2e4), lambda z: 1 + 0.5 * np.sin(2 * np.pi * z / 1.6e4)
o = [math.log2(e[0] / e[1]) for k in ("cp5", "weno5") for e in [[abs(np.subtract(*ds(n, k, rho, r))).sum() / n for n in (m, 2 * m)] for m in (40, 80)]]
nan = [not np.isfinite(ds(32, "cp5", lambda z: np.where(z < 1.6e4, 1., 1. / q), lambda z: 0.3 + 0 * z)[0]).all() for q in (10., 100.)]
print("order cp5, weno5:", o, "non-finite at 10:1, 100:1 contact:", nan)  # main 2d9fa419: order 2; 84aab21c: non-finite
sys.exit(1 if min(o) < 4.5 or any(nan) else 0)
