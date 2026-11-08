"""G3 control for snapy#280 stage 1: compare two g3_decks dumps bit for bit.

Every tensor is loaded as float64 and compared with ``torch.equal``, which is
the tolerance the design note asks for in the G3 row (note line 587: "0 bits:
every output and restart tensor ``torch.equal``").

    g3_compare.py <base-dir> <branch-dir>
"""

import sys
from pathlib import Path

import numpy as np
import torch


def manifest(d):
    out = {}
    for m in sorted(Path(d).glob("*.manifest")):
        deck = m.name[: -len(".manifest")]
        for line in m.read_text().splitlines():
            blk, key, numel, fn, h = line.split()
            out[(deck, int(blk), key)] = (int(numel), fn, int(h))
    return out


def main(base, branch):
    a, b = manifest(base), manifest(branch)
    if set(a) != set(b):
        print("MANIFEST MISMATCH")
        print("  only in base  :", sorted(set(a) - set(b))[:10])
        print("  only in branch:", sorted(set(b) - set(a))[:10])
        return 1

    decks = sorted({k[0] for k in a})
    ntensor = nfile = ndiff = 0
    nelem = 0
    worst = []

    for key in sorted(a):
        numel, fna, ha = a[key]
        _, fnb, hb = b[key]
        xa = torch.from_numpy(
            np.fromfile(Path(base) / fna, dtype="<f8").astype(np.float64))
        xb = torch.from_numpy(
            np.fromfile(Path(branch) / fnb, dtype="<f8").astype(np.float64))
        ntensor += 1
        nfile += 2
        nelem += xa.numel()
        if not torch.equal(xa, xb):
            ndiff += 1
            d = (xa - xb).abs()
            worst.append((key, d.max().item(), int((xa != xb).sum())))
        elif ha != hb:
            ndiff += 1
            worst.append((key, float("nan"), -1))

    print(f"decks   compared : {len(decks)}  ({', '.join(decks)})")
    print(f"files   compared : {nfile}")
    print(f"tensors compared : {ntensor}")
    print(f"elements         : {nelem}")
    print(f"tensors differing: {ndiff}")
    for k, m, n in worst[:20]:
        print(f"   DIFF {k}  max|a-b| = {m:.6e}  elements differing = {n}")
    return 0 if ndiff == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
