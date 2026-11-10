#!/usr/bin/env python3
"""Case M: bottom composition transfer can make dry carry's base zero."""
import tempfile
from pathlib import Path
from typing import Dict

import torch
import yaml


class _BottomDry(torch.nn.Module):
    def forward(self, variables: Dict[str, torch.Tensor], dt: float,
                stage: int) -> Dict[str, torch.Tensor]:
        del dt, stage
        du = torch.zeros_like(variables["hydro_u"])
        du[0].select(-1, 2).fill_(-1.e-3)
        du[5].select(-1, 2).fill_(1.e-3)
        return {"hydro_du": du}


def main() -> int:
    from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR

    source = Path(__file__).resolve().parent / "test_tracer_dry_convention.yaml"
    with source.open() as stream:
        card = yaml.safe_load(stream)
    card["geometry"]["cells"] = {"nx1": 8, "nx2": 1, "nx3": 1, "nghost": 2}
    card["geometry"]["bounds"] = {
        "x1min": 0., "x1max": 8000., "x2min": 0., "x2max": 1000.,
        "x3min": 0., "x3max": 1000.,
    }
    card["dynamics"]["equation-of-state"].update({
        "density-floor": 1.e-6, "pressure-floor": 1.e-6, "limiter": False,
    })
    for direction in ("vertical", "horizontal"):
        card["dynamics"]["reconstruct"][direction] = {
            "type": "plm", "scale": False, "shock": False,
        }
    for axis in ("x1-inner", "x1-outer", "x2-inner", "x2-outer",
                 "x3-inner", "x3-outer"):
        card["boundary-condition"]["external"][axis] = "reflecting"
    card["integration"] = {"type": "rk1", "cfl": 0.4,
                           "implicit-scheme": 0, "nlim": 4, "tlim": 1.e9}
    card["forcing"] = {"relax-bot-comp": {
        "tau": 1.e-4, "species": ["vapor"], "xfrac": [1.],
    }}
    card["scalar"]["reconstruct"] = {
        "type": "plm", "shock": False,
    }

    with tempfile.TemporaryDirectory(prefix="snapy-case-m-") as directory:
        directory = Path(directory)
        path = directory / "case.yaml"
        path.write_text(yaml.safe_dump(card))
        module = directory / "dry.pt"
        torch.jit.script(_BottomDry().eval()).save(str(module))
        block = MeshBlock(MeshBlockOptions.from_yaml(str(path)))
        block.set_user_stage_forcings([str(module)])
        coord = block.module("coord")
        eos = block.module("hydro.eos")
        shape = (eos.nvar(), coord.buffer("x3v").shape[0],
                 coord.buffer("x2v").shape[0], coord.buffer("x1v").shape[0])
        w = torch.zeros(shape, dtype=torch.float64)
        w[kIDN] = 1.
        w[kIPR] = 1.e5
        w[kICY] = 0.5
        r = torch.full((1,) + shape[1:], 0.5, dtype=torch.float64)
        variables, _ = block.initialize({"hydro_w": w, "scalar_r": r})
        assert 1.e-4 <= block.max_time_step(variables)
        block.forward(variables, 1.e-4, 0)
        bottom = variables["scalar_s"][..., 2]
        assert torch.isfinite(bottom).all(), "Case M dry carry produced NaN"
        assert torch.allclose(bottom, torch.full_like(bottom, -5.e-4),
                              rtol=0., atol=1.e-12), bottom
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
