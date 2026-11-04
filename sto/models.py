"""Model registry. Each loader imports its package only when called, so a script needs only the
environment for the family it runs in (see setup_mac.sh: one environment per family).

For SrTiO3 the '+U' in PBE+U datasets does nothing: Materials Project and OMat24 put U on
transition metals such as Ni, Co and Fe, but not on Ti. So every 'PBE(+U)' model here
was trained on plain PBE for this material.
"""
from __future__ import annotations

import argparse
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import config

MACE_URLS = {
    "mpa-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
    "omat-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
    "matpes-pbe-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
    "matpes-r2scan-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model",
    "mh-1": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
    "mp-0b3": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
}
ASL = "Academic Software License (non-commercial)"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str  # which environment it runs in
    dataset: str
    functional: str  # 'PBE' or 'r2SCAN' (training reference for SrTiO3)
    licence: str
    loader: Callable  # loader(device) -> ASE calculator
    default: bool = True


def _mace(candidates, head=None):
    def load(device):
        from mace.calculators import mace_mp

        errors = []
        for c in candidates:
            try:
                kw = {"model": c, "device": device, "default_dtype": "float64"}
                if head is not None:
                    kw["head"] = head
                return mace_mp(**kw)
            except Exception as exc:  # try the next spelling / URL
                errors.append(f"{c}: {type(exc).__name__}: {exc}")
        raise RuntimeError("could not load MACE model; tried\n  " + "\n  ".join(errors))

    return load


def _mace_file(path):
    def load(device):
        from mace.calculators import MACECalculator

        return MACECalculator(model_paths=str(path), device=device, default_dtype="float64")

    return load


def _orb(device):
    from orb_models.forcefield import pretrained

    try:  # orb-models >= 0.6
        from orb_models.forcefield.inference.calculator import ORBCalculator
    except ImportError:  # orb-models <= 0.5
        from orb_models.forcefield.calculator import ORBCalculator

    out = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-highest")
    if isinstance(out, tuple):  # newer orb-models return (model, atoms_adapter)
        model, adapter = out
        return ORBCalculator(model, atoms_adapter=adapter, device=device)
    return ORBCalculator(out, device=device)


def _mattersim(device):
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)


def _sevennet(device):
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(model="7net-mf-ompa", modal="mpa", device=device)


def _chgnet(device):
    from chgnet.model.dynamics import CHGNetCalculator

    return CHGNetCalculator(use_device=device)


REGISTRY = [
    ModelSpec("mace-mp-0b3", "mace", "MPtrj", "PBE", "MIT", _mace(["medium-0b3", MACE_URLS["mp-0b3"]])),
    ModelSpec("mace-mpa-0", "mace", "MPtrj + sAlex", "PBE", "MIT", _mace([MACE_URLS["mpa-0"], "medium-mpa-0"])),
    ModelSpec("mace-omat-0", "mace", "OMat24", "PBE", ASL, _mace([MACE_URLS["omat-0"], "medium-omat-0"])),
    ModelSpec("mace-matpes-pbe-0", "mace", "MatPES (PBE)", "PBE", ASL, _mace([MACE_URLS["matpes-pbe-0"]])),
    ModelSpec("mace-matpes-r2scan-0", "mace", "MatPES (r2SCAN)", "r2SCAN", ASL, _mace([MACE_URLS["matpes-r2scan-0"]])),
    ModelSpec("mace-mh-1-omat", "mace", "MH-1, omat_pbe head", "PBE", ASL, _mace([MACE_URLS["mh-1"]], head="omat_pbe")),
    ModelSpec("mace-mh-1-r2scan", "mace", "MH-1, matpes_r2scan head", "r2SCAN", ASL,
              _mace([MACE_URLS["mh-1"]], head="matpes_r2scan")),
    ModelSpec("orb-v3-omat", "orb", "OMat24", "PBE", "see orb-models", _orb),
    ModelSpec("mattersim-5m", "mattersim", "MatterSim (proprietary, PBE)", "PBE", "see mattersim", _mattersim),
    ModelSpec("sevennet-mf-ompa", "sevenn", "OMat24 + sAlex + MPtrj (mpa modal)", "PBE", "see sevenn", _sevennet),
    ModelSpec("chgnet-0.3", "chgnet", "MPtrj", "PBE", "see chgnet", _chgnet),
]
FAMILIES = ("mace", "orb", "mattersim", "sevenn", "chgnet")


def finetuned() -> list[ModelSpec]:
    """Fine-tuned MACE models dropped into models/ (from notebooks/finetune_mace.ipynb)."""
    return [
        ModelSpec(f"mace-ft-{p.stem}", "mace", f"fine-tuned on the DFT data ({p.name})", "as the DFT",
                  "as base model", _mace_file(p), default=False)
        for p in sorted(Path(config.MODELS_DIR).glob("*.model"))
    ]


def all_models() -> list[ModelSpec]:
    return REGISTRY + finetuned()


def get(name: str) -> ModelSpec:
    for m in all_models():
        if m.name == name:
            return m
    raise KeyError(f"unknown model {name!r}; known: {[m.name for m in all_models()]}")


def add_selection_args(ap: argparse.ArgumentParser):
    ap.add_argument("--family", choices=FAMILIES, help="run only models from this environment")
    ap.add_argument("--models", nargs="+", metavar="PATTERN",
                    help="model names or glob patterns, e.g. 'mace-*' or 'mace-ft-*' (default: all standard models)")
    ap.add_argument("--device", default=config.DEVICE)


def selected(args) -> list[ModelSpec]:
    pool = all_models()
    if args.models:
        pool = [m for m in pool if any(fnmatch.fnmatch(m.name, p) for p in args.models)]
    else:
        pool = [m for m in pool if m.default]
    if args.family:
        pool = [m for m in pool if m.family == args.family]
    return pool
