"""Saddle check of the [110] scan, as 05_phonons.py does for the [001] study.

At each strain, Gamma-point phonons of the lowest run (the 20-atom cell folds the R point and one M point
of the cubic cell onto Gamma, so polar and tilt modes both appear). An imaginary optical mode means the
run is a saddle, not a minimum: the structure is moved along each such mode (both signs, largest atomic
displacement 0.05 A) and relaxed at 0.2 meV/A, up to three rounds. A lower result is saved as
<strain>_followed<k> with the scan's record fields, and its phonons are checked in turn.

    ~/.sto-envs/mace/bin/python orientations/110/phonons_110.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sto110 as S  # noqa: E402
from scan_110 import report, save  # noqa: E402

_spec = importlib.util.spec_from_file_location("phonons_001", S.REPO / "scripts" / "05_phonons.py")
ph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ph)  # the study's tested phonon and mode-following code, read only

ROUNDS = 3


def records(outdir: Path, tag: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(outdir.glob(f"{tag}_*.json"))]


def phonons(atoms, calc):
    freqs, modes, masses, ndisp = ph.gamma_phonons(atoms, calc)
    rows, summary = ph.summarise(freqs, modes, masses, atoms.get_chemical_symbols(), atoms.get_positions(),
                                 np.array(atoms.cell))
    return freqs, modes, masses, rows, summary, ndisp


def check_strain(calc, outdir: Path, pdir: Path, s: float):
    tag = S.config.strain_tag(s)
    recs = records(outdir, tag)
    e_np = min(r["e_per_fu"] for r in recs if r["seed"].startswith("np_"))
    cur = min(recs, key=lambda r: r["e_per_fu"])
    history = []
    for k in range(1, ROUNDS + 2):
        atoms = read(outdir / f"{tag}_{cur['seed']}.extxyz")
        freqs, modes, masses, rows, summary, ndisp = phonons(atoms, calc)
        optical = sorted(r["freq_cm1"] for r in rows if r["label"] != "acoustic")
        history.append({"seed": cur["seed"], "e_per_fu": cur["e_per_fu"], "n_imaginary": summary["n_imaginary"],
                        "lowest_polar_cm1": summary["lowest_polar_cm1"], "lowest_tilt_cm1": summary["lowest_tilt_cm1"],
                        "lowest_optical_cm1": optical[:6], "n_displacements": ndisp})
        print(f"  {s:+6.2f}% {cur['seed']:16s} imaginary optical modes: {summary['n_imaginary']}  "
              f"lowest {optical[0]:8.1f} cm-1", flush=True)
        if summary["n_imaginary"] == 0 or k > ROUNDS:
            break
        r, f = ph.relax_along_unstable(atoms, calc, freqs, modes, masses)
        if r["energy"] / S.N_FU >= cur["e_per_fu"] - 1e-7:
            print(f"  {s:+6.2f}% following the {f:.1f} cm-1 mode found nothing lower", flush=True)
            break
        rec = save(r, s, f"followed{k}", cur["a_target"], cur["a0_model"], outdir, {
            "gain_vs_nonpolar_mev": S.an.polar_energy_gain(e_np, r["energy"] / S.N_FU, 1),
            "nonpolar_ref": cur.get("nonpolar_ref", cur["seed"]), "followed_from": cur["seed"],
            "followed_mode_cm1": float(f), "fmax": S.config.FMAX_FOLLOW})
        report(rec)
        cur = rec
    S.io.write_json(pdir / f"{tag}.json", {"strain_pct": s, "checks": history})


def main():
    outdir = S.HERE / sys.argv[1] if len(sys.argv) > 1 else S.RESULTS  # optional: another results folder
    pdir = outdir.parent / f"{outdir.name}_phonons"
    pdir.mkdir(exist_ok=True)
    calc = S.models.get(S.MODEL).loader("cpu")
    strains = sorted({json.loads(p.read_text())["strain_pct"] for p in outdir.glob("*_np_untilted.json")})
    t0 = time.time()
    for s in strains:
        check_strain(calc, outdir, pdir, s)
    print(f"phonons done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
