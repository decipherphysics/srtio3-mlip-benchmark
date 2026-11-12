"""Saddle check of the [111] results, as for [110]: Gamma-point phonons of the lowest run at each strain
(scan runs and relaxed DFT structures alike). An imaginary optical mode means a saddle: the structure is
moved along it (both signs, 0.05 A) and relaxed at 0.2 meV/A, up to three rounds; a lower result is saved
as <strain>_followed<k>.

    ~/.sto-envs/mace/bin/python orientations/111/phonons_111.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sto111 as S  # noqa: E402
import film_measure as F  # noqa: E402

_spec = importlib.util.spec_from_file_location("phonons_001", S.REPO / "scripts" / "05_phonons.py")
ph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ph)  # the study's tested phonon and mode-following code, read only

ROUNDS = 3


def check_strain(calc, outdir: Path, pdir: Path, s: float):
    tag = S.config.strain_tag(s)
    recs = [json.loads(p.read_text()) for p in sorted(outdir.glob(f"{tag}_*.json"))]
    cur = min(recs, key=lambda r: r["e_per_fu"])
    history = []
    for k in range(1, ROUNDS + 2):
        atoms = read(outdir / f"{tag}_{cur['seed']}.extxyz")
        n_fu = len(atoms) // 5
        freqs, modes, masses, ndisp = ph.gamma_phonons(atoms, calc)
        rows, summary = ph.summarise(freqs, modes, masses, atoms.get_chemical_symbols(), atoms.get_positions(),
                                     np.array(atoms.cell))
        optical = sorted(r["freq_cm1"] for r in rows if r["label"] != "acoustic")
        history.append({"seed": cur["seed"], "e_per_fu": cur["e_per_fu"], "natoms": len(atoms),
                        "n_imaginary": summary["n_imaginary"], "lowest_optical_cm1": optical[:6]})
        print(f"  {s:+6.2f}% {cur['seed']:16s} ({len(atoms)} atoms) imaginary optical modes: {summary['n_imaginary']}  "
              f"lowest {optical[0]:8.1f} cm-1", flush=True)
        if summary["n_imaginary"] == 0 or k > ROUNDS:
            break
        r, f = ph.relax_along_unstable(atoms, calc, freqs, modes, masses)
        if r["energy"] / n_fu >= cur["e_per_fu"] - 1e-7:
            print(f"  {s:+6.2f}% following the {f:.1f} cm-1 mode found nothing lower", flush=True)
            break
        m = r["atoms"]
        rec = {"model": S.MODEL, "orientation": S.ORIENT, "strain_pct": s, "seed": f"followed{k}",
               "a_target": cur["a_target"], "a0_model": cur["a0_model"], "n_fu": n_fu, "natoms": len(m),
               "energy": r["energy"], "e_per_fu": r["energy"] / n_fu, "converged": r["converged"],
               "nsteps": r["nsteps"], "seconds": r["seconds"], "fmax_final": r["fmax_final"],
               "followed_from": cur["seed"], "followed_mode_cm1": float(f)}
        rec.update(F.describe(m))
        write(outdir / f"{tag}_followed{k}.extxyz", m, format="extxyz")
        S.io.write_json(outdir / f"{tag}_followed{k}.json", rec)
        print(f"  {s:+6.2f}% followed{k}: {1000 * (cur['e_per_fu'] - rec['e_per_fu']):.3f} meV/fu lower, "
              f"Ti {rec['ti_disp']:.3f} A", flush=True)
        cur = rec
    S.io.write_json(pdir / f"{tag}.json", {"strain_pct": s, "checks": history})


def main():
    outdir, pdir = S.RESULTS, S.HERE / "results_phonons"
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
