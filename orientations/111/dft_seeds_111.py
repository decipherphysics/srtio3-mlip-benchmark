"""Relax the [111] DFT final structures with the model, as extra starts for the scan (they include
60-atom states the 30-atom scan cell cannot hold). Each distinct DFT state at each lattice parameter is
scaled in the plane to the same strain measured from the model's a0, then relaxed with the film
constraint and no symmetry. Results go to results/<strain>_dft_<phase>.json and .extxyz. Resumable.

    ~/.sto-envs/mace/bin/python orientations/111/dft_seeds_111.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sto111 as S  # noqa: E402
import film_measure as F  # noqa: E402
from scan_111 import model_a0  # noqa: E402


def main():
    frames = read(S.HERE / "dft" / "finals.extxyz", ":")
    frames = [a for a in frames if "path" in a.info]  # drops the two bulk 'Relaxed' runs
    for a in frames:  # the folder name is the label; extxyz reads the label 3.9_2 back as the number 3.92
        a.info["a_label"] = str(a.info["path"]).split("/")[1]
    frames = [a for a in frames if a.info["a_label"] != "Relaxed" and a.info["phase"] != "Pm-3m"]
    seen, todo = set(), []
    for a in sorted(frames, key=lambda x: (float(str(x.info["a_label"]).split("_")[0]), x.info["e0"] / len(x))):
        lab = float(str(a.info["a_label"]).split("_")[0])
        d = F.describe(a)
        key = (lab, round(1000 * a.info["e0"] / d["n_fu"], 1), round(d["ti_disp"], 2), round(d["tilt_total"], 0))
        if key in seen:  # another folder that ended in the same state
            continue
        seen.add(key)
        todo.append((lab, a, d))
    calc = S.models.get(S.MODEL).loader("cpu")
    a0 = model_a0(calc)
    print(f"=== {len(todo)} distinct DFT states of {len(frames)} runs; model a0 = {a0:.4f} A", flush=True)
    t0 = time.time()
    for lab, a, d in todo:
        s = round((lab / S.config.DFT_A0_CUBIC - 1) * 100, 3)
        tag, name = S.config.strain_tag(s), f"dft_{a.info['phase']}"
        stem = S.RESULTS / f"{tag}_{name}"
        if Path(f"{stem}.json").exists():
            continue
        a_t = a0 * (1 + s / 100)
        p, c = S.cells.to_film_frame(a.get_positions(), np.array(a.cell))
        p, c, _ = S.cells.strain_cell(p, c, S.ORIENT, a_t, poisson=0.0)
        sym = a.get_chemical_symbols()
        r = S.st.relax(S.st.to_atoms(p, sym, c), calc, film=True, symmetric=False, fmax=S.config.FMAX)
        m = r["atoms"]
        n_fu = len(m) // 5
        rec = {"model": S.MODEL, "orientation": S.ORIENT, "strain_pct": s, "seed": name, "a_target": a_t,
               "a0_model": a0, "n_fu": n_fu, "natoms": len(m), "energy": r["energy"], "e_per_fu": r["energy"] / n_fu,
               "converged": r["converged"], "nsteps": r["nsteps"], "seconds": r["seconds"],
               "fmax_final": r["fmax_final"], "dft_path": a.info["path"], "dft_e_per_fu": a.info["e0"] / n_fu,
               "dft_converged": bool(a.info["converged"])}
        rec.update(F.describe(m))
        m.info.update(model=S.MODEL, orientation="o111", strain_pct=s, seed=name, model_energy=r["energy"])
        write(f"{stem}.extxyz", m, format="extxyz")
        S.io.write_json(Path(f"{stem}.json"), rec)
        print(f"  {s:+6.2f}% {name:14s} E {rec['e_per_fu']:.5f} eV/fu  Ti {rec['ti_disp']:.3f} A (normal {rec['ti_normal']:.3f}, "
              f"in-plane {rec['ti_inplane']:.3f})  tilt {rec['tilt_total']:.1f} deg  {r['nsteps']:4d} steps {r['seconds']:6.1f} s"
              f"{'' if r['converged'] else '  NOT CONVERGED'}", flush=True)
    print(f"dft seeds done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
