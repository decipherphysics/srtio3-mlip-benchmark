"""Step 3: every model evaluated on every DFT frame (no relaxation): energy, force and stress errors.

    ./run.sh scripts/03_singlepoints.py            # all environments
    ~/.sto-envs/mace/bin/python scripts/03_singlepoints.py --family mace

Writes results/singlepoints/<model>.csv (one row per frame) and prints a summary.
Model and DFT energies have different zero points (different codes, pseudopotentials,
functionals), so one constant per formula unit is fitted and removed before energy errors are
reported. Energy differences between structures, which is what the study is about, do not
depend on that constant.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase.io import read  # noqa: E402

from sto import analysis as an  # noqa: E402
from sto import config, io, models  # noqa: E402

EV_A3_TO_GPA = 160.21766


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    models.add_selection_args(ap)
    ap.add_argument("--frames", default=str(config.DFT_FRAMES))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    frames = read(args.frames, index=":")
    print(f"{len(frames)} DFT frames")
    out_dir = config.RESULTS / "singlepoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in models.selected(args):
        out = out_dir / f"{spec.name}.csv"
        if out.exists() and not args.overwrite:
            print(f"{spec.name}: exists, skipping (use --overwrite)")
            continue
        try:
            calc = spec.loader(args.device)
        except Exception as exc:
            print(f"{spec.name}: could not load ({exc})")
            continue
        t0 = time.time()
        rows = []
        for k, fr in enumerate(frames):
            a = fr.copy()
            a.calc = calc
            try:
                e = float(a.get_potential_energy())
                f = a.get_forces()
                s = a.get_stress(voigt=True)
            except Exception:
                traceback.print_exc()
                continue
            f_ref = fr.arrays["REF_forces"]
            df = f - f_ref
            row = {"frame": k, "run_path": fr.info.get("run_path", ""), "step_index": fr.info.get("step_index", -1),
                   "is_final": bool(fr.info.get("is_final", False)),
                   "orientation": config.orientation_str(fr.info.get("orientation", "")),
                   "strain_percent": fr.info.get("strain_percent", np.nan), "natoms": len(a), "n_fu": len(a) // 5,
                   "E_ref": fr.info["REF_energy"], "E_model": e,
                   "force_mae": float(np.abs(df).mean()), "force_rmse": float(np.sqrt((df ** 2).mean())),
                   "force_max_err": float(np.linalg.norm(df, axis=1).max()),
                   "ref_force_max": float(np.linalg.norm(f_ref, axis=1).max())}
            if "REF_stress" in fr.info:
                ds = (np.asarray(s) - np.asarray(fr.info["REF_stress"])) * EV_A3_TO_GPA
                row["stress_mae_gpa"] = float(np.abs(ds).mean())
            rows.append(row)
        if not rows:
            print(f"{spec.name}: no frames evaluated")
            continue
        comps = [{"Sr": r["n_fu"], "Ti": r["n_fu"], "O": 3 * r["n_fu"]} for r in rows]
        mu, resid = an.fit_offsets([r["E_model"] for r in rows], [r["E_ref"] for r in rows], comps)
        for r, d in zip(rows, resid):
            r["E_err_per_atom_mev"] = 1000 * d / r["natoms"]
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        e_mae = np.mean([abs(r["E_err_per_atom_mev"]) for r in rows])
        f_mae = 1000 * np.mean([r["force_mae"] for r in rows])
        s_mae = np.nanmean([r.get("stress_mae_gpa", np.nan) for r in rows])
        summary = {"model": spec.name, "n_frames": len(rows), "energy_mae_mev_per_atom": e_mae,
                   "force_mae_mev_per_A": f_mae, "stress_mae_gpa": s_mae,
                   "offset_ev_per_fu": mu["Sr"] + mu["Ti"] + 3 * mu["O"], "seconds": time.time() - t0}
        io.write_json(out_dir / f"{spec.name}_summary.json", summary)
        print(f"{spec.name:22s} E {e_mae:6.2f} meV/atom   F {f_mae:6.1f} meV/A   S {s_mae:5.2f} GPa   "
              f"({len(rows)} frames, {summary['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
