"""Step 1: turn the included VASP runs into one extxyz file plus a summary of final structures.

    ~/.sto-envs/mace/bin/python scripts/01_convert.py [--every 1]

Writes
  data/dft_frames.extxyz  every ionic step (or every Nth) with REF_energy, REF_forces, REF_stress.
                          The same file can be used to fine-tune a MACE model.
  data/dft_final.csv      final structure of each run: strain, c/a, Ti displacement, tilts.
  data/dft_a0.json        the cubic lattice parameter implied by the strain labels.

REF_energy is VASP's free energy (TOTEN), which is the energy the forces are derivatives of.
Assumes each DFT cell has its first two lattice vectors in the film plane.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase import Atoms  # noqa: E402
from ase.io import read, write  # noqa: E402

from sto import analysis as an  # noqa: E402
from sto import config, io  # noqa: E402


def _float(x):
    try:
        return float(str(x).replace("p", "."))
    except ValueError:
        return math.nan


def load_images(run_dir: Path):
    for name in ("vasprun.xml", "OUTCAR"):
        src = run_dir / name
        if src.exists():
            try:
                return read(src, index=":"), src
            except Exception as exc:
                print(f"  {src}: {type(exc).__name__}: {exc}")
    return [], None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", default=str(config.AUDIT_CSV))
    ap.add_argument("--every", type=int, default=1, help="keep every Nth ionic step (final step always kept)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.audit)))
    rows = [r for r in rows if r["include"].strip().lower() in ("y", "yes", "1", "true")]
    print(f"{len(rows)} runs marked include=y")
    frames, finals = [], []
    for row in rows:
        d = Path(row["path"])
        images, src = load_images(d)
        if not images:
            print(f"  skipped (unreadable): {d}")
            continue
        orient = config.orientation_str(row["orientation"]) if row["orientation"] else ""
        strain = _float(row["strain_percent"])
        n = len(images)
        keep = sorted(set(range(0, n, max(1, args.every))) | {n - 1})
        for i in keep:
            img = images[i]
            try:
                e = float(img.get_potential_energy(force_consistent=True))
            except Exception:
                e = float(img.get_potential_energy())
            new = Atoms(img.get_chemical_symbols(), positions=img.get_positions(), cell=np.array(img.cell), pbc=True)
            new.info["REF_energy"] = e
            new.arrays["REF_forces"] = np.array(img.get_forces())
            try:
                new.info["REF_stress"] = np.array(img.get_stress(voigt=True))
            except Exception:
                pass  # no stress printed (ISIF = 0)
            new.info.update(config_type=f"o{orient or 'unknown'}", orientation=orient or "unknown",
                            strain_percent=strain, run_path=config.archive_relative(d), step_index=i, is_final=(i == n - 1),
                            run_type=row["run_type"])
            frames.append(new)

        last = images[-1]
        pos, sym, cell = last.get_positions(), last.get_chemical_symbols(), np.array(last.cell)
        n_fu = len(sym) // 5
        rec = {"path": config.archive_relative(d), "source": src.name, "orientation": orient, "strain_percent": strain,
               "natoms": len(sym), "n_fu": n_fu, "run_type": row["run_type"],
               "energy": float(last.get_potential_energy(force_consistent=True)) if last.calc else math.nan}
        rec["e_per_fu"] = rec["energy"] / n_fu if n_fu else math.nan
        rec.update(an.polar_summary(pos, sym, cell))
        rec.update(an.tilt_summary(pos, sym, cell, normal=np.cross(cell[0], cell[1])))
        if orient in ("001", "111"):
            rec.update(an.tetragonality(cell, n_fu, orient))
            if not math.isnan(strain):
                rec["a0_implied"] = rec["a_inplane"] / (1 + strain / 100)
        rec["phase_guess"] = "polar" if rec["ti_disp_polar"] > config.D_TOL else "nonpolar"
        finals.append(rec)

    if not frames:
        print("No frames written.")
        return 1
    config.DATA.mkdir(exist_ok=True)
    write(config.DFT_FRAMES, frames, format="extxyz")
    keys = sorted({k for r in finals for k in r})
    first = ["path", "orientation", "strain_percent", "phase_guess", "e_per_fu", "ti_disp_polar",
             "tilt_all", "a_inplane", "c_over_a", "a0_implied"]
    keys = [k for k in first if k in keys] + [k for k in keys if k not in first]
    with open(config.DFT_FINAL_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(finals)

    a0 = {}
    for o in ("001", "111"):
        vals = [r["a0_implied"] for r in finals if r.get("orientation") == o and "a0_implied" in r]
        if vals:
            a0[o] = {"median": float(np.median(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "n": len(vals)}
    io.write_json(config.DFT_A0_JSON, a0)

    print(f"Wrote {len(frames)} frames from {len(finals)} runs to {config.DFT_FRAMES}")
    print(f"Final-structure summary: {config.DFT_FINAL_CSV}")
    for o, v in a0.items():
        spread = v["max"] - v["min"]
        flag = "" if spread < 0.005 else "   <- spread > 0.005 A: strain labels may use different references"
        print(f"  [{o}] cubic a0 implied by the strain labels: {v['median']:.4f} A (n = {v['n']}){flag}")
    print(f"  polar final structures: {sum(r['phase_guess'] == 'polar' for r in finals)} / {len(finals)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
