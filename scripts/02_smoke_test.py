"""Step 2: load every model in one environment, download its weights, and check it on cubic SrTiO3.

setup_mac.sh runs this once per environment. Re-run any time with
    ~/.sto-envs/<family>/bin/python scripts/02_smoke_test.py --family <family>

For each model it reports: load time, the relaxed cubic lattice parameter a0, and the time per
force call on the 30-atom [111] cell, with a rough estimate of the full strain-scan runtime.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sto import analysis as an  # noqa: E402
from sto import cells, config, io, models  # noqa: E402
from sto import structures as st  # noqa: E402

STEPS_PER_RELAX = 150  # rough average for the strain scan
RELAX_PER_MODEL = len(config.STRAINS_PCT) * len(config.ORIENTATIONS) * 5  # Pm-3m + one tilted nonpolar + 3 polar seeds


def versions():
    out = {"python": platform.python_version(), "machine": platform.machine()}
    for mod in ("torch", "ase", "phonopy", "mace", "orb_models", "mattersim", "sevenn", "chgnet"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    models.add_selection_args(ap)
    args = ap.parse_args()
    specs = models.selected(args)
    v = versions()
    print("Versions:", ", ".join(f"{k} {x}" for k, x in v.items()))
    if v["machine"] != "arm64" and platform.system() == "Darwin":
        print("WARNING: Python is not running natively on Apple silicon (Rosetta). Expect it to be slow.")
    out_dir = config.RESULTS / "smoke"
    failed = []
    for spec in specs:
        res = {"model": spec.name, "family": spec.family, "versions": v}
        try:
            t0 = time.time()
            calc = spec.loader(args.device)
            res["load_seconds"] = time.time() - t0

            pos, sym, cell = an.cubic_srtio3(3.905)
            atoms = st.to_atoms(pos, sym, cell)
            atoms.calc = calc
            res["energy_per_fu_at_3.905"] = float(atoms.get_potential_energy())
            res["max_force_cubic"] = float(np.abs(atoms.get_forces()).max())
            res["stress_cubic_voigt"] = atoms.get_stress().tolist()

            r = st.relaxed_cubic_a0(calc)
            res.update(a0=r["a0"], a0_converged=r["converged"], a0_energy_per_fu=r["energy"])

            p30, s30, c30 = cells.film_cell("111", r["a0"])
            big = st.to_atoms(cells.rattle(p30, 0.01, seed=1), s30, c30)
            big.calc = calc
            big.get_forces()  # first call can include compilation
            big.positions += 1e-4  # force a fresh evaluation
            t1 = time.time()
            for _ in range(3):
                big.positions += 1e-4
                big.get_forces()
                big.get_stress()
            res["seconds_per_call_30_atoms"] = (time.time() - t1) / 3
            hours = res["seconds_per_call_30_atoms"] * STEPS_PER_RELAX * RELAX_PER_MODEL / 3600
            res["estimated_full_scan_hours"] = hours

            sane = 3.80 < r["a0"] < 4.05 and res["max_force_cubic"] < 1e-3
            res["sane"] = bool(sane)
            print(f"{'OK ' if sane else 'ODD'} {spec.name:22s} a0 = {r['a0']:.4f} A   "
                  f"load {res['load_seconds']:5.1f} s   {res['seconds_per_call_30_atoms'] * 1000:6.0f} ms/call   "
                  f"full scan ~{hours:4.1f} h")
            if not sane:
                print("    a0 outside 3.80-4.05 A or forces on cubic SrTiO3 not ~0: check this model before using it")
        except Exception as exc:
            failed.append(spec.name)
            res["error"] = f"{type(exc).__name__}: {exc}"
            res["traceback"] = traceback.format_exc()
            print(f"FAIL {spec.name}: {res['error']}")
        io.write_json(out_dir / f"{spec.name}.json", res)
    if failed:
        print(f"\n{len(failed)} model(s) failed: {', '.join(failed)}. Send me results/smoke/<model>.json.")
        return 1
    print(f"\nAll {len(specs)} model(s) in this environment work. Results in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
