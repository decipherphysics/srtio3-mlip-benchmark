"""[111] scan: relaxations of SrTiO3 films with one model (MACE MH-1, matpes_r2scan head) at the nine
strains of the [001] study, measured from the model's own cubic a0.

At each strain:
  1. the seven centrosymmetric references of sto111.REFERENCES, with their symmetry held;
  2. polar starts on the two lowest distinct references: every Ti moved 0.04 A along each of six
     directions (sto111.seed_directions), then relaxed without symmetry.
Force limit 2 meV/A, as in the [001] study. Resumable: a run whose .json exists is skipped.

    ~/.sto-envs/mace/bin/python orientations/111/scan_111.py              # all nine strains
    ~/.sto-envs/mace/bin/python orientations/111/scan_111.py --quick      # one strain, three references, two starts
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sto111 as S  # noqa: E402


def save(r, s, name, a_t, a0, outdir: Path, extra=None) -> dict:
    a = r["atoms"]
    rec = {"model": S.MODEL, "orientation": S.ORIENT, "strain_pct": s, "seed": name, "a_target": a_t,
           "a0_model": a0, "n_fu": S.N_FU, "natoms": len(a), "energy": r["energy"],
           "e_per_fu": r["energy"] / S.N_FU, "converged": r["converged"], "nsteps": r["nsteps"],
           "seconds": r["seconds"], "fmax_final": r["fmax_final"], "stress_voigt": r["stress_voigt"],
           "plane_tilt_deg": r.get("plane_tilt_deg", 0.0)}
    rec.update(S.describe(a))
    rec["inplane_drift"] = abs(rec["a_inplane"] - a_t)
    if extra:
        rec.update(extra)
    a.info.update(model=S.MODEL, orientation=f"o{S.ORIENT}", strain_pct=s, seed=name, model_energy=r["energy"])
    tag = S.config.strain_tag(s)
    write(outdir / f"{tag}_{name}.extxyz", a, format="extxyz")
    S.io.write_json(outdir / f"{tag}_{name}.json", rec)
    return rec


TILT_KEYS = tuple(f"tilt_about_{k}" for k in S.FILM_AXES)


def report(rec):
    gain = rec.get("gain_vs_nonpolar_mev")
    g = f"gain {gain:8.3f} meV/fu" if gain is not None else " " * 21
    flag = "" if rec["converged"] else "  NOT CONVERGED"
    d = "/".join(f"{rec[k]:+.3f}" for k in ("d_1-10", "d_11-2", "d_111"))
    t = "/".join(f"{rec[k]:.1f}" for k in TILT_KEYS)
    print(f"  {rec['strain_pct']:+6.2f}% {rec['seed']:16s} E {rec['e_per_fu']:.5f} eV/fu  {g}  "
          f"Ti [1-10]/[11-2]/[111] {d} A  tilt {t} deg  {rec['nsteps']:4d} steps {rec['seconds']:6.1f} s{flag}",
          flush=True)


def distinct(refs, e_tol_mev=0.01, tilt_tol=0.05):
    """The references in order of energy, dropping any that relaxed into the same structure as a lower one."""
    out = []
    for rec, at in sorted(refs, key=lambda x: x[0]["e_per_fu"]):
        same = any(1000 * abs(rec["e_per_fu"] - r["e_per_fu"]) < e_tol_mev
                   and all(abs(rec[k] - r[k]) < tilt_tol for k in TILT_KEYS) for r, _ in out)
        if not same:
            out.append((rec, at))
    return out


def run_strain(calc, a0, s, outdir: Path, args):
    tag = S.config.strain_tag(s)
    a_t = a0 * (1 + s / 100)
    names = ["untilted", "t111", "t001"] if args.quick else list(S.REFERENCES)

    refs = []
    for name in names:  # 1. centrosymmetric references, symmetry held
        stem = outdir / f"{tag}_np_{name}"
        if Path(f"{stem}.json").exists() and not args.overwrite:
            refs.append((S.io.read_json(Path(f"{stem}.json")), read(f"{stem}.extxyz")))
            continue
        pos, sym, cell = S.reference_cell(name, a0, a_t)
        r = S.st.relax(S.st.to_atoms(pos, sym, cell), calc, film=True, symmetric=True, fmax=args.fmax)
        rec = save(r, s, f"np_{name}", a_t, a0, outdir)
        report(rec)
        refs.append((rec, r["atoms"]))
    ordered = distinct(refs)
    e_np = ordered[0][0]["e_per_fu"]

    for brec, base in ordered[: (1 if args.quick else 2)]:  # 2. polar starts on the two lowest
        bname = brec["seed"][3:]
        dirs = S.seed_directions(np.array(base.cell))
        if args.quick:
            dirs = {k: dirs[k] for k in ("P111", "P1-10")}
        for dname, d in dirs.items():
            name = f"{bname}_{dname}"
            if Path(f"{outdir / f'{tag}_{name}'}.json").exists() and not args.overwrite:
                continue
            sym = base.get_chemical_symbols()
            pos = S.cells.polar_seed(base.get_positions(), sym, d, S.config.POLAR_SEED_AMP)
            pos = S.cells.rattle(pos, 0.002, seed=zlib.crc32(f"{S.MODEL}|111|{tag}|{name}".encode()))
            r = S.st.relax(S.st.to_atoms(pos, sym, np.array(base.cell)), calc, film=True, symmetric=False,
                           fmax=args.fmax)
            rec = save(r, s, name, a_t, a0, outdir, {
                "gain_vs_nonpolar_mev": S.an.polar_energy_gain(e_np, r["energy"] / S.N_FU, 1),
                "nonpolar_ref": ordered[0][0]["seed"], "start_ref": brec["seed"], "start_dir": dname})
            report(rec)


def model_a0(calc) -> float:
    """The model's cubic a0 from the [001] study (read only); computed here if it is missing."""
    f = S.REPO / "results" / "relax" / "a0" / S.MODEL / "a0.json"
    if f.exists():
        return S.io.read_json(f)["a0"]
    return S.st.relaxed_cubic_a0(calc)["a0"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strains", nargs="+", type=float, help="in-plane strains in %% (default: the [001] grid)")
    ap.add_argument("--quick", action="store_true", help="dry run: one strain, three references, two starts")
    ap.add_argument("--fmax", type=float, default=S.config.FMAX)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    strains = args.strains or ([S.config.STRAINS_PCT[2]] if args.quick else S.config.STRAINS_PCT)
    outdir = S.HERE / ("results_dryrun" if args.quick else "results")
    outdir.mkdir(parents=True, exist_ok=True)

    calc = S.models.get(S.MODEL).loader("cpu")
    a0 = model_a0(calc)
    print(f"=== {S.MODEL}, [{S.ORIENT}], cubic a0 = {a0:.4f} A, strains {strains}", flush=True)
    t0 = time.time()
    for s in strains:
        try:
            run_strain(calc, a0, s, outdir, args)
        except Exception:
            print(f"  FAILED at {s:+.2f}%", flush=True)
            traceback.print_exc()
        print(f"  {s:+.2f}% done, {(time.time() - t0) / 60:.1f} min so far", flush=True)
    print(f"scan done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
