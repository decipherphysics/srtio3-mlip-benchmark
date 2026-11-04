"""Step 4: the main experiment. Each model relaxes SrTiO3 under epitaxial [001] and [111] strain.

    caffeinate -i ./run.sh scripts/04_strain_relax.py            # everything (hours; resumable)
    ./run.sh scripts/04_strain_relax.py --quick                  # 5 strains, 2 polar seeds: a first look
    ~/.sto-envs/mace/bin/python scripts/04_strain_relax.py --models mace-mpa-0 --orientations 001
    ~/.sto-envs/mace/bin/python scripts/04_strain_relax.py --redescribe   # remeasure saved results, no relaxing

At every strain point, for every model:
  1. nonpolar: centrosymmetric references, each relaxed with its symmetry kept (ASE FixSymmetry):
     Pm-3m (untilted) at every strain, I4/mcm (a0a0c- tilt) under compression and Imma (a-a-c0)
     under tension, both at 0 %, and for [001] C2/c (a-a-a-) at 0 %, as in the DFT runs at 3.90 A.
     The lowest is the reference for the polar energy gain.
  2. polar: three starts from the lowest nonpolar structure with Ti nudged along the normal
     and two in-plane directions, relaxed with no symmetry constraint.
a1 and a2 are held exactly at a_ref * (1 + strain), as in the DFT runs; a3 relaxes in all three
components (length and tilt), so the out-of-plane stresses go to zero.
a_ref is the model's own cubic a0 (default, --strain-ref own) or the DFT a0 (--strain-ref dft).

Each relaxation writes results/relax/<model>/<orientation>/<strain>_<seed>.json and .extxyz.
Points already done are skipped, so the run can be stopped and restarted at any time.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase.io import read, write  # noqa: E402

from sto import analysis as an  # noqa: E402
from sto import cells, config, io, models  # noqa: E402
from sto import structures as st  # noqa: E402


def _f(stem: Path, ext: str) -> Path:
    """stem + extension. Not Path.with_suffix: strain tags such as '+2.00' contain a dot."""
    return stem.parent / (stem.name + ext)


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def nonpolar_phases(orientation, strain_pct) -> list[str]:
    """Centrosymmetric references at this strain (cells.NONPOLAR_TILTS): untilted at every strain,
    I4/mcm-type tilt under compression, Imma-type under tension, both near 0 % (the grid point at
    a = 3.90 A is +0.008 %). For [001] also a-a-a- (C2/c) near 0 %: the lowest centrosymmetric
    phase of the DFT runs at 3.90 A, 0.2 meV/f.u. below I4/mcm."""
    out = ["Pm-3m"]
    if strain_pct < 0.05:
        out.append("I4mcm")
    if strain_pct > -0.05:
        out.append("Imma")
    if orientation == "001" and abs(strain_pct) < 0.05:
        out.append("C2c")
    return out


def model_a0(spec, calc, out_dir: Path) -> float:
    f = out_dir / "a0.json"
    if f.exists():
        return io.read_json(f)["a0"]
    r = st.relaxed_cubic_a0(calc)
    io.write_json(f, {"model": spec.name, "a0": r["a0"], "energy_per_fu": r["energy"], "converged": r["converged"]})
    return r["a0"]


def save(r, spec, orientation, strain_pct, seed_name, a_target, a0_model, a_ref, n_fu, path_stem: Path, extra=None):
    a = r["atoms"]
    rec = {"model": spec.name, "family": spec.family, "functional": spec.functional, "dataset": spec.dataset,
           "orientation": orientation, "strain_pct": strain_pct, "seed": seed_name, "a_target": a_target,
           "a0_model": a0_model, "a_ref": a_ref, "n_fu": n_fu, "natoms": len(a), "energy": r["energy"],
           "e_per_fu": r["energy"] / n_fu, "converged": r["converged"], "nsteps": r["nsteps"],
           "seconds": r["seconds"], "fmax_final": r["fmax_final"], "stress_voigt": r["stress_voigt"],
           "plane_tilt_deg": r.get("plane_tilt_deg", 0.0)}
    rec.update(st.describe(a, orientation, n_fu))
    rec["inplane_drift"] = abs(rec["a_inplane"] - a_target)
    if extra:
        rec.update(extra)
    a.info.update(model=spec.name, orientation=f"o{orientation}", strain_pct=strain_pct, seed=seed_name,
                  model_energy=r["energy"])
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    write(_f(path_stem, ".extxyz"), a, format="extxyz")
    io.write_json(_f(path_stem, ".json"), rec)
    return rec


def redescribe(base: Path) -> int:
    """Recompute the structure-derived fields of every saved result under `base` (Ti displacement,
    tilts, c/a) from its .extxyz with the current sto.analysis. Nothing is relaxed; energies and
    the rest of each record are kept. For use after a measure changes."""
    files = sorted(base.glob("*/*/*.json"))
    for f in files:
        rec = io.read_json(f)
        a = read(str(f)[:-5] + ".extxyz")
        rec.update(st.describe(a, rec["orientation"], rec["n_fu"]))
        rec["inplane_drift"] = abs(rec["a_inplane"] - rec["a_target"])
        io.write_json(f, rec)
    print(f"{len(files)} results in {base} remeasured")
    return 0


def dft_seed_structures(orientation, strain_pct):
    """Polar final structures from the DFT at this orientation and strain (from 01_convert.py)."""
    if not config.DFT_FRAMES.exists():
        return []
    out = []
    for fr in read(config.DFT_FRAMES, index=":"):
        if not fr.info.get("is_final", False):
            continue
        if config.orientation_str(fr.info.get("orientation", "")) != orientation:
            continue
        s = fr.info.get("strain_percent", np.nan)
        if s is None or np.isnan(s) or abs(s - strain_pct) > 1e-6:
            continue
        p = an.polar_summary(fr.get_positions(), fr.get_chemical_symbols(), np.array(fr.cell))
        if p["ti_disp_polar"] > config.D_TOL:
            out.append(fr)
    return out


def run_point(spec, calc, orientation, strain_pct, a_ref, a0_model, args, base_dir):
    tag = config.strain_tag(strain_pct)
    odir = base_dir / orientation
    n_fu = cells.N_FU[orientation]
    a_t = a_ref * (1 + strain_pct / 100)
    common = (spec, orientation, strain_pct)

    # 1. centrosymmetric references; the lowest is what the polar runs are measured against
    refs = []
    for phase in nonpolar_phases(orientation, strain_pct):
        name = f"nonpolar_{phase}"
        stem = odir / f"{tag}_{name}"
        if _f(stem, ".json").exists() and _f(stem, ".extxyz").exists() and not args.overwrite:
            refs.append((io.read_json(_f(stem, ".json")), read(_f(stem, ".extxyz"))))
            continue
        pos, sym, cell = cells.film_cell(orientation, a0_model, tilt=phase, tilt_deg=config.NONPOLAR_TILT_DEG)
        pos, cell, _ = cells.strain_cell(pos, cell, orientation, a_t)
        r = st.relax(st.to_atoms(pos, sym, cell), calc, film=True, symmetric=True, fmax=args.fmax)
        rec = save(r, *common, name, a_t, a0_model, a_ref, n_fu, stem)
        report(rec)
        refs.append((rec, r["atoms"]))
    rec_np, base = min(refs, key=lambda x: x[0]["e_per_fu"])

    # 2. polar starts
    bpos, bsym, bcell = base.get_positions(), base.get_chemical_symbols(), np.array(base.cell)
    dirs = cells.seed_directions(bcell, orientation)
    if args.quick:
        dirs = dict(list(dirs.items())[:2])
    starts = [(name, cells.polar_seed(bpos, bsym, d, config.POLAR_SEED_AMP), bsym, bcell, None)
              for name, d in dirs.items()]
    if args.dft_seeds:
        for k, fr in enumerate(dft_seed_structures(orientation, strain_pct)):
            p, c = cells.to_film_frame(fr.get_positions(), np.array(fr.cell))  # a1, a2 into the xy-plane
            p, c, _ = cells.strain_cell(p, c, orientation, a_t, poisson=0.0)
            starts.append((f"dft{k}", p, fr.get_chemical_symbols(), c, {"dft_seed_path": fr.info.get("run_path", "")}))
    for name, pos, sym, cell, extra in starts:
        stem = odir / f"{tag}_{name}"
        if _f(stem, ".json").exists() and not args.overwrite:
            continue
        pos = cells.rattle(pos, 0.002, seed=_seed(spec.name, orientation, tag, name))
        n_fu_s = len(sym) // 5
        r = st.relax(st.to_atoms(pos, sym, cell), calc, film=True, symmetric=False, fmax=args.fmax)
        rec = save(r, *common, name, a_t, a0_model, a_ref, n_fu_s, stem, extra)
        rec["gain_vs_nonpolar_mev"] = an.polar_energy_gain(rec_np["e_per_fu"], rec["e_per_fu"], 1)
        rec["nonpolar_ref"] = rec_np["seed"]
        io.write_json(_f(stem, ".json"), rec)
        report(rec)


def report(rec):
    gain = rec.get("gain_vs_nonpolar_mev")
    g = f"gain {gain:7.3f} meV/fu" if gain is not None else " " * 20
    flag = "" if rec["converged"] else "  NOT CONVERGED"
    print(f"  {rec['model']:20s} [{rec['orientation']}] {rec['strain_pct']:+5.2f}% {rec['seed']:13s} "
          f"P(Ti) {rec['ti_disp_polar']:.4f} A  tilt {rec['tilt_all']:5.2f} deg  c/a {rec['c_over_a']:.4f}  "
          f"{g}  {rec['nsteps']:4d} steps {rec['seconds']:6.1f} s{flag}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    models.add_selection_args(ap)
    ap.add_argument("--orientations", nargs="+", default=list(config.ORIENTATIONS), choices=config.ORIENTATIONS)
    ap.add_argument("--strains", nargs="+", type=float, help="in-plane strains in %% (default: the DFT grid, a = 3.70 ... 4.10 A, see config.STRAINS_PCT)")
    ap.add_argument("--quick", action="store_true", help="5 strains and 2 polar seeds")
    ap.add_argument("--strain-ref", choices=("own", "dft"), default="own",
                    help="strain relative to each model's own cubic a0 (default) or to the DFT a0")
    ap.add_argument("--a0-dft", type=float, help="DFT cubic a0 in A (default: read from data/dft_a0.json)")
    ap.add_argument("--dft-seeds", action="store_true", help="also start from the DFT polar structures")
    ap.add_argument("--fmax", type=float, default=config.FMAX)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--redescribe", action="store_true",
                    help="recompute Ti displacement, tilts and c/a of the saved results (no relaxing), then stop")
    args = ap.parse_args()
    if args.redescribe:
        return redescribe(config.RESULTS / "relax" / ("own_a0" if args.strain_ref == "own" else "dft_a0"))
    strains = args.strains or (config.QUICK_STRAINS_PCT if args.quick else config.STRAINS_PCT)

    a0_dft = args.a0_dft
    if args.strain_ref == "dft" and a0_dft is None:
        if not config.DFT_A0_JSON.exists():
            ap.error("no data/dft_a0.json yet: run 01_convert.py or pass --a0-dft")
        a0_dft = {o: v["median"] for o, v in io.read_json(config.DFT_A0_JSON).items()}

    for spec in models.selected(args):
        print(f"\n=== {spec.name} ({spec.dataset}, {spec.functional}) ===", flush=True)
        try:
            calc = spec.loader(args.device)
        except Exception as exc:
            print(f"could not load {spec.name}: {exc}")
            continue
        ref_name = "own_a0" if args.strain_ref == "own" else "dft_a0"
        base_dir = config.RESULTS / "relax" / ref_name / spec.name
        base_dir.mkdir(parents=True, exist_ok=True)
        a0_model = model_a0(spec, calc, config.RESULTS / "relax" / "a0" / spec.name)
        print(f"  cubic a0 = {a0_model:.4f} A")
        t0 = time.time()
        for orientation in args.orientations:
            if args.strain_ref == "own":
                a_ref = a0_model
            else:
                a_ref = a0_dft[orientation] if isinstance(a0_dft, dict) else a0_dft
            for s in strains:
                try:
                    run_point(spec, calc, orientation, s, a_ref, a0_model, args, base_dir)
                except Exception:
                    print(f"  FAILED {spec.name} [{orientation}] {s:+.2f}%")
                    traceback.print_exc()
        print(f"  {spec.name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
