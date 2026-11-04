"""Step 5: Gamma-point phonons of the relaxed structures, with each mode labelled polar / tilt / other.

    ./run.sh scripts/05_phonons.py --orientations 001    # every strain of the DFT grid
    ~/.sto-envs/mace/bin/python scripts/05_phonons.py --dft    # the DFT phonon runs (DFPT)

At each strain: every centrosymmetric reference from step 4 and the lowest polar run. The supercells
(20 atoms for [001], 30 for [111]) fold the R point (and for [001] one M point) of the cubic cell onto
Gamma, so polar (soft TO) and octahedral-tilt modes both appear. Modes are labelled polar from their
Born-charge dipole, and tilt when the O motion is rigid rotation of the O6 cages.
An imaginary polar mode in a nonpolar reference is the instability that drives ferroelectricity;
near 0 % the Pm-3m and I4/mcm references compare with the bulk DFPT runs. The lowest polar run
should have no imaginary modes left. If it has one, it is a saddle, not the ground state: the
structure is displaced along each unstable mode and relaxed again (force limit config.FMAX_FOLLOW),
and the lowest result is saved as a new step-4 run, <strain>_polar_followed, with its phonons.
LO-TO splitting is not included (no Born charges), so compare with DFT TO modes.
Writes results/phonons/<ref>/<model>/<orientation>/<strain>_<seed>.json (role: lowest_nonpolar,
nonpolar, best_polar or saddle).
"""
from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase import Atoms  # noqa: E402
from ase.data import atomic_masses, atomic_numbers  # noqa: E402
from ase.io import read, write  # noqa: E402

from sto import analysis as an  # noqa: E402
from sto import config, io, models  # noqa: E402
from sto import structures as st  # noqa: E402
from sto import vasp_audit as va  # noqa: E402

THZ_TO_CM1 = 33.35641
ACOUSTIC_CM1 = 10.0
FOLLOW_AMP = 0.05  # A, largest atomic displacement when following an unstable mode
FOLLOW_ROUNDS = 3


def gamma_phonons(atoms, calc, delta=0.01, symprec=1e-3):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unit = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=np.array(atoms.cell),
                        scaled_positions=atoms.get_scaled_positions())
    # primitive_matrix must be explicit: phonopy >= 4 reads None as 'auto' and would shrink the cell,
    # losing exactly the folded R/M-point modes (tilts) this analysis needs.
    ph = Phonopy(unit, supercell_matrix=np.eye(3, dtype=int), primitive_matrix=np.eye(3), symprec=symprec)
    ph.generate_displacements(distance=delta, is_plusminus=True)
    forces = []
    for sc in ph.supercells_with_displacements:
        a = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        a.calc = calc
        f = a.get_forces()
        forces.append(f - f.mean(axis=0))  # remove any net drift
    ph.forces = np.array(forces)
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    ph.run_qpoints([[0.0, 0.0, 0.0]], with_eigenvectors=True)
    try:
        d = ph.get_qpoints_dict()
        freqs, vecs = np.asarray(d["frequencies"][0]), np.asarray(d["eigenvectors"][0])
    except AttributeError:  # older phonopy
        freqs, vecs = np.asarray(ph.qpoints.frequencies[0]), np.asarray(ph.qpoints.eigenvectors[0])
    n = len(atoms)
    if len(freqs) != 3 * n:
        raise RuntimeError(f"phonopy returned {len(freqs)} modes for {n} atoms: the cell was reduced")
    modes = [vecs[:, j].reshape(n, 3) for j in range(3 * n)]  # eigenvectors are columns, atom-major
    return freqs * THZ_TO_CM1, modes, np.asarray(ph.primitive.masses), len(ph.supercells_with_displacements)


def summarise(freqs, modes, masses, symbols, positions, cell):
    rows = []
    for f, e in zip(freqs, modes):
        rows.append({"freq_cm1": float(f), **an.mode_character(e, masses, symbols, positions, cell)})
    optical = [r for r in rows if r["label"] != "acoustic"]

    def lowest(label):
        v = [r["freq_cm1"] for r in optical if r["label"] == label]
        return min(v) if v else None

    return rows, {"lowest_polar_cm1": lowest("polar"), "lowest_tilt_cm1": lowest("tilt"),
                  "n_imaginary": sum(r["freq_cm1"] < -ACOUSTIC_CM1 for r in optical)}


def relax_along_unstable(atoms, calc, freqs, modes, masses):
    """Displace along each imaginary mode (both signs; for a degenerate pair also their sum), largest
    atomic displacement FOLLOW_AMP, and relax with the film constraint. Returns (lowest relaxation, its mode)."""
    imag = [i for i, f in enumerate(freqs) if f < -ACOUSTIC_CM1]
    vecs = [(np.real(modes[i]), freqs[i]) for i in imag]
    if len(vecs) == 2:
        vecs.append((vecs[0][0] + vecs[1][0], freqs[imag[0]]))
    best = None
    for e, f in vecs:
        u = e / np.sqrt(masses)[:, None]
        u *= FOLLOW_AMP / np.linalg.norm(u, axis=1).max()
        for sign in (1, -1):
            a = atoms.copy()
            a.positions += sign * u
            r = st.relax(a, calc, film=True, symmetric=False, fmax=config.FMAX_FOLLOW, steps=config.MAX_STEPS)
            if best is None or r["energy"] < best[0]["energy"]:
                best = (r, f)
    return best


def follow(calc, rec, e_nonpolar, atoms, freqs, modes, masses, rdir, pdir, tag):
    """The lowest polar run is a saddle. Relax along its unstable modes; if that goes lower, save the result
    as the step-4 run <tag>_polar_followed (record as written by 04) with its phonons, and repeat while the
    new structure is still unstable."""
    o, n_fu = rec["orientation"], rec["n_fu"]
    for _ in range(FOLLOW_ROUNDS):
        r, f = relax_along_unstable(atoms, calc, freqs, modes, masses)
        if r["energy"] / n_fu >= rec["e_per_fu"]:
            print(f"  following {rec['seed']} found nothing lower", flush=True)
            return
        a = r["atoms"]
        rec = {**rec, "seed": "polar_followed", "followed_from": rec.get("followed_from", rec["seed"]),
               "followed_mode_cm1": float(f), "energy": r["energy"], "e_per_fu": r["energy"] / n_fu,
               "converged": r["converged"], "nsteps": r["nsteps"], "seconds": r["seconds"],
               "fmax_final": r["fmax_final"], "stress_voigt": r["stress_voigt"],
               "plane_tilt_deg": r.get("plane_tilt_deg", 0.0), **st.describe(a, o, n_fu)}
        rec["inplane_drift"] = abs(rec["a_inplane"] - rec["a_target"])
        rec["gain_vs_nonpolar_mev"] = an.polar_energy_gain(e_nonpolar, rec["e_per_fu"], 1)
        a.info.update(model=rec["model"], orientation=f"o{o}", strain_pct=rec["strain_pct"], seed="polar_followed",
                      model_energy=r["energy"])
        write(rdir / f"{tag}_polar_followed.extxyz", a, format="extxyz")
        io.write_json(rdir / f"{tag}_polar_followed.json", rec)
        atoms = a
        freqs, modes, masses, ndisp = gamma_phonons(atoms, calc)
        table, summ = summarise(freqs, modes, masses, atoms.get_chemical_symbols(), atoms.get_positions(),
                                np.array(atoms.cell))
        role = "saddle" if summ["n_imaginary"] else "best_polar"
        io.write_json(pdir / f"{tag}_polar_followed.json", {"model": rec["model"], "orientation": o, "strain_pct": rec["strain_pct"],
                                                           "structure": "polar_followed", "role": role, "n_displacements": ndisp,
                                                           **summ, "modes": table})
        print(f"  {rec['model']:20s} [{o}] {tag} polar_followed  gain {rec['gain_vs_nonpolar_mev']:.3f} meV/fu  "
              f"P(Ti) {rec['ti_disp_polar']:.4f} A  imaginary {summ['n_imaginary']}", flush=True)
        if role == "best_polar":
            return


def run_dft():
    rows = [r for r in csv.DictReader(open(config.AUDIT_CSV)) if r["run_type"] == "phonon"
            and r["include"].strip().lower() in ("y", "yes", "1", "true")]
    print(f"{len(rows)} phonon runs marked include=y in the audit")
    out = []
    for r in rows:
        d = Path(r["path"])
        freqs, vecs = va.parse_outcar_phonons(d / "OUTCAR")
        if freqs is None or not (d / "POSCAR").exists():
            print(f"  no dynamical matrix (or no POSCAR) in {d}")
            continue
        struct = read(d / "POSCAR", format="vasp")
        sym = struct.get_chemical_symbols()
        masses = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
        modes = [np.array(v) for v in vecs]
        table, summ = summarise(freqs, modes, masses, sym, struct.get_positions(), np.array(struct.cell))
        rec = {"path": config.archive_relative(d), "structure": d.parent.name, "orientation": r["orientation"],
               "strain_percent": r["strain_percent"], **summ, "modes": table}
        out.append(rec)
        print(f"  {d.parent.parent.name}/{d.parent.name}: lowest polar {summ['lowest_polar_cm1']:.1f} cm-1, "
              f"lowest tilt {summ['lowest_tilt_cm1']} cm-1, imaginary {summ['n_imaginary']}")
    io.write_json(config.RESULTS / "phonons" / "dft.json", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    models.add_selection_args(ap)
    ap.add_argument("--orientations", nargs="+", default=list(config.ORIENTATIONS))
    ap.add_argument("--strains", nargs="+", type=float, default=config.STRAINS_PCT)
    ap.add_argument("--strain-ref", choices=("own", "dft"), default="own")
    ap.add_argument("--dft", action="store_true", help="only analyse the DFT phonon runs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.dft:
        run_dft()
        return 0
    ref = "own_a0" if args.strain_ref == "own" else "dft_a0"
    for spec in models.selected(args):
        try:
            calc = spec.loader(args.device)
        except Exception as exc:
            print(f"could not load {spec.name}: {exc}")
            continue
        for o in args.orientations:
            rdir = config.RESULTS / "relax" / ref / spec.name / o
            for s in args.strains:
                tag = config.strain_tag(s)
                recs = [io.read_json(p) for p in sorted(rdir.glob(f"{tag}_*.json"))]
                if not recs:
                    print(f"  {spec.name} [{o}] {tag}: no step-4 results yet")
                    continue
                nonpolar = [r for r in recs if r["seed"].startswith("nonpolar")]
                polar = [r for r in recs if not r["seed"].startswith("nonpolar")]
                lowest_np = min(nonpolar, key=lambda r: r["e_per_fu"]) if nonpolar else None
                todo = [(r, "lowest_nonpolar" if r is lowest_np else "nonpolar") for r in nonpolar]
                if polar:
                    todo.append((min(polar, key=lambda r: r["e_per_fu"]), "best_polar"))
                pdir = config.RESULTS / "phonons" / ref / spec.name / o
                for rec, role in todo:
                    out = pdir / f"{tag}_{rec['seed']}.json"
                    if out.exists() and not args.overwrite:
                        # done, unless it is an unstable lowest polar run not yet followed (results from before 29 Oct)
                        if not (role == "best_polar" and io.read_json(out)["n_imaginary"]
                                and rec["seed"] != "polar_followed"
                                and not (rdir / f"{tag}_polar_followed.json").exists()):
                            continue
                    try:
                        atoms = read(rdir / f"{tag}_{rec['seed']}.extxyz")
                        freqs, modes, masses, ndisp = gamma_phonons(atoms, calc)
                        table, summ = summarise(freqs, modes, masses, atoms.get_chemical_symbols(),
                                                atoms.get_positions(), np.array(atoms.cell))
                        if role == "best_polar" and summ["n_imaginary"]:
                            role = "saddle"
                        io.write_json(out, {"model": spec.name, "orientation": o, "strain_pct": s, "structure": rec["seed"],
                                            "role": role, "n_displacements": ndisp, **summ, "modes": table})
                        print(f"  {spec.name:20s} [{o}] {tag} {rec['seed']:15s} lowest polar {summ['lowest_polar_cm1']!s:>8.8} "
                              f"tilt {summ['lowest_tilt_cm1']!s:>8.8} cm-1  imaginary {summ['n_imaginary']}", flush=True)
                        if role == "saddle" and rec["seed"] != "polar_followed":
                            follow(calc, rec, lowest_np["e_per_fu"] if lowest_np else np.nan, atoms, freqs, modes,
                                   masses, rdir, pdir, tag)
                    except Exception:
                        print(f"  FAILED {spec.name} [{o}] {tag} {rec['seed']}")
                        traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
