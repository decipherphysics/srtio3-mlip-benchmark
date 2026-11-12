"""Copy out what the comparison needs from the [111] SrTiO3 VASP runs (read only): for each top-level
relaxation <phase>/<a>/, the final structure (CONTCAR), the final energy E0 (sigma -> 0, from OSZICAR),
whether it reached the force limit, the number of ionic steps and the INCAR's IOPTCELL. Writes
dft/finals.extxyz and dft/finals.csv here; the runs themselves are not touched. The raw VASP files are not
in this repository, so this only reruns where they are present.

    ~/.sto-envs/mace/bin/python orientations/111/extract_dft.py [folder of the runs]

The folder defaults to Epistrain/111 inside the VASP archive ($STO_ARCHIVE, or vasp_STO_runs/).
"""
import csv
import os
import re
import sys
from pathlib import Path

from ase.io import read, write

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("STO_ARCHIVE", REPO / "vasp_STO_runs")) / "Epistrain" / "111"
OUT = Path(__file__).resolve().parent / "dft"


def last_e0(oszicar: Path):
    e0, n = None, 0
    for line in oszicar.read_text(errors="replace").splitlines():
        m = re.search(r"E0=\s*([-+.\dEe]+)", line)
        if m:
            e0, n = float(m.group(1)), n + 1
    return e0, n


def reached(outcar: Path) -> bool:
    with open(outcar, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 400_000))
        return b"reached required accuracy" in fh.read()


def main(source: Path = SOURCE):
    if not source.is_dir():
        sys.exit(f"no VASP runs at {source}")
    OUT.mkdir(exist_ok=True)
    frames, rows = [], []
    for run in sorted(p for p in source.glob("*/*") if p.is_dir()):
        files = {f: run / f for f in ("CONTCAR", "OSZICAR", "OUTCAR", "INCAR")}
        if not all(files[f].exists() for f in ("CONTCAR", "OSZICAR", "OUTCAR")):
            continue
        try:
            a = read(files["CONTCAR"], format="vasp")
        except Exception as exc:  # an empty or broken CONTCAR
            rows.append({"phase": run.parent.name, "a_label": run.name, "error": str(exc)})
            continue
        e0, nsteps = last_e0(files["OSZICAR"])
        ioptcell = ""
        if files["INCAR"].exists():
            m = re.search(r"^\s*IOPTCELL\s*=\s*([\d\s]+)", files["INCAR"].read_text(errors="replace"), re.M)
            ioptcell = " ".join(m.group(1).split()) if m else ""
        info = {"phase": run.parent.name, "a_label": run.name, "e0": e0, "natoms": len(a),
                "converged": reached(files["OUTCAR"]), "nsteps": nsteps, "ioptcell": ioptcell,
                "path": str(run.relative_to(source))}
        a.info.update(info)
        frames.append(a)
        rows.append(info)
    write(OUT / "finals.extxyz", frames, format="extxyz")
    keys = ["phase", "a_label", "e0", "natoms", "converged", "nsteps", "ioptcell", "path", "error"]
    with open(OUT / "finals.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"{len(frames)} runs copied out to {OUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE)
