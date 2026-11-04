"""Step 0: scan a VASP archive and write data/audit.csv, one row per run.

    python scripts/00_audit.py /path/to/vasp/archive

Open data/audit.csv afterwards and check three columns by hand:
  orientation     001 or 111 (guessed from folder names)
  strain_percent  in-plane strain in % (guessed from folder names; check the sign)
  include         y / n  (default y for converged SrTiO3 runs)
Only plain text is read (INCAR, OUTCAR, POSCAR/CONTCAR, KPOINTS), so this runs in any environment.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sto import config  # noqa: E402
from sto import vasp_audit as va  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", nargs="?", default=config.ARCHIVE, help="archive folder (or set STO_ARCHIVE)")
    ap.add_argument("--out", default=str(config.AUDIT_CSV))
    args = ap.parse_args()
    if not args.archive:
        ap.error("give the archive folder, e.g.  python scripts/00_audit.py ~/Documents/vasp_runs")
    root = Path(args.archive).expanduser().resolve()
    if not root.is_dir():
        ap.error(f"{root} is not a folder")

    runs = va.find_runs(root)
    if not runs:
        print(f"No OUTCAR or vasprun.xml found under {root}")
        return 1
    rows = []
    for r in runs:
        try:
            rows.append(va.audit_run(r))
        except Exception as exc:  # keep going; one broken run should not stop the audit
            print(f"  could not read {r}: {exc}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sto = [r for r in rows if r["is_srtio3"]]
    inc = [r for r in rows if r["include"] == "y"]
    print(f"\n{len(rows)} runs found, {len(sto)} are SrTiO3, {len(inc)} marked include=y")
    print(f"Written to {out}\n")

    def show(title, values):
        c = Counter(values)
        print(f"{title}:")
        for k, v in c.most_common():
            print(f"   {k or '(blank)':<40} {v}")

    show("Functional (SrTiO3 runs)", [r["functional"] for r in sto])
    show("ENCUT", [r["encut"] for r in sto])
    show("k-points", [r["kpoints"] for r in sto])
    show("Run type", [r["run_type"] for r in sto])
    show("Orientation guess", [r["orientation"] for r in sto])
    show("Strain guess (%)", [r["strain_percent"] for r in sto])
    show("Spin polarised (ISPIN)", [r["ispin"] for r in sto])
    print(f"DFT+U used in {sum(r['ldau'] for r in sto)} SrTiO3 runs")
    print(f"Phonon runs (IBRION 5-8): {sum(r['run_type'] == 'phonon' for r in sto)}"
          f", of which DFPT (LEPSILON): {sum(r['lepsilon_dfpt'] for r in sto)}")
    print(f"Berry-phase polarisation runs: {sum(r['berry_phase'] for r in sto)}")
    unconverged = [r for r in sto if r["run_type"] == "relax" and not r["converged"]]
    if unconverged:
        print(f"\n{len(unconverged)} SrTiO3 relaxations did not reach the force criterion (include=n):")
        for r in unconverged[:10]:
            print("   ", r["path"])
    blanks = [r for r in inc if not r["orientation"] or (not r["strain_percent"] and r["orientation"] != "bulk")]
    if blanks:
        print(f"\nFill in orientation / strain_percent by hand for {len(blanks)} included runs, e.g.:")
        for r in blanks[:10]:
            print("   ", r["path"])
    if len({r["functional"] for r in sto}) > 1:
        print("\nWARNING: more than one functional in the archive. Keep one (set include=n on the rest).")
    if len({r["encut"] for r in inc}) > 1:
        print("WARNING: included runs use different ENCUT values; energies are only comparable at one cutoff.")
    print("\nNext: check data/audit.csv, then run  python scripts/01_convert.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
