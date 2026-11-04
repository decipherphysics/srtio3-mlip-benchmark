"""Scan a VASP archive and summarise each run's settings. Plain-text parsing, no ASE needed."""
from __future__ import annotations

import os
import re
from pathlib import Path

from . import config

RUN_MARKERS = ("OUTCAR", "vasprun.xml")


def find_runs(root) -> list[Path]:
    """Every directory under `root` that contains an OUTCAR or vasprun.xml."""
    runs = []
    for dirpath, _dirs, files in os.walk(root):
        if any(m in files for m in RUN_MARKERS):
            runs.append(Path(dirpath))
    return sorted(runs)


def parse_incar(path) -> dict:
    """INCAR as an upper-case dict; handles comments (# or !) and several tags per line separated by ';'."""
    tags = {}
    p = Path(path)
    if not p.exists():
        return tags
    for line in p.read_text(errors="replace").splitlines():
        line = re.split(r"[#!]", line, maxsplit=1)[0]
        for part in line.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                tags[k.strip().upper()] = v.strip()
    return tags


def parse_poscar_formula(path) -> dict:
    """Element counts from a VASP5 POSCAR/CONTCAR (species line followed by counts line)."""
    p = Path(path)
    if not p.exists():
        return {}
    lines = p.read_text(errors="replace").splitlines()
    if len(lines) < 7:
        return {}
    species, counts = lines[5].split(), lines[6].split()
    if not all(c.isdigit() for c in counts) or any(s.isdigit() for s in species):
        return {}  # VASP4 format without species line
    out = {}
    for s, c in zip(species, counts):
        out[s.split("_")[0]] = out.get(s.split("_")[0], 0) + int(c)
    return out


def parse_poscar_symbols(path) -> list[str]:
    """Per-atom symbols in POSCAR order (VASP5 format)."""
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text(errors="replace").splitlines()
    if len(lines) < 7:
        return []
    species, counts = lines[5].split(), lines[6].split()
    if not all(c.isdigit() for c in counts):
        return []
    return [s.split("_")[0] for s, c in zip(species, counts) for _ in range(int(c))]


def is_srtio3(formula: dict) -> bool:
    sr, ti, o = formula.get("Sr", 0), formula.get("Ti", 0), formula.get("O", 0)
    return sr > 0 and sr == ti and o == 3 * ti and len(formula) == 3


def parse_kpoints(path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    lines = [l.strip() for l in p.read_text(errors="replace").splitlines()]
    if len(lines) >= 4:
        return f"{lines[2]} {lines[3]}".strip()
    return " ".join(lines)


def parse_outcar(path) -> dict:
    """Key facts from an OUTCAR: POTCARs, final energies, ionic steps, convergence."""
    p = Path(path)
    out = {"potcars": "", "final_free_energy": None, "final_energy_sigma0": None,
           "n_ionic_steps": 0, "reached_accuracy": False, "finished": False, "nions": None}
    if not p.exists():
        return out
    titles = []
    free = sigma0 = None
    steps = 0
    with p.open(errors="replace") as fh:
        for line in fh:
            if "TITEL" in line:
                titles.append(line.split("=", 1)[1].strip())
            elif "free  energy   TOTEN" in line:
                steps += 1
                try:
                    free = float(line.split("=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif "energy(sigma->0)" in line:
                try:
                    sigma0 = float(line.rsplit("=", 1)[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif "reached required accuracy" in line:
                out["reached_accuracy"] = True
            elif "General timing" in line:  # printed only when VASP ends normally
                out["finished"] = True
            elif "NIONS" in line and out["nions"] is None:
                m = re.search(r"NIONS\s*=\s*(\d+)", line)
                if m:
                    out["nions"] = int(m.group(1))
    out.update(potcars="; ".join(titles), final_free_energy=free,
               final_energy_sigma0=sigma0, n_ionic_steps=steps)
    return out


_MODE = re.compile(r"^\s*(\d+)\s+(f/i|f)\s*=\s*([-\d.Ee+]+)\s+THz.*?([-\d.Ee+]+)\s+cm-1")
THZ_TO_CM1 = 33.35641


def parse_outcar_phonons(path):
    """Gamma-point modes from a VASP DFPT/finite-difference run (IBRION = 5-8).

    Reads only the first block headed 'Eigenvectors and eigenvalues of the dynamical matrix' (the
    eigenvectors proper). With NWRITE = 3 VASP repeats that heading for a block already divided by
    sqrt(mass); that one is skipped, since the mode analysis divides by sqrt(mass) itself. Returns (freqs_cm1, eigvecs) with imaginary
    modes as negative frequencies and eigvecs of shape (n_modes, n_atoms, 3); (None, None) if absent.
    """
    p = Path(path)
    if not p.exists():
        return None, None
    lines = p.read_text(errors="replace").splitlines()
    starts = [i for i, l in enumerate(lines) if "Eigenvectors and eigenvalues of the dynamical matrix" in l]
    if not starts:
        return None, None
    freqs, vecs, cur = [], [], None
    for line in lines[starts[0] + 1:]:
        if ("Eigenvectors after division by SQRT(mass)" in line or "ELASTIC MODULI" in line
                or "Eigenvectors and eigenvalues of the dynamical matrix" in line):
            break
        m = _MODE.match(line)
        if m:
            if cur is not None:
                vecs.append(cur)
            f = float(m.group(4))
            freqs.append(-f if m.group(2) == "f/i" else f)
            cur = []
            continue
        if cur is not None:
            parts = line.split()
            if len(parts) == 6:
                try:
                    cur.append([float(x) for x in parts[3:]])
                except ValueError:
                    pass
    if cur is not None:
        vecs.append(cur)
    if not freqs:
        return None, None
    return freqs, vecs


def functional_label(incar: dict, potcars: str) -> str:
    """Best guess at the exchange-correlation functional from INCAR tags and POTCAR titles."""
    meta = incar.get("METAGGA", "").upper()
    if meta:
        return {"R2SCAN": "r2SCAN", "SCAN": "SCAN"}.get(meta, f"METAGGA={meta}")
    if incar.get("LHFCALC", "").upper().startswith((".T", "T")):
        return "hybrid (HSE/PBE0, check HFSCREEN)"
    gga = incar.get("GGA", "").upper()
    if gga == "PS":
        return "PBEsol"
    if gga in ("PE", "PBE"):
        return "PBE"
    if gga:
        return f"GGA={gga}"
    if "PBE" in potcars.upper():
        return "PBE (from POTCAR, no GGA tag)"
    if potcars:
        return "LDA (check: no GGA tag and non-PBE POTCARs)"
    return "unknown"


_ORIENT = re.compile(r"(?<!\d)(001|111)(?!\d)")
_STRAIN = re.compile(r"(?:strain|eps)[_=]?([+-]?\d+(?:[.p]\d+)?)", re.IGNORECASE)
# vasp_STO_runs layout: Epistrain/<001|111>/<in-plane a, A>/<phase>/...  and  Relaxation/<phase>/...
_LAYOUT = re.compile(r"(?:^|/)Epistrain/(001|111)/(\d+\.\d+)(?:/|$)")


def guess_orientation_and_strain(path) -> tuple[str, str]:
    """Guess orientation and strain (%) from folder names; leave blank when unsure. Edit by hand after.

    In the vasp_STO_runs layout the strain is (a / DFT_A0_CUBIC - 1) x 100 from the lattice-parameter
    folder, and runs under Relaxation/ are 'bulk' with no strain.
    A '-' straight after 'strain'/'eps' is read as a minus sign (eps-0.5 -> -0.5), so check the
    sign column in the audit CSV if your folders used '-' as a separator.
    """
    text = Path(path).as_posix()
    m = _LAYOUT.search(text)
    if m:
        return m.group(1), f"{(float(m.group(2)) / config.DFT_A0_CUBIC - 1) * 100:.3f}"
    if re.search(r"(?:^|/)Relaxation(?:/|$)", text):
        return "bulk", ""
    o = _ORIENT.findall(text)
    s = _STRAIN.findall(Path(path).name) or _STRAIN.findall(text)
    strain = s[-1].replace("p", ".") if s else ""
    return (o[-1] if o else ""), strain


def run_type(incar: dict) -> str:
    """'phonon' (IBRION 5-8), 'relax' (NSW > 0 with IBRION 1-3) or 'static'."""
    def _int(key, default):
        try:
            return int(float(incar.get(key, default)))
        except ValueError:
            return default
    ibrion, nsw = _int("IBRION", -1), _int("NSW", 0)
    if ibrion in (5, 6, 7, 8):
        return "phonon"
    if nsw > 0 and ibrion in (1, 2, 3):
        return "relax"
    return "static"


def default_include(row: dict, formula: dict) -> str:
    """'y' for SrTiO3 runs with a final energy: relaxations that converged, other runs that finished."""
    ok = is_srtio3(formula) and row["final_free_energy"] is not None
    ok = ok and (row["converged"] if row["run_type"] == "relax" else row.get("finished", True))
    return "y" if ok else "n"


def audit_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    incar = parse_incar(run_dir / "INCAR")
    outcar = parse_outcar(run_dir / "OUTCAR")
    formula = parse_poscar_formula(run_dir / "CONTCAR") or parse_poscar_formula(run_dir / "POSCAR")
    orient, strain = guess_orientation_and_strain(run_dir)
    ldau = incar.get("LDAU", "").upper().startswith((".T", "T"))
    row = _audit_row(run_dir, incar, outcar, formula, orient, strain, ldau)
    row["include"] = default_include(row, formula)
    text = run_dir.as_posix() + "/"
    for rel, reason in config.EXCLUDE.items():
        if f"/{rel}/" in text or text.startswith(f"{rel}/"):
            row["include"], row["notes"] = "n", f"excluded: {reason}"
    return row


def _audit_row(run_dir, incar, outcar, formula, orient, strain, ldau) -> dict:
    return {
        "path": str(run_dir),
        "run_type": run_type(incar),
        "is_srtio3": is_srtio3(formula),
        "formula": " ".join(f"{k}{v}" for k, v in formula.items()),
        "natoms": sum(formula.values()) if formula else outcar["nions"],
        "functional": functional_label(incar, outcar["potcars"]),
        "encut": incar.get("ENCUT", ""),
        "kpoints": parse_kpoints(run_dir / "KPOINTS") or incar.get("KSPACING", ""),
        "ispin": incar.get("ISPIN", "1"),
        "ldau": ldau,
        "ldauu": incar.get("LDAUU", "") if ldau else "",
        "ibrion": incar.get("IBRION", ""),
        "nsw": incar.get("NSW", ""),
        "isif": incar.get("ISIF", ""),
        "lepsilon_dfpt": incar.get("LEPSILON", "").upper().startswith((".T", "T")),
        "berry_phase": incar.get("LCALCPOL", "").upper().startswith((".T", "T"))
        or incar.get("LBERRY", "").upper().startswith((".T", "T")),
        "final_free_energy": outcar["final_free_energy"],
        "final_energy_sigma0": outcar["final_energy_sigma0"],
        "n_ionic_steps": outcar["n_ionic_steps"],
        "converged": outcar["reached_accuracy"],
        "finished": outcar["finished"],
        "has_vasprun": (run_dir / "vasprun.xml").exists(),
        "potcars": outcar["potcars"],
        "orientation": orient,
        "strain_percent": strain,
        "include": "",
        "notes": "",
    }
