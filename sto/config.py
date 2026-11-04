"""Paths and run settings. Environment variables override the values marked (env)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS_DIR = ROOT / "models"  # fine-tuned MACE models go here (*.model)

ARCHIVE = os.environ.get("STO_ARCHIVE", str(ROOT / "vasp_STO_runs"))  # (env) folder holding the VASP runs
DEVICE = os.environ.get("STO_DEVICE", "cpu")  # (env) CPU on Apple silicon: MPS has no float64

AUDIT_CSV = DATA / "audit.csv"
DFT_FRAMES = DATA / "dft_frames.extxyz"
DFT_FINAL_CSV = DATA / "dft_final.csv"
DFT_A0_JSON = DATA / "dft_a0.json"

DFT_A0_CUBIC = 3.8997  # A, PBEsol cubic Pm-3m relaxation (vasp_STO_runs/Relaxation/Pm-3m); strain reference

# runs the audit always marks include=n (path inside the archive, sub-runs included): reason
EXCLUDE = {
    # e.g. "Epistrain/001/3.90/Ima2": "reason". The drive's wrong-plane Ima2 series was never copied
    # (its fixed cell plane is (110)-type); see vasp_STO_runs/README.md.
}

ORIENTATIONS = ("001", "111")
# the in-plane lattice parameters of the DFT runs (A); the strain grid is these, relative to DFT_A0_CUBIC,
# rounded as 00_audit.py writes them, so model and DFT points line up
DFT_LATTICE_PARAMS = [3.70, 3.75, 3.80, 3.85, 3.90, 3.95, 4.00, 4.05, 4.10]
STRAINS_PCT = [round((a / DFT_A0_CUBIC - 1) * 100, 3) for a in DFT_LATTICE_PARAMS]
QUICK_STRAINS_PCT = STRAINS_PCT[2:7]  # 3.80 ... 4.00 A

FMAX = 2e-3  # eV/A, relaxation force threshold (tight: polar energy gains can be < 1 meV/f.u.)
# eV/A, for relaxations that follow an unstable phonon (05_phonons.py): there the landscape is so flat
# that 2 meV/A can leave soft distortions of ~0.01-0.08 A (a 20 cm-1 O mode has k ~ 0.02 eV/A^2)
FMAX_FOLLOW = 2e-4
MAX_STEPS = 3000
NONPOLAR_TILT_DEG = 4.0  # deg, starting antiphase tilt of the I4/mcm and Imma references (DFT: ~4-5 deg)
POLAR_SEED_AMP = 0.04  # A, Ti off-centring used to start each polar run
E_TOL_MEV = 0.1  # meV/f.u., minimum polar energy gain to call a point polar
# meV/f.u., second line: the estimated scale of quantum fluctuations in SrTiO3, which keep bulk STO
# paraelectric (Watanabe, PRB 99, 064107 (2019); my PhD thesis, ch. 3). A gain below it is polar at
# 0 K but may not be in experiment. An estimate, not a calculation of zero-point motion.
E_QF_MEV = 2.0
D_TOL = 0.005  # A, minimum uniform Ti displacement to call a point polar


def strain_tag(strain_pct: float) -> str:
    return f"{strain_pct:+.2f}"


def orientation_str(value) -> str:
    """extxyz turns '001' into the integer 1 on reading; this undoes that."""
    s = str(value).strip().strip("[]")
    try:
        return f"{int(float(s)):03d}"
    except ValueError:
        return s


def archive_relative(path) -> str:
    """A run's folder inside ARCHIVE, as written to the data files (no local folder names)."""
    try:
        return str(Path(path).relative_to(ARCHIVE))
    except ValueError:
        return str(path)
