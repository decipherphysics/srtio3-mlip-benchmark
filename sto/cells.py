"""Strained SrTiO3 film cells, built in plain numpy so they can be tested without ASE.

Film frame: a1 along x, a2 in the xy-plane, film normal along z. Strain is applied to a1 and a2;
a3 (out of plane) is left free for the relaxation.
"""
from __future__ import annotations

import numpy as np

from . import analysis as an

# Rows are new lattice vectors in units of the cubic ones.
#  001: sqrt2 x sqrt2 x 2 (20 atoms) - holds every out-of-phase (R-point) tilt, including a0a0c-
#       and a-a-c0, plus in-phase rotation about the normal (c+). Not a+ or b+ about in-plane axes.
#  111: hexagonal a = sqrt2 a_pc, c = 2 sqrt3 a_pc (30 atoms) - holds every R-point tilt, incl. a-a-a-.
SUPERCELL_MATRIX = {
    "001": [[1, 1, 0], [-1, 1, 0], [0, 0, 2]],
    "111": [[1, -1, 0], [0, 1, -1], [2, 2, 2]],
}
N_FU = {"001": 4, "111": 6}
POISSON_GUESS = 0.6  # c-strain ~ -0.6 x in-plane strain; only a starting guess, a3 is relaxed

# Centrosymmetric references: antiphase (R-point) tilt axes in pseudocubic x, y, z (z = [001]pc).
NONPOLAR_TILTS = {
    "Pm-3m": (0, 0, 0),  # a0a0a0, no tilt
    "I4mcm": (0, 0, 1),  # a0a0c-, tilt about [001]pc
    "Imma": (1, 1, 0),   # a-a-c0, equal tilts about [100]pc and [010]pc
    "C2c": (1, 1, 1),    # a-a-a-, equal tilts about all three (R-3c in bulk, C2/c under [001] strain)
}


def film_rotation(cell) -> np.ndarray:
    """Rotation q (new vectors = old vectors @ q) taking a1 to x and a2 into the xy-plane."""
    q, r = np.linalg.qr(np.asarray(cell, float).T)
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    return q * s  # flip columns so R has a positive diagonal -> proper rotation for right-handed cells


def to_film_frame(positions, cell):
    """Rigid rotation so that a1 || x, a2 lies in the xy-plane and a3 has positive z."""
    q = film_rotation(cell)
    return np.asarray(positions, float) @ q, np.asarray(cell, float) @ q


def film_cell(orientation: str, a0: float, tilt=None, tilt_deg: float = 0.0):
    """Unstrained SrTiO3 in the film supercell for `orientation`: (positions, symbols, cell).

    `tilt` is a NONPOLAR_TILTS key (or axes in pseudocubic x, y, z); each octahedron is then turned by
    `tilt_deg` about those axes, alternating in sign between neighbours (antiphase), so the result
    stays centrosymmetric."""
    pos, sym, cell = an.cubic_srtio3(a0)
    pos, sym, cell = an.supercell(pos, sym, cell, SUPERCELL_MATRIX[orientation])
    if tilt is not None and tilt_deg:
        axes = NONPOLAR_TILTS[tilt] if isinstance(tilt, str) else tilt
        pos = antiphase_tilt(pos, sym, a0, axes, tilt_deg)
    pos, cell = to_film_frame(pos, cell)
    return pos, sym, cell


def antiphase_tilt(positions, symbols, a0, axes, angle_deg):
    """Rotate every TiO6 octahedron of a cubic-frame supercell by angle_deg about `axes` (pseudocubic),
    with the sign alternating between neighbouring octahedra (R-point tilt). Small-angle (linear) form;
    the relaxation sets the final angle. Ti sits at half-integer, O at mixed integer/half-integer
    coordinates in units of a0 (as in analysis.cubic_srtio3)."""
    out = np.array(positions, float)
    theta = np.radians(angle_deg) * np.asarray(axes, float)
    for i, s in enumerate(symbols):
        if s != "O":
            continue
        f = out[i] / a0
        k = int(np.argmin(np.abs(f - np.round(f))))  # the axis along which O sits half a cell from Ti
        d = np.zeros(3)
        d[k] = -0.5 * a0  # O - Ti for the Ti at O + a0/2 along k
        ti = (out[i] - d) / a0 - 0.5
        sign = (-1) ** int(np.round(ti).sum())  # supercell vectors have even index sums, so this is well defined
        out[i] += sign * np.cross(theta, d)
    return out


def strain_cell(positions, cell, orientation: str, a_target: float, poisson=POISSON_GUESS):
    """Scale the in-plane vectors to pseudocubic in-plane lattice parameter `a_target`.

    Fractional coordinates are kept. a3 is scaled by (1 - poisson * eps) as a starting guess.
    Returns (positions, cell, eps) where eps is the applied in-plane strain relative to the input cell.
    """
    cell = np.asarray(cell, float)
    frac = np.asarray(positions, float) @ np.linalg.inv(cell)
    a_now = an.inplane_pc_lattice(cell, orientation)
    f = a_target / a_now
    new = cell.copy()
    new[:2] *= f
    new[2] *= 1.0 - poisson * (f - 1.0)
    return frac @ new, new, f - 1.0


def seed_directions(cell, orientation: str) -> dict:
    """Unit vectors for polar seeds, in the film frame.

    001: normal [001]pc, a1 = [110]pc, bisector of a1 and a2 = [010]pc.
    111: normal [111]pc, a1 = [1-10]pc, n x a1 = [11-2]pc type.
    """
    cell = np.asarray(cell, float)
    n = np.cross(cell[0], cell[1])
    n /= np.linalg.norm(n)
    e1 = cell[0] / np.linalg.norm(cell[0])
    if orientation == "001":
        e2 = e1 + cell[1] / np.linalg.norm(cell[1])
        names = ("polar_normal", "polar_110", "polar_100")
    elif orientation == "111":
        e2 = np.cross(n, e1)
        names = ("polar_normal", "polar_1-10", "polar_11-2")
    else:
        raise ValueError(orientation)
    e2 /= np.linalg.norm(e2)
    return dict(zip(names, (n, e1, e2)))


def polar_seed(positions, symbols, direction, amp=0.04):
    """Move every Ti by `amp` Angstrom along `direction` (a Slater-like start for a polar relaxation)."""
    out = np.array(positions, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    for i, s in enumerate(symbols):
        if s == "Ti":
            out[i] += amp * d
    return out


def rattle(positions, amp, seed=0):
    """Small reproducible random displacements to break symmetry."""
    rng = np.random.default_rng(seed)
    return np.asarray(positions, float) + rng.normal(scale=amp, size=np.shape(positions))
