"""[110] epitaxial strain for SrTiO3: the film cell, the centrosymmetric references, the polar seeds and
the measures used to name each phase.

Built on this repository's `sto` package, imported read-only. The [110] pieces are added in memory
here, so nothing in `sto` changes.

Film frame: x = [001]pc and y = [1-10]pc lie in the plane, z = [110]pc is the film normal.
Strain: the same pseudocubic in-plane lattice parameter along [001] and [1-10], as on a cubic
substrate. a1 and a2 are held; a3 relaxes in all three components (EpitaxialCellFilter), as in the
[001] study and in the DFT runs with the stress patch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sto import analysis as an  # noqa: E402
from sto import cells, config, io, models  # noqa: E402,F401
from sto import structures as st  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODEL = "mace-mh-1-r2scan"  # the closest of the eleven to the PBEsol DFT on [001]
ORIENT = "110"

# Rows are the new lattice vectors in cubic units: 2[001] x [1-10] x [110], right-handed, 4 f.u.
# (20 atoms). Every vector has an even index sum, so every out-of-phase (R-point) tilt fits, as in
# the [001] cell, and so does an in-phase rotation about [001].
MATRIX = [[0, 0, 2], [1, -1, 0], [1, 1, 0]]
N_FU = 4
cells.SUPERCELL_MATRIX.setdefault(ORIENT, MATRIX)
cells.N_FU.setdefault(ORIENT, N_FU)

_inplane_orig = an.inplane_pc_lattice


def inplane_pc_lattice(cell, orientation: str) -> float:
    """As sto.analysis.inplane_pc_lattice, plus [110]: one (110) layer (a/sqrt2 thick) holds one
    formula unit per sqrt2 a^2 of area."""
    if orientation != ORIENT:
        return _inplane_orig(cell, orientation)
    cell = np.asarray(cell, float)
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    m = max(1, round(area / (np.sqrt(2.0) * an.A_GUESS ** 2)))
    return float(np.sqrt(area / (np.sqrt(2.0) * m)))


# sto.analysis.tetragonality and sto.cells.strain_cell look the function up on the module
an.inplane_pc_lattice = inplane_pc_lattice

# Centrosymmetric references: out-of-phase tilt axes in pseudocubic x, y, z, named by the tilt axis.
# Up to the film's symmetry (mirrors normal to [001], [1-10] and [110]) these are all the distinct
# patterns with one, two or three equal tilts.
REFERENCES = {
    "untilted": None,     # a0a0a0
    "t001": (0, 0, 1),    # a0a0c-, axis [001], in the plane
    "t100": (1, 0, 0),    # a0a0c-, axis [100], 45 deg out of the plane
    "t1-10": (1, -1, 0),  # a-a-c0, axis [1-10], in the plane
    "t110": (1, 1, 0),    # a-a-c0, axis [110], the film normal
    "t101": (1, 0, 1),    # a-a-c0, axis [101], 60 deg from the normal
    "t1-11": (1, -1, 1),  # a-a-a-, axis [1-11], in the plane
    "t111": (1, 1, 1),    # a-a-a-, axis [111], 35 deg from the normal
}


def reference_cell(name: str, a0: float, a_target: float):
    """(positions, symbols, cell) of a reference, built at the model's a0 and strained to a_target."""
    axes = REFERENCES[name]
    pos, sym, cell = cells.film_cell(ORIENT, a0, tilt=axes, tilt_deg=config.NONPOLAR_TILT_DEG if axes else 0.0)
    pos, cell, _ = cells.strain_cell(pos, cell, ORIENT, a_target)
    return pos, sym, cell


def seed_directions(cell) -> dict:
    """Unit vectors for the polar seeds, in the film frame: the three film axes, the in-plane body
    diagonal [1-11], the body diagonal [111] (35 deg out of the plane) and the cube axis [100]
    (45 deg out of the plane)."""
    cell = np.asarray(cell, float)
    n = np.cross(cell[0], cell[1])
    n /= np.linalg.norm(n)                    # [110]
    e1 = cell[0] / np.linalg.norm(cell[0])    # [001]
    e2 = np.cross(n, e1)                      # [1-10]
    d = {"P001": e1, "P1-10": e2, "P110": n,
         "P1-11": e1 + np.sqrt(2.0) * e2,     # [1-11] = [001] + [1-10]
         "P111": e1 + np.sqrt(2.0) * n,       # [111] = [001] + [110]
         "P100": e2 + n}                      # [100] = ([110] + [1-10]) / 2
    return {k: v / np.linalg.norm(v) for k, v in d.items()}


# ------------------------------------------------------------------ measures
def pc_axes(cell) -> np.ndarray:
    """Pseudocubic [100], [010], [001] (rows) of a film cell, from its lattice vectors."""
    c = np.asarray(cell, float)
    return np.array([(c[1] + c[2]) / 2.0, (c[2] - c[1]) / 2.0, c[0] / 2.0])


def octahedron_rotations(positions, symbols, cell):
    """Rotation of each TiO6 cage about pseudocubic x, y and z (degrees), from its O-O axes
    (O at +k minus O at -k), which polar shifts of Ti or O do not turn. Returns (angles (nTi, 3),
    Ti indices, pseudocubic axes)."""
    P = np.asarray(positions, float)
    cell = np.asarray(cell, float)
    ax = pc_axes(cell)
    u = ax / np.linalg.norm(ax, axis=1)[:, None]
    ti = [i for i, s in enumerate(symbols) if s == "Ti"]
    o = np.array([i for i, s in enumerate(symbols) if s == "O"], int)
    rows = []
    for i in ti:
        vecs, _ = an.neighbour_vectors(P, cell, i, o, 6)  # Ti -> its six O
        proj = vecs @ u.T
        v = np.array([vecs[np.argmax(proj[:, k])] - vecs[np.argmin(proj[:, k])] for k in range(3)])
        x, y, z = u
        th_x = 0.5 * (np.arctan2(v[1] @ z, v[1] @ y) - np.arctan2(v[2] @ y, v[2] @ z))
        th_y = 0.5 * (np.arctan2(v[2] @ x, v[2] @ z) - np.arctan2(v[0] @ z, v[0] @ x))
        th_z = 0.5 * (np.arctan2(v[0] @ y, v[0] @ x) - np.arctan2(v[1] @ x, v[1] @ y))
        rows.append(np.degrees([th_x, th_y, th_z]))
    return np.array(rows), ti, ax


def tilt_pattern(positions, symbols, cell, zero=0.3, same=0.15) -> dict:
    """Mean rotation about each pseudocubic axis, whether neighbours along that axis turn the same way
    (+) or the opposite way (-), and the Glazer symbol built from them."""
    P = np.asarray(positions, float)
    th, ti, ax = octahedron_rotations(P, symbols, cell)
    mag = np.abs(th).mean(axis=0)
    phase = []
    for k in range(3):
        if mag[k] < zero:
            phase.append("0")
            continue
        s = []
        for a, i in enumerate(ti):
            d = an.min_image(P[ti] - (P[i] + ax[k]), cell)
            b = int(np.argmin(np.linalg.norm(d, axis=1)))  # the Ti one step along +k
            s.append(np.sign(th[a, k] * th[b, k]))
        phase.append("+" if np.mean(s) > 0 else "-")
    letters = []
    for k in range(3):  # Glazer: axis k takes the letter of an earlier axis with the same angle
        letter = "abc"[k]
        for j in range(k):
            if abs(mag[k] - mag[j]) < same:
                letter = letters[j][0]
                break
        letters.append(letter + phase[k])
    # The out-of-phase (R-point) part as one rotation vector, then along the film axes. This tells
    # apart patterns the Glazer symbol does not, e.g. a-a-c0 about [110] (the normal) and about [1-10].
    n = np.rint((P[ti] @ np.linalg.inv(ax)) - 0.5).astype(int)  # Ti sit at half-integer cube positions
    w = (((-1.0) ** n.sum(axis=1))[:, None] * th).mean(axis=0)
    u = ax / np.linalg.norm(ax, axis=1)[:, None]
    w_film = w @ u  # film frame: x = [001], y = [1-10], z = [110]
    return {"rot_x": float(mag[0]), "rot_y": float(mag[1]), "rot_z": float(mag[2]),
            "phase_x": phase[0], "phase_y": phase[1], "phase_z": phase[2], "glazer": "".join(letters),
            "tilt_about_001": float(abs(w_film[0])), "tilt_about_1-10": float(abs(w_film[1])),
            "tilt_about_110": float(abs(w_film[2]))}


def polar_components(positions, symbols, cell) -> dict:
    """Uniform Ti off-centring (Ti minus O6 centroid, mean over Ti) along the film axes and the
    pseudocubic axes, in Angstrom."""
    d = an.ti_displacements(np.asarray(positions, float), symbols, cell).mean(axis=0)
    ax = pc_axes(cell)
    u = ax / np.linalg.norm(ax, axis=1)[:, None]
    return {"d_001": float(d[0]), "d_1-10": float(d[1]), "d_110": float(d[2]),
            "d_pc_x": float(d @ u[0]), "d_pc_y": float(d @ u[1]), "d_pc_z": float(d @ u[2])}


def describe(atoms) -> dict:
    """The study's measures (sto.structures.describe) plus the [110] ones above."""
    pos, sym, cell = atoms.get_positions(), atoms.get_chemical_symbols(), np.array(atoms.cell)
    out = st.describe(atoms, ORIENT, N_FU)
    out.update(tilt_pattern(pos, sym, cell))
    out.update(polar_components(pos, sym, cell))
    out["a_001"] = float(np.linalg.norm(cell[0]) / 2.0)
    out["a_1-10"] = float(np.linalg.norm(cell[1]) / np.sqrt(2.0))
    return out


def symmetry(atoms, symprec=5e-3):
    """(Hermann-Mauguin symbol, number of operations, centrosymmetric?) from spglib."""
    import spglib
    d = spglib.get_symmetry_dataset((np.array(atoms.cell), atoms.get_scaled_positions(), atoms.numbers),
                                    symprec=symprec)
    if d is None:
        return "?", 0, False
    inv = any(np.array_equal(r, -np.eye(3, dtype=int)) for r in d.rotations)
    return d.international, len(d.rotations), bool(inv)
