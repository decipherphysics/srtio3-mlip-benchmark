"""[111] epitaxial strain for SrTiO3: the film cell, the centrosymmetric references, the polar seeds and
the measures used to name each phase. The [111] twin of orientations/110/sto110.py.

Built on this repository's `sto` package, imported read-only; its [111] film cell (hexagonal,
a = sqrt2 a_pc, c = 2 sqrt3 a_pc, 30 atoms) and in-plane lattice measure are used as they are.

Film frame: x = [1-10]pc and y = [11-2]pc lie in the plane, z = [111]pc is the film normal.
a1 and a2 are held; a3 relaxes in all three components (EpitaxialCellFilter), as in the DFT runs.
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
ORIENT = "111"
MATRIX = np.array(cells.SUPERCELL_MATRIX[ORIENT], float)  # [1-10], [01-1], 2[111]
N_FU = cells.N_FU[ORIENT]  # 6
FILM_AXES = ("1-10", "11-2", "111")  # x, y (in the plane), z (the normal)

# Centrosymmetric references: out-of-phase tilt axes in pseudocubic x, y, z, named by the tilt axis.
# Up to the film's symmetry (-3m about [111]) these are the distinct axes: the normal, the other body
# diagonals, a cube axis, the face diagonals out of and in the plane, and the in-plane [11-2].
REFERENCES = {
    "untilted": None,           # a0a0a0
    "t111": (1, 1, 1),          # a-a-a-, axis [111], the normal (R-3c)
    "t11-1": (1, 1, -1),        # a-a-a-, axis [11-1], 71 deg from the normal
    "t001": (0, 0, 1),          # a0a0c-, axis [001], 55 deg from the normal
    "t110": (1, 1, 0),          # a-a-c0, axis [110], 35 deg from the normal
    "t1-10": (1, -1, 0),        # a-a-c0, axis [1-10], in the plane
    "t11-2": (0.5, 0.5, -1),    # a-a-c- with c = 2a, axis [11-2], in the plane
}


def reference_cell(name: str, a0: float, a_target: float):
    """(positions, symbols, cell) of a reference, built at the model's a0 and strained to a_target."""
    axes = REFERENCES[name]
    pos, sym, cell = cells.film_cell(ORIENT, a0, tilt=axes, tilt_deg=config.NONPOLAR_TILT_DEG if axes else 0.0)
    pos, cell, _ = cells.strain_cell(pos, cell, ORIENT, a_target)
    return pos, sym, cell


def film_axes(cell) -> np.ndarray:
    """Unit vectors (rows) of the film frame: [1-10] along a1, [11-2] = n x a1, n = [111]."""
    cell = np.asarray(cell, float)
    n = np.cross(cell[0], cell[1])
    n /= np.linalg.norm(n)
    e1 = cell[0] / np.linalg.norm(cell[0])
    return np.array([e1, np.cross(n, e1), n])


def pc_axes(cell) -> np.ndarray:
    """Pseudocubic [100], [010], [001] (rows) of a film cell, from its lattice vectors."""
    return np.linalg.solve(MATRIX, np.asarray(cell, float))


def seed_directions(cell) -> dict:
    """Unit vectors for the polar seeds, in the film frame: the normal [111], the in-plane [1-10] and
    [11-2], and the oblique [110] (35 deg from the normal), [001] (55 deg) and [11-1] (71 deg)."""
    f = film_axes(cell)
    u = pc_axes(cell)
    u /= np.linalg.norm(u, axis=1)[:, None]
    d = {"P111": f[2], "P1-10": f[0], "P11-2": f[1],
         "P110": u[0] + u[1], "P001": u[2], "P11-1": u[0] + u[1] - u[2]}
    return {k: v / np.linalg.norm(v) for k, v in d.items()}


# ------------------------------------------------------------------ measures
def octahedron_rotations(positions, symbols, cell):
    """Rotation of each TiO6 cage about pseudocubic x, y and z (degrees), from its O-O axes
    (O at +k minus O at -k), which polar shifts of Ti or O do not turn."""
    P = np.asarray(positions, float)
    cell = np.asarray(cell, float)
    ax = pc_axes(cell)
    u = ax / np.linalg.norm(ax, axis=1)[:, None]
    ti = [i for i, s in enumerate(symbols) if s == "Ti"]
    o = np.array([i for i, s in enumerate(symbols) if s == "O"], int)
    rows = []
    for i in ti:
        vecs, _ = an.neighbour_vectors(P, cell, i, o, 6)
        proj = vecs @ u.T
        v = np.array([vecs[np.argmax(proj[:, k])] - vecs[np.argmin(proj[:, k])] for k in range(3)])
        x, y, z = u
        th_x = 0.5 * (np.arctan2(v[1] @ z, v[1] @ y) - np.arctan2(v[2] @ y, v[2] @ z))
        th_y = 0.5 * (np.arctan2(v[2] @ x, v[2] @ z) - np.arctan2(v[0] @ z, v[0] @ x))
        th_z = 0.5 * (np.arctan2(v[0] @ y, v[0] @ x) - np.arctan2(v[1] @ x, v[1] @ y))
        rows.append(np.degrees([th_x, th_y, th_z]))
    return np.array(rows), ti, ax


def tilt_pattern(positions, symbols, cell, zero=0.3, same=0.15) -> dict:
    """Mean rotation about each pseudocubic axis, its phase between neighbours along that axis, the
    Glazer symbol, and the out-of-phase rotation vector along the film axes."""
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
            b = int(np.argmin(np.linalg.norm(d, axis=1)))
            s.append(np.sign(th[a, k] * th[b, k]))
        phase.append("+" if np.mean(s) > 0 else "-")
    letters = []
    for k in range(3):
        letter = "abc"[k]
        for j in range(k):
            if abs(mag[k] - mag[j]) < same:
                letter = letters[j][0]
                break
        letters.append(letter + phase[k])
    n = np.rint((P[ti] @ np.linalg.inv(ax)) - 0.5).astype(int)  # Ti sit at half-integer cube positions
    w = (((-1.0) ** n.sum(axis=1))[:, None] * th).mean(axis=0)
    u = ax / np.linalg.norm(ax, axis=1)[:, None]
    w_film = film_axes(cell) @ (w @ u)
    out = {"rot_x": float(mag[0]), "rot_y": float(mag[1]), "rot_z": float(mag[2]),
           "phase_x": phase[0], "phase_y": phase[1], "phase_z": phase[2], "glazer": "".join(letters)}
    out.update({f"tilt_about_{k}": float(abs(c)) for k, c in zip(FILM_AXES, w_film)})
    return out


def polar_components(positions, symbols, cell) -> dict:
    """Uniform Ti off-centring (Ti minus O6 centroid, mean over Ti) along the film axes and the
    pseudocubic axes, in Angstrom."""
    d = an.ti_displacements(np.asarray(positions, float), symbols, cell).mean(axis=0)
    f = film_axes(cell)
    u = pc_axes(cell)
    u /= np.linalg.norm(u, axis=1)[:, None]
    out = {f"d_{k}": float(d @ e) for k, e in zip(FILM_AXES, f)}
    out.update({"d_pc_x": float(d @ u[0]), "d_pc_y": float(d @ u[1]), "d_pc_z": float(d @ u[2])})
    return out


def describe(atoms) -> dict:
    """The study's measures (sto.structures.describe) plus the [111] ones above."""
    pos, sym, cell = atoms.get_positions(), atoms.get_chemical_symbols(), np.array(atoms.cell)
    out = st.describe(atoms, ORIENT, N_FU)
    out.update(tilt_pattern(pos, sym, cell))
    out.update(polar_components(pos, sym, cell))
    return out


def symmetry(atoms, symprec=5e-3):
    """(Hermann-Mauguin symbol, point-group order, centrosymmetric?) from spglib. The order is the
    crystal's, from the space-group database, so it does not depend on the cell: spglib lists each rotation
    once per lattice point of the cell given, and leaves out rotations the cell's own lattice lacks (the
    60-atom DFT cells are rectangular, so an R3c state there loses its threefold axis)."""
    import spglib
    d = spglib.get_symmetry_dataset((np.array(atoms.cell), atoms.get_scaled_positions(), atoms.numbers),
                                    symprec=symprec)
    if d is None:
        return "?", 0, False
    inv = any(np.array_equal(r, -np.eye(3, dtype=int)) for r in d.rotations)
    order = len({r.tobytes() for r in spglib.get_symmetry_from_database(d.hall_number)["rotations"]})
    return d.international, order, bool(inv)
