"""Structure analysis for strained SrTiO3. Pure numpy/scipy, so it can be tested without ASE.

All functions take Cartesian positions (N, 3) in Angstrom, a list of chemical symbols,
and a (3, 3) cell whose rows are the lattice vectors.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment

A_GUESS = 3.9  # approximate pseudocubic SrTiO3 lattice parameter, used only for rounding


# ---------------------------------------------------------------- geometry helpers
def min_image(vectors: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Wrap displacement vectors into the minimum-image convention."""
    cell = np.asarray(cell, float)
    frac = np.asarray(vectors, float) @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


def _indices(symbols, element):
    return np.array([i for i, s in enumerate(symbols) if s == element], dtype=int)


_SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)


def neighbour_vectors(positions, cell, centre, candidates, k):
    """Vectors from atom `centre` to its k nearest atoms among `candidates`, including periodic images.

    Images matter in small cells: in the 5-atom cubic cell each Ti's six O neighbours are
    images of only three distinct O atoms.
    """
    cell = np.asarray(cell, float)
    base = min_image(positions[candidates] - positions[centre], cell)
    shifts = _SHIFTS @ cell
    vecs = (base[:, None, :] + shifts[None, :, :]).reshape(-1, 3)
    owner = np.repeat(candidates, len(shifts))
    d = np.linalg.norm(vecs, axis=1)
    order = np.argsort(d)[:k]
    return vecs[order], owner[order]


# ---------------------------------------------------------------- polar displacement
def ti_displacements(positions, symbols, cell) -> np.ndarray:
    """Displacement of each Ti from the centroid of its six nearest O atoms (Angstrom), shape (nTi, 3)."""
    positions = np.asarray(positions, float)
    ti, o = _indices(symbols, "Ti"), _indices(symbols, "O")
    out = []
    for i in ti:
        vecs, _ = neighbour_vectors(positions, cell, i, o, 6)
        out.append(-vecs.mean(axis=0))  # Ti minus O6 centroid
    return np.array(out)


def polar_summary(positions, symbols, cell) -> dict:
    """Uniform (polar) part of the Ti off-centring: mean vector, its magnitude, and mean per-Ti magnitude."""
    d = ti_displacements(positions, symbols, cell)
    mean = d.mean(axis=0)
    return {
        "ti_disp_x": mean[0], "ti_disp_y": mean[1], "ti_disp_z": mean[2],
        "ti_disp_polar": float(np.linalg.norm(mean)),
        "ti_disp_mean_abs": float(np.linalg.norm(d, axis=1).mean()),
    }


# ---------------------------------------------------------------- octahedral tilts
def ti_o_ti_bonds(positions, symbols, cell):
    """For each O: the angle (degrees) at that O between the O-O axes through it of its two TiO6
    cages (from this O to the opposite O of each cage), and the unit vector along the bond.
    Octahedral rotations turn the axes and bend this angle. Polar distortions do not: Ti
    off-centring does not enter, and the two O of an axis shift together. (Cage centroids would
    read a tilt wherever apical and equatorial O shift by different amounts, as in P4mm.)"""
    positions = np.asarray(positions, float)
    ti, o = _indices(symbols, "Ti"), _indices(symbols, "O")
    cage = {int(i): neighbour_vectors(positions, cell, i, o, 6)[0] for i in ti}  # Ti -> its six O
    angles, dirs = [], []
    for j in o:
        vecs, owners = neighbour_vectors(positions, cell, j, ti, 2)  # O -> its two Ti
        a1, a2 = (v + cage[int(t)][np.argmin(cage[int(t)] @ -v)] for v, t in zip(vecs, owners))  # O -> opposite O
        angles.append(np.degrees(np.arctan2(np.linalg.norm(np.cross(a1, a2)), np.dot(a1, a2))))
        b = a2 - a1
        dirs.append(b / np.linalg.norm(b))
    return np.array(angles), np.array(dirs)


def tilt_summary(positions, symbols, cell, normal=(0.0, 0.0, 1.0)) -> dict:
    """Octahedral rotation angles from the angle at each O between its two cages' O-O axes:
    phi = (180 - angle) / 2.

    Bonds are classed by their angle to `normal` (the film normal):
    'out' (|cos| > 0.9), 'in' (|cos| < 0.2), 'oblique' otherwise. For [111] strain all three
    pseudocubic bonds are oblique, so only the overall value is meaningful.
    """
    angles, dirs = ti_o_ti_bonds(positions, symbols, cell)
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    c = np.abs(dirs @ n)
    phi = (180.0 - angles) / 2.0
    out = {"tilt_all": float(phi.mean())}
    for name, mask in (("tilt_out", c > 0.9), ("tilt_in", c < 0.2), ("tilt_oblique", (c >= 0.2) & (c <= 0.9))):
        out[name] = float(phi[mask].mean()) if mask.any() else np.nan
    return out


# ---------------------------------------------------------------- lattice
def inplane_pc_lattice(cell, orientation: str) -> float:
    """Pseudocubic in-plane lattice parameter from the area spanned by the first two cell vectors.

    [001]: area per formula unit in a layer = a^2.  [111]: area per formula unit in a layer = sqrt(3) a^2.
    The number of formula units per layer is found by rounding with A_GUESS.
    """
    cell = np.asarray(cell, float)
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    if orientation == "001":
        per = A_GUESS ** 2
        m = max(1, round(area / per))
        return float(np.sqrt(area / m))
    if orientation == "111":
        per = np.sqrt(3.0) * A_GUESS ** 2
        m = max(1, round(area / per))
        return float(np.sqrt(area / (np.sqrt(3.0) * m)))
    raise ValueError(f"orientation must be '001' or '111', got {orientation!r}")


def tetragonality(cell, n_fu: int, orientation: str) -> dict:
    """Volume per formula unit, in-plane pseudocubic a, and r = (V / f.u.) / a^3.

    For [001] r equals c/a. For [111] r is the same out-of-plane/in-plane ratio expressed
    through the volume, so the two orientations are directly comparable.
    """
    v = abs(np.linalg.det(np.asarray(cell, float))) / n_fu
    a = inplane_pc_lattice(cell, orientation)
    return {"vol_per_fu": float(v), "a_inplane": a, "c_over_a": float(v / a ** 3)}


# ---------------------------------------------------------------- nonpolar reference
def centrosymmetrize(positions, symbols, cell, centres_from=("Ti", "Sr")):
    """Return a centrosymmetric version of a structure, keeping inversion-symmetric distortions such as tilts.

    Tries inversion centres at every Ti and Sr site and at their midpoints with the
    inversion image. For each candidate centre c, every atom is paired with an atom of the same
    species in the inverted structure r' = 2c - r (Hungarian assignment on minimum-image distances),
    and the structure is replaced by the average of the atom and its partner's image.
    The centre with the smallest pairing cost wins. Returns (new_positions, rms_shift_in_Angstrom).
    """
    positions = np.asarray(positions, float)
    symbols = list(symbols)
    best = None
    centres = [positions[i] for el in centres_from for i in _indices(symbols, el)]
    for c in centres:
        new = positions.copy()
        cost_total = 0.0
        for el in sorted(set(symbols)):
            idx = _indices(symbols, el)
            inv = 2.0 * c - positions[idx]
            diff = min_image(positions[idx][:, None, :] - inv[None, :, :], cell)
            cost = np.linalg.norm(diff, axis=2)
            r, col = linear_sum_assignment(cost)
            cost_total += cost[r, col].sum()
            # partner image of atom idx[r] is inv[col]; average r and nearest image of inv[col]
            target = positions[idx][r] - diff[r, col]  # = image of inv[col] nearest to atom
            new[idx[r]] = 0.5 * (positions[idx][r] + target)
        if best is None or cost_total < best[0]:
            best = (cost_total, new)
    shift = min_image(best[1] - positions, cell)
    return best[1], float(np.sqrt((shift ** 2).sum(axis=1).mean()))


# ---------------------------------------------------------------- phonon modes
# Born effective charges of cubic SrTiO3 (Zhong, King-Smith & Vanderbilt, PRL 72, 3618 (1994)).
# O is anisotropic: parallel / perpendicular to its Ti-O-Ti bond. They sum to zero per formula unit.
BORN = {"Sr": 2.54, "Ti": 7.12}
BORN_O_PAR, BORN_O_PERP = -5.66, -2.00
NOMINAL_CHARGE = {"Sr": 2.0, "Ti": 4.0, "O": -2.0}  # fallback when no positions are given


def _uniform_fraction(u):
    """1 when all rows of u are equal (uniform motion), 0 when they cancel (staggered motion)."""
    tot = (np.linalg.norm(u, axis=1) ** 2).sum()
    if tot < 1e-30:
        return np.nan
    return float(np.linalg.norm(u.sum(axis=0)) ** 2 / (len(u) * tot))


def o_bond_directions(positions, symbols, cell) -> dict:
    """Unit vector along the Ti-O-Ti bond through each O (atom index -> vector)."""
    positions = np.asarray(positions, float)
    ti, o = _indices(symbols, "Ti"), _indices(symbols, "O")
    out = {}
    for j in o:
        vecs, _ = neighbour_vectors(positions, cell, j, ti, 2)
        b = vecs[1] - vecs[0]
        out[int(j)] = b / np.linalg.norm(b)
    return out


def rotation_fraction(u, positions, symbols, cell) -> float:
    """Share of the O motion that is rigid rotation of the O6 cages (1 = pure octahedral rotation).

    For each Ti, the best rigid rotation w of its six O neighbours is fitted (least squares) and
    sum |w x r|^2 is compared with sum |u|^2. Uniform O shifts, Gamma-point O deformations
    (Axe, silent modes) and breathing all give ~0, because opposite O of one cage then move
    together or radially.
    """
    positions = np.asarray(positions, float)
    ti, o = _indices(symbols, "Ti"), _indices(symbols, "O")
    num = den = 0.0
    for i in ti:
        r, owners = neighbour_vectors(positions, cell, i, o, 6)
        uk = u[owners]
        inertia = sum(np.dot(v, v) * np.eye(3) - np.outer(v, v) for v in r)
        w = np.linalg.solve(inertia, sum(np.cross(v, x) for v, x in zip(r, uk)))
        num += sum(np.dot(c, c) for c in np.cross(w, r))
        den += float((uk ** 2).sum())
    return float(num / den) if den > 1e-30 else np.nan


def mode_character(eigvec, masses, symbols, positions=None, cell=None) -> dict:
    """Classify a Gamma-point phonon of the supercell from its dynamical-matrix eigenvector (N, 3).

    polar    : |sum_i Z*_i u_i| / sum_i |Z*_i u_i| in [0, 1], with SrTiO3 Born charges (anisotropic O)
               when positions are given, nominal charges otherwise. 0 for acoustic and tilt modes,
               large for Slater, Last and Axe polar modes.
    rotation : share of the O motion that is rigid O6-cage rotation (needs positions).
    o_weight : share of the displacement norm^2 carried by O.
    *_uniform: per-species uniform fraction (1 = all atoms of that species move together).
    label    : 'acoustic', 'polar', 'tilt' or 'other'.
    """
    e = np.asarray(eigvec).reshape(-1, 3)
    e = np.real(e) if np.abs(np.imag(e)).max() < 1e-8 else np.abs(e) * np.sign(np.real(e))
    u = e / np.sqrt(np.asarray(masses, float))[:, None]
    if positions is not None:
        bonds = o_bond_directions(positions, symbols, cell)
        zu = np.zeros_like(u)
        for i, s in enumerate(symbols):
            if s == "O":
                b = bonds[i]
                par = np.dot(u[i], b) * b
                zu[i] = BORN_O_PAR * par + BORN_O_PERP * (u[i] - par)
            else:
                zu[i] = BORN.get(s, 0.0) * u[i]
        rot = rotation_fraction(u, positions, symbols, cell)
    else:
        z = np.array([NOMINAL_CHARGE.get(s, 0.0) for s in symbols])
        zu = z[:, None] * u
        rot = np.nan
    polar = float(np.linalg.norm(zu.sum(axis=0)) / max(np.linalg.norm(zu, axis=1).sum(), 1e-30))
    norms = np.linalg.norm(u, axis=1)
    w = norms ** 2 / max((norms ** 2).sum(), 1e-30)
    out = {"polar": polar, "rotation": rot}
    for el in ("Sr", "Ti", "O"):
        idx = _indices(symbols, el)
        out[f"{el.lower()}_weight"] = float(w[idx].sum())
        out[f"{el.lower()}_uniform"] = _uniform_fraction(u[idx])
    if _uniform_fraction(u) > 0.95 and polar < 0.05:
        label = "acoustic"
    elif polar > 0.3:
        label = "polar"
    elif out["o_weight"] > 0.7 and (rot > 0.7 if not np.isnan(rot) else out["o_uniform"] < 0.2):
        label = "tilt"
    else:
        label = "other"
    out["label"] = label
    return out


# ---------------------------------------------------------------- energies
def fit_offsets(e_model, e_ref, compositions, elements=("Sr", "Ti", "O")):
    """Least-squares per-element energy offsets mu so that e_model - e_ref ~ sum_el n_el * mu_el.

    For pure SrTiO3 the composition matrix is rank 1, so this reduces to a single offset per
    formula unit (lstsq returns the minimum-norm solution). Returns (mu dict, corrected residuals in eV).
    """
    X = np.array([[comp.get(el, 0) for el in elements] for comp in compositions], float)
    y = np.asarray(e_model, float) - np.asarray(e_ref, float)
    mu, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ mu
    return dict(zip(elements, mu)), resid


def polar_energy_gain(e_nonpolar, e_polar, n_fu) -> float:
    """Energy gained by going polar, in meV per formula unit (positive = polar state lower)."""
    return 1000.0 * (e_nonpolar - e_polar) / n_fu


def critical_strain(strains, gains_mev, disps, e_tol=0.1, d_tol=0.005, zero_tol=0.05) -> dict:
    """Onset of the polar phase on the tensile and compressive sides.

    A point is polar when the polar energy gain exceeds e_tol (meV/f.u.) and the uniform Ti
    displacement exceeds d_tol (A). A point with |strain| < zero_tol (%) counts as unstrained: the
    grid point at a = 3.90 A is +0.008 % from the DFT a0. On each side the critical strain is the
    smallest |strain| from which every point further out is polar, so a single stray polar point
    cannot set it. It is 0 when the unstrained point and that whole side are polar, and NaN when
    the outermost point on that side is nonpolar. Values are grid points: the true onset lies
    between this point and the next smaller |strain|.
    """
    s = np.asarray(strains, float)
    polar = (np.asarray(gains_mev, float) > e_tol) & (np.asarray(disps, float) > d_tol)
    at_zero = np.abs(s) < zero_tol
    zero_polar = bool(at_zero.any() and polar[at_zero].all())
    out = {}
    for side, mask in (("tensile", s > 0), ("compressive", s < 0)):
        idx = np.where(mask & ~at_zero)[0]
        if not len(idx):
            out[side] = np.nan
            continue
        order = idx[np.argsort(np.abs(s[idx]))]
        p = polar[order]
        if not p[-1]:
            out[side] = np.nan
            continue
        nonpolar = np.where(~p)[0]
        if not len(nonpolar):
            out[side] = 0.0 if zero_polar else float(abs(s[order[0]]))
        else:
            out[side] = float(abs(s[order[nonpolar[-1] + 1]]))
    return out


def cubic_srtio3(a=3.905):
    """Five-atom cubic SrTiO3: (positions, symbols, cell)."""
    frac = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], float)
    cell = np.eye(3) * a
    return frac @ cell, ["Sr", "Ti", "O", "O", "O"], cell


def supercell(positions, symbols, cell, matrix):
    """Integer supercell (rows of `matrix` give new lattice vectors in units of the old ones)."""
    matrix = np.asarray(matrix, int)
    new_cell = matrix @ cell
    inv_new = np.linalg.inv(new_cell)
    n = int(round(abs(np.linalg.det(matrix))))
    rng = range(-max(abs(matrix).sum(axis=0)) - 1, max(abs(matrix).sum(axis=0)) + 2)
    pos, sym = [], []
    seen = set()
    for t in itertools.product(rng, rng, rng):
        shift = np.array(t) @ cell
        for p, s in zip(positions, symbols):
            f = (p + shift) @ inv_new
            if np.all(f > -1e-8) and np.all(f < 1 - 1e-8):
                key = (s, *np.round(f, 6))
                if key not in seen:
                    seen.add(key)
                    pos.append(p + shift)
                    sym.append(s)
    if len(pos) != n * len(positions):
        raise RuntimeError(f"supercell construction found {len(pos)} atoms, expected {n * len(positions)}")
    return np.array(pos), sym, new_cell
