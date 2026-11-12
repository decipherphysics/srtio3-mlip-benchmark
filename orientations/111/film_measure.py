"""Measures of a [111] SrTiO3 film that work on any cell (the model's 30-atom cells and the 60-atom DFT
cells alike), so both are described the same way.

The pseudocubic axes are found from the structure itself (Ti-Ti neighbour vectors), not from a known
supercell matrix. The film has a threefold axis, so in-plane directions are given by their size and by
an angle folded into 0-30 degrees: 0 = along a [1-10]-type axis, 30 = along a [11-2]-type axis.
"""
from __future__ import annotations

import numpy as np

import sto111 as S

an = S.an


def cube_axes(positions, symbols, cell):
    """Pseudocubic lattice vectors (rows, right-handed, each pointing out of the film) from the mean
    Ti-Ti nearest-neighbour vectors, and the film normal."""
    P = np.asarray(positions, float)
    cell = np.asarray(cell, float)
    n = np.cross(cell[0], cell[1])
    n /= np.linalg.norm(n)
    ti = np.array([i for i, s in enumerate(symbols) if s == "Ti"], int)
    vecs = []
    for i in ti:
        v, _ = an.neighbour_vectors(P, cell, i, ti, 7)
        vecs.extend(x for x in v if np.linalg.norm(x) > 1.0)  # drop the atom itself
    vecs = np.array([x if x @ n > 0 else -x for x in vecs])
    groups = []
    for v in vecs:
        for g in groups:
            if v @ g[0] / np.linalg.norm(v) / np.linalg.norm(g[0]) > 0.9:
                g.append(v)
                break
        else:
            groups.append([v])
    if len(groups) != 3:
        raise RuntimeError(f"found {len(groups)} Ti-Ti directions, expected 3")
    ax = np.array([np.mean(g, axis=0) for g in groups])
    if np.linalg.det(ax) < 0:
        ax = ax[[1, 0, 2]]
    return ax, n


def _split(v, ax, n):
    """Normal part, in-plane size, and in-plane angle (0-30 deg; 0 = [1-10]-type, 30 = [11-2]-type)."""
    vn = float(v @ n)
    ip = v - vn * n
    size = float(np.linalg.norm(ip))
    if size < 1e-12:
        return vn, 0.0, 0.0
    u = ax / np.linalg.norm(ax, axis=1)[:, None]
    x = u[0] - u[1]
    x -= (x @ n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    phi = np.degrees(np.arctan2(ip @ y, ip @ x)) % 60.0
    return vn, size, float(min(phi, 60.0 - phi))


def describe(atoms) -> dict:
    pos, sym, cell = atoms.get_positions(), atoms.get_chemical_symbols(), np.array(atoms.cell)
    n_fu = len(sym) // 5
    ax, n = cube_axes(pos, sym, cell)
    u = ax / np.linalg.norm(ax, axis=1)[:, None]
    ti = [i for i, s in enumerate(sym) if s == "Ti"]
    o = np.array([i for i, s in enumerate(sym) if s == "O"], int)
    th = []
    for i in ti:  # rotation of each O6 cage about the three cube axes, from its O-O axes
        vecs, _ = an.neighbour_vectors(pos, cell, i, o, 6)
        proj = vecs @ u.T
        v = np.array([vecs[np.argmax(proj[:, k])] - vecs[np.argmin(proj[:, k])] for k in range(3)])
        x, y, z = u
        th.append(np.degrees([0.5 * (np.arctan2(v[1] @ z, v[1] @ y) - np.arctan2(v[2] @ y, v[2] @ z)),
                              0.5 * (np.arctan2(v[2] @ x, v[2] @ z) - np.arctan2(v[0] @ z, v[0] @ x)),
                              0.5 * (np.arctan2(v[0] @ y, v[0] @ x) - np.arctan2(v[1] @ x, v[1] @ y))]))
    th = np.array(th)
    mag = np.abs(th).mean(axis=0)
    phase = []
    for k in range(3):
        if mag[k] < 0.3:
            phase.append("0")
            continue
        s = []
        for a, i in enumerate(ti):
            d = an.min_image(pos[ti] - (pos[i] + ax[k]), cell)
            s.append(np.sign(th[a, k] * th[int(np.argmin(np.linalg.norm(d, axis=1))), k]))
        phase.append("+" if np.mean(s) > 0 else "-")
    idx = np.rint(an.min_image(pos[ti] - pos[ti[0]], cell) @ np.linalg.inv(ax)).astype(int)
    w = (((-1.0) ** idx.sum(axis=1))[:, None] * th).mean(axis=0) @ u  # out-of-phase rotation vector
    d = an.ti_displacements(pos, sym, cell).mean(axis=0)
    dn, dip, daz = _split(d, ax, n)
    wn, wip, waz = _split(w, ax, n)
    a_in = an.inplane_pc_lattice(cell, "111")
    vol = abs(np.linalg.det(cell)) / n_fu
    return {"n_fu": n_fu, "ti_disp": float(np.linalg.norm(d)), "ti_normal": abs(dn), "ti_inplane": dip,
            "ti_inplane_angle": daz, "tilt_total": float(np.linalg.norm(w)), "tilt_normal": abs(wn),
            "tilt_inplane": wip, "tilt_inplane_angle": waz,
            "rot": [float(m) for m in mag], "phases": "".join(phase), "inphase": "+" in phase,
            "a_inplane": a_in, "vol_per_fu": float(vol), "c_over_a": float(vol / a_in ** 3),
            "cube_axis_stretch_pct": [float(100 * (np.linalg.norm(v) / 1.0 - 1)) for v in ax]}
