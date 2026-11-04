"""ASE glue: Atoms objects, constrained relaxations, and each model's relaxed cubic lattice parameter."""
from __future__ import annotations

import time

import numpy as np
from ase import Atoms
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.optimize import BFGS, FIRE

from . import analysis as an
from . import cells, config


class EpitaxialCellFilter(UnitCellFilter):
    """Film relaxation as in the DFT runs (VASP with the in-plane stress components held): a1 and a2
    stay exactly as given and a3 relaxes in all three components, so at the end the out-of-plane
    stresses (zz, xz, yz) are zero. a1 and a2 must lie in the xy-plane (cells.film_cell does this).

    ASE's own cell masks act on a symmetric strain, so a shear of a3 also tilts and stretches a1 and
    a2: in the first [001] scan (28 October 2026) some films grew their in-plane cell by up to
    0.28 A that way. Here the cell changes only through the third column f of the deformation
    gradient F = [x, y, f], so F a1 = a1 and F a2 = a2 exactly."""

    def __init__(self, atoms):
        if np.abs(np.array(atoms.cell)[:2, 2]).max() > 1e-8:
            raise ValueError("a1 and a2 must lie in the xy-plane")
        super().__init__(atoms)

    def set_positions(self, new, **kwargs):
        new = np.array(new, float)
        f = np.eye(3)
        f[:, 2] = new[-3:, 2] / self.cell_factor  # the last three rows hold cell_factor * F
        new[-3:] = self.cell_factor * f
        super().set_positions(new, **kwargs)

    def get_forces(self, **kwargs):
        forces = super().get_forces(**kwargs)
        forces[-3:, :2] = 0.0  # only the third column of F moves
        return forces


def _voigt_to_3x3(v):
    xx, yy, zz, yz, xz, xy = v
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _3x3_to_voigt(m):
    return [m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1]]


def to_film_frame(atoms):
    """Rigidly rotate atoms (in place) so a1 || x and a2 lies in the xy-plane. Returns the rotation q
    (new vectors = old vectors @ q) and the angle in degrees by which the film normal had tilted."""
    cell = np.array(atoms.cell)
    n = np.cross(cell[0], cell[1])
    tilt = float(np.degrees(np.arccos(np.clip(abs(n[2]) / np.linalg.norm(n), -1, 1))))
    q = cells.film_rotation(cell)
    atoms.set_cell(cell @ q, scale_atoms=False)
    atoms.positions = atoms.get_positions() @ q
    return q, tilt


def to_atoms(positions, symbols, cell) -> Atoms:
    return Atoms(symbols=list(symbols), positions=np.asarray(positions, float),
                 cell=np.asarray(cell, float), pbc=True)


def relax(atoms, calc, film=False, hydrostatic=False, symmetric=False, symprec=1e-4,
          fmax=config.FMAX, steps=config.MAX_STEPS, logfile=None) -> dict:
    """Relax positions and the cell. film=True holds a1 and a2 and relaxes a3 (EpitaxialCellFilter);
    otherwise the whole cell relaxes, or only its volume with hydrostatic=True. symmetric=True keeps
    the starting space group (ASE FixSymmetry), which is how the centrosymmetric reference is held
    nonpolar."""
    a = atoms.copy()
    a.calc = calc
    if symmetric:
        a.set_constraint(FixSymmetry(a, symprec=symprec))
    filt = EpitaxialCellFilter(a) if film else FrechetCellFilter(a, hydrostatic_strain=hydrostatic)
    t0 = time.time()
    nsteps, converged = 0, False
    for opt_cls, kw in ((BFGS, {"maxstep": 0.05}), (FIRE, {})):  # FIRE as a fallback
        opt = opt_cls(filt, logfile=logfile, **kw)
        converged = bool(opt.run(fmax=fmax, steps=steps))
        nsteps += int(opt.nsteps)
        if converged:
            break
    energy = float(a.get_potential_energy())
    forces = a.get_forces()
    stress = np.asarray(a.get_stress())
    a.set_constraint()
    a.calc = None
    plane_tilt = 0.0
    if film:
        moved = np.abs(np.array(a.cell)[:2] - np.array(atoms.cell)[:2]).max()
        if moved > 1e-8:
            raise RuntimeError(f"in-plane cell moved by {moved:.2e} A")
        # a1 along x, a2 in the xy-plane (a rotation about z for a DFT seed; otherwise nothing moves)
        q, plane_tilt = to_film_frame(a)
        stress = np.array(_3x3_to_voigt(q.T @ _voigt_to_3x3(stress) @ q))
    return {"atoms": a, "energy": energy, "converged": converged, "nsteps": nsteps,
            "seconds": time.time() - t0, "fmax_final": float(np.linalg.norm(forces, axis=1).max()),
            "stress_voigt": stress.tolist(), "plane_tilt_deg": plane_tilt}


def relaxed_cubic_a0(calc, a_start=3.9) -> dict:
    """Lattice parameter of cubic (Pm-3m) SrTiO3 for this model: 5-atom cell, symmetry kept."""
    pos, sym, cell = an.cubic_srtio3(a_start)
    r = relax(to_atoms(pos, sym, cell), calc, hydrostatic=True, symmetric=True, symprec=1e-3)
    r["a0"] = float(np.cbrt(r["atoms"].get_volume()))
    return r


def describe(atoms, orientation: str, n_fu: int) -> dict:
    """Polar displacement, tilts and lattice metrics of a relaxed film cell."""
    pos, sym, cell = atoms.get_positions(), atoms.get_chemical_symbols(), np.array(atoms.cell)
    normal = np.cross(cell[0], cell[1])
    out = {}
    out.update(an.polar_summary(pos, sym, cell))
    out.update(an.tilt_summary(pos, sym, cell, normal=normal))
    out.update(an.tetragonality(cell, n_fu, orientation))
    return out
