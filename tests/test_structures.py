import numpy as np
import pytest
from ase.build import bulk, fcc111
from ase.calculators.emt import EMT

from sto import structures as st


def _film(kind):
    """A compressed fcc Cu film (EMT is cheap and gives stresses): a square in-plane cell like the
    [001] runs or a hexagonal one like the [111] runs, with a3 sheared so the cell has to relax it."""
    if kind == "square":
        atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    else:
        atoms = fcc111("Cu", size=(1, 1, 3), a=3.61, periodic=True)
    cell = np.array(atoms.cell)
    cell[:2] *= 0.98
    cell[2] += [0.15, -0.10, 0.0]
    atoms.set_cell(cell, scale_atoms=True)
    return atoms


@pytest.mark.parametrize("kind", ["square", "hexagonal"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_film_relax_holds_a1_a2_and_relaxes_a3(kind, symmetric):
    atoms = _film(kind)
    if symmetric:  # FixSymmetry needs a symmetric start: no shear
        cell = np.array(atoms.cell)
        cell[2, :2] = 0.0
        atoms.set_cell(cell, scale_atoms=True)
    start = np.array(atoms.cell)
    r = st.relax(atoms, EMT(), film=True, symmetric=symmetric, fmax=1e-4, steps=2000)
    end = np.array(r["atoms"].cell)
    assert r["converged"]
    assert np.abs(end[:2] - start[:2]).max() < 1e-10  # a1 and a2 exactly as given
    assert r["plane_tilt_deg"] == pytest.approx(0.0, abs=1e-8)
    xx, yy, zz, yz, xz, xy = r["stress_voigt"]
    assert max(abs(zz), abs(yz), abs(xz)) < 1e-4  # out-of-plane stresses relaxed (eV/A^3)
    assert min(abs(xx), abs(yy)) > 1e-3  # the in-plane stress the substrate holds
    assert np.abs(end[2, :2]).max() < 1e-3  # fcc under biaxial strain: the shear relaxes away
    assert end[2, 2] > start[2, 2]  # and c lengthens under in-plane compression


def test_film_relax_holds_a1_a2_with_monoclinic_symmetry():
    """FixSymmetry with an in-plane 2-fold axis, as C2/c has under [001] strain: a3 sheared along x
    only, so the 2-fold axis is y. a1 and a2 must still be held while a3 relaxes."""
    atoms = _film("square")
    cell = np.array(atoms.cell)
    cell[2, 1] = 0.0
    atoms.set_cell(cell, scale_atoms=True)
    start = np.array(atoms.cell)
    r = st.relax(atoms, EMT(), film=True, symmetric=True, fmax=1e-4, steps=2000)
    end = np.array(r["atoms"].cell)
    assert r["converged"]
    assert np.abs(end[:2] - start[:2]).max() < 1e-10
    xx, yy, zz, yz, xz, xy = r["stress_voigt"]
    assert max(abs(zz), abs(yz), abs(xz)) < 1e-4
    assert np.abs(end[2, :2]).max() < 1e-3  # the shear relaxes away, as without symmetry


def test_film_relax_needs_a1_a2_in_plane():
    atoms = _film("square")
    cell = np.array(atoms.cell)
    cell[0, 2] = 0.1
    atoms.set_cell(cell)
    with pytest.raises(ValueError):
        st.relax(atoms, EMT(), film=True)
