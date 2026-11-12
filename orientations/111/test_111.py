"""Checks of the [111] set-up: the cell, the strain, the references' space groups, the seeds and the
tilt and polar measures.  ~/.sto-envs/mace/bin/python -m pytest -q orientations/111/test_111.py"""
import numpy as np
import pytest
from ase import Atoms

import sto111 as S

A0 = 3.9


def atoms(pos, sym, cell):
    return Atoms(symbols=sym, positions=pos, cell=cell, pbc=True)


def test_cell_geometry():
    pos, sym, cell = S.reference_cell("untilted", A0, A0)
    assert len(sym) == 30 and sym.count("Sr") == 6 and sym.count("Ti") == 6 and sym.count("O") == 18
    assert np.linalg.det(cell) > 0
    assert np.linalg.norm(cell[0]) == pytest.approx(np.sqrt(2) * A0) and np.linalg.norm(cell[1]) == pytest.approx(np.sqrt(2) * A0)
    cos = cell[0] @ cell[1] / np.linalg.norm(cell[0]) / np.linalg.norm(cell[1])
    assert np.degrees(np.arccos(cos)) == pytest.approx(120.0)  # hexagonal in-plane lattice
    assert np.allclose(cell[2], [0, 0, 2 * np.sqrt(3) * A0], atol=1e-9)
    ax = S.pc_axes(cell)
    assert np.allclose(ax @ ax.T, A0 ** 2 * np.eye(3))


@pytest.mark.parametrize("a_t", [3.70, 3.90, 4.10])
def test_strain_keeps_the_hexagonal_plane(a_t):
    pos, sym, cell = S.reference_cell("t111", A0, a_t)
    assert S.an.inplane_pc_lattice(cell, "111") == pytest.approx(a_t, abs=1e-9)
    assert np.linalg.norm(cell[0]) / np.sqrt(2) == pytest.approx(a_t) and np.linalg.norm(cell[1]) / np.sqrt(2) == pytest.approx(a_t)
    assert abs(cell[0][2]) < 1e-12 and abs(cell[1][2]) < 1e-12


@pytest.mark.parametrize("name, unstrained, strained", [
    ("untilted", "Pm-3m", "R-3m"), ("t111", "R-3c", "R-3c"), ("t11-1", "R-3c", "C2/c"), ("t001", "I4/mcm", "C2/c"),
    ("t110", "Imma", "C2/c"), ("t1-10", "Imma", "C2/m"), ("t11-2", "C2/c", "C2/c")])
def test_reference_space_groups(name, unstrained, strained):
    assert S.symmetry(atoms(*S.reference_cell(name, A0, A0)), 1e-3)[0] == unstrained
    for f in (0.98, 1.02):
        sg, _, inv = S.symmetry(atoms(*S.reference_cell(name, A0, A0 * f)), 1e-3)
        assert sg == strained and inv


@pytest.mark.parametrize("name, film", [
    ("t111", (0, 0, 1)), ("t11-1", (0, (8 / 9) ** 0.5, 1 / 3)), ("t001", (0, (2 / 3) ** 0.5, 3 ** -0.5)),
    ("t110", (0, 3 ** -0.5, (2 / 3) ** 0.5)), ("t1-10", (1, 0, 0)), ("t11-2", (0, 1, 0))])
def test_tilt_axis_in_the_film_frame(name, film):
    pos, sym, cell = S.reference_cell(name, A0, A0)
    t = S.tilt_pattern(pos, sym, cell)
    total = 4.0 * np.linalg.norm(S.REFERENCES[name])
    assert [t[f"tilt_about_{k}"] for k in S.FILM_AXES] == pytest.approx([total * c for c in film], abs=0.06)


@pytest.mark.parametrize("name, glazer", [("untilted", "a0a0a0"), ("t111", "a-a-a-"), ("t001", "a0a0c-"),
                                          ("t110", "a-a-c0"), ("t1-10", "a-a-c0")])
def test_glazer(name, glazer):
    assert S.tilt_pattern(*S.reference_cell(name, A0, A0))["glazer"] == glazer


def test_seed_directions_and_polar_components():
    n = np.array([0, 0, 1.0])
    d0 = S.seed_directions(S.reference_cell("untilted", A0, A0)[2])
    angle = {k: np.degrees(np.arccos(abs(v @ n))) for k, v in d0.items()}
    assert angle == pytest.approx({"P111": 0, "P1-10": 90, "P11-2": 90, "P110": 35.264, "P001": 54.736,
                                   "P11-1": 70.529}, abs=1e-3)
    pos, sym, cell = S.reference_cell("untilted", A0, 3.8)  # strained: the cube axes lean ~1 deg
    d = S.seed_directions(cell)
    for name, key in (("P111", "d_111"), ("P1-10", "d_1-10"), ("P11-2", "d_11-2")):
        c = S.polar_components(S.cells.polar_seed(pos, sym, d[name], amp=0.04), sym, cell)
        assert c[key] == pytest.approx(0.04, abs=1e-6)
    c = S.polar_components(S.cells.polar_seed(pos, sym, d["P001"], amp=0.04), sym, cell)
    assert c["d_pc_z"] == pytest.approx(0.04, abs=1e-6)


def test_symmetry_order_does_not_depend_on_the_cell():
    """The ground-state rule compares runs in 30-atom (scan) and 60-atom (DFT) cells, so the count must not
    change with the cell: R-3c has point group -3m, order 12, in its own cell and in a doubled cell whose
    lattice lacks the threefold axis."""
    a = atoms(*S.reference_cell("t111", A0, A0))
    assert S.symmetry(a, 1e-3)[1] == S.symmetry(a.repeat((2, 1, 1)), 1e-3)[1] == 12
