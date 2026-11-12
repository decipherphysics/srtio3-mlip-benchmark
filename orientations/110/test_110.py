"""Checks of the [110] set-up: the cell, the strain, the references' space groups, the seeds and the
tilt and polar measures.  ~/.sto-envs/mace/bin/python -m pytest -q orientations/110/test_110.py"""
import numpy as np
import pytest
from ase import Atoms

import sto110 as S

A0 = 3.9


def atoms(pos, sym, cell):
    return Atoms(symbols=sym, positions=pos, cell=cell, pbc=True)


def test_cell_geometry():
    pos, sym, cell = S.reference_cell("untilted", A0, A0)
    assert len(sym) == 20 and sym.count("Sr") == 4 and sym.count("Ti") == 4 and sym.count("O") == 12
    assert np.linalg.det(cell) > 0
    assert np.allclose(cell[0], [2 * A0, 0, 0], atol=1e-9)
    assert np.allclose(cell[1], [0, np.sqrt(2) * A0, 0], atol=1e-9)
    assert np.allclose(cell[2], [0, 0, np.sqrt(2) * A0], atol=1e-9)
    ax = S.pc_axes(cell)
    assert np.allclose(np.linalg.norm(ax, axis=1), A0) and np.allclose(ax @ ax.T, A0 ** 2 * np.eye(3))


@pytest.mark.parametrize("a_t", [3.70, 3.90, 4.10])
def test_strain_is_equal_along_both_inplane_axes(a_t):
    pos, sym, cell = S.reference_cell("t110", A0, a_t)
    assert S.an.inplane_pc_lattice(cell, "110") == pytest.approx(a_t, abs=1e-9)
    assert np.linalg.norm(cell[0]) / 2 == pytest.approx(a_t, abs=1e-9)
    assert np.linalg.norm(cell[1]) / np.sqrt(2) == pytest.approx(a_t, abs=1e-9)
    assert abs(cell[0][2]) < 1e-12 and abs(cell[1][2]) < 1e-12  # in the xy-plane, as the film filter needs


def test_other_orientations_unchanged():
    cell = np.diag([np.sqrt(2) * A0, np.sqrt(2) * A0, 2 * A0])
    assert S.an.inplane_pc_lattice(cell, "001") == pytest.approx(A0)


@pytest.mark.parametrize("name, group", [("untilted", "Pm-3m"), ("t001", "I4/mcm"), ("t100", "I4/mcm"),
                                         ("t1-10", "Imma"), ("t110", "Imma"), ("t101", "Imma"),
                                         ("t1-11", "R-3c"), ("t111", "R-3c")])
def test_unstrained_references_have_their_bulk_space_groups(name, group):
    a = atoms(*S.reference_cell(name, A0, A0))
    sg, nops, inv = S.symmetry(a, 1e-3)
    assert sg == group and inv


@pytest.mark.parametrize("name, group", [("untilted", "Cmmm"), ("t001", "Ibam"), ("t100", "C2/m"),
                                         ("t1-10", "Imma"), ("t110", "Imma"), ("t101", "P-1"),
                                         ("t1-11", "C2/c"), ("t111", "C2/c")])
@pytest.mark.parametrize("factor", [0.98, 1.02])
def test_strained_references_stay_centrosymmetric(name, group, factor):
    """(110) strain keeps the film's mirrors normal to [001], [1-10] and [110]: a tilt axis along one of
    them keeps its full group (Imma), any other lowers it."""
    sg, _, inv = S.symmetry(atoms(*S.reference_cell(name, A0, A0 * factor)), 1e-3)
    assert inv and sg == group


@pytest.mark.parametrize("name, glazer", [("untilted", "a0a0a0"), ("t001", "a0a0c-"), ("t110", "a-a-c0"),
                                          ("t1-10", "a-a-c0"), ("t111", "a-a-a-"), ("t1-11", "a-a-a-")])
def test_tilt_pattern(name, glazer):
    pos, sym, cell = S.reference_cell(name, A0, A0)
    t = S.tilt_pattern(pos, sym, cell)
    assert t["glazer"] == glazer
    for k, axis in zip("xyz", S.REFERENCES[name] or (0, 0, 0)):
        assert t[f"rot_{k}"] == pytest.approx(4.0 * abs(axis), abs=0.05)


@pytest.mark.parametrize("name, film", [("t001", (1, 0, 0)), ("t100", (0, 0.5 ** 0.5, 0.5 ** 0.5)),
                                        ("t1-10", (0, 1, 0)), ("t110", (0, 0, 1)), ("t101", (0.5 ** 0.5, 0.5, 0.5)),
                                        ("t1-11", (3 ** -0.5, (2 / 3) ** 0.5, 0)),
                                        ("t111", (3 ** -0.5, 0, (2 / 3) ** 0.5))])
def test_tilt_axis_in_the_film_frame(name, film):
    """Unit tilt axis along x = [001], y = [1-10], z = [110], times the total tilt."""
    pos, sym, cell = S.reference_cell(name, A0, A0)
    t = S.tilt_pattern(pos, sym, cell)
    total = 4.0 * np.linalg.norm(S.REFERENCES[name])
    got = [t["tilt_about_001"], t["tilt_about_1-10"], t["tilt_about_110"]]
    assert got == pytest.approx([total * c for c in film], abs=0.05)


def test_tilt_pattern_ignores_polar_shifts():
    pos, sym, cell = S.reference_cell("t001", A0, A0)
    pos = S.cells.polar_seed(pos, sym, [0.3, 0.5, 0.8], amp=0.2)
    assert S.tilt_pattern(pos, sym, cell)["glazer"] == "a0a0c-"


def test_seed_directions_and_polar_components():
    pos, sym, cell = S.reference_cell("untilted", A0, 3.8)
    d = S.seed_directions(cell)
    n = np.array([0, 0, 1.0])
    assert np.allclose(d["P001"], [1, 0, 0]) and np.allclose(d["P1-10"], [0, 1, 0]) and np.allclose(d["P110"], n)
    assert d["P1-11"] @ n == pytest.approx(0)
    assert np.degrees(np.arccos(d["P111"] @ n)) == pytest.approx(35.264, abs=1e-3)
    assert np.degrees(np.arccos(d["P100"] @ n)) == pytest.approx(45.0)
    expect = {"P001": ("d_001", "d_pc_z"), "P1-10": ("d_1-10", None), "P110": ("d_110", None),
              "P100": (None, "d_pc_x")}
    for name, (film_key, pc_key) in expect.items():
        p = S.cells.polar_seed(pos, sym, d[name], amp=0.04)
        c = S.polar_components(p, sym, cell)
        if film_key:
            assert c[film_key] == pytest.approx(0.04, abs=1e-6)
        if pc_key:  # the strained pseudocubic axes are no longer exactly the cubic directions
            assert c[pc_key] == pytest.approx(0.04, abs=2e-4)
