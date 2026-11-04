import tempfile
from pathlib import Path

import numpy as np
import pytest

from sto import analysis as an
from sto import cells
from sto import vasp_audit as va

MASS = {"Sr": 87.62, "Ti": 47.867, "O": 15.999}


@pytest.mark.parametrize("orientation,natoms", [("001", 20), ("111", 30)])
def test_film_cell_frame_and_lattice(orientation, natoms):
    pos, sym, cell = cells.film_cell(orientation, 3.9)
    assert len(pos) == natoms
    assert abs(cell[0, 1]) < 1e-12 and abs(cell[0, 2]) < 1e-12 and abs(cell[1, 2]) < 1e-12
    assert np.linalg.det(cell) > 0
    assert an.inplane_pc_lattice(cell, orientation) == pytest.approx(3.9, rel=1e-12)
    # a3 is along the film normal for both supercells
    assert np.linalg.norm(cell[2, :2]) < 1e-10
    # rotation is rigid: still cubic SrTiO3, untilted and centred
    assert an.polar_summary(pos, sym, cell)["ti_disp_polar"] < 1e-10
    assert an.tilt_summary(pos, sym, cell)["tilt_all"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("orientation", ["001", "111"])
@pytest.mark.parametrize("phase", ["I4mcm", "Imma", "C2c"])
def test_tilted_nonpolar_starts(orientation, phase):
    pos, sym, cell = cells.film_cell(orientation, 3.9, tilt=phase, tilt_deg=4.0)
    ref, _, _ = cells.film_cell(orientation, 3.9)
    ti = [i for i, s in enumerate(sym) if s != "O"]
    assert np.allclose(pos[ti], ref[ti])  # only O moves
    assert an.polar_summary(pos, sym, cell)["ti_disp_polar"] < 1e-10
    assert an.centrosymmetrize(pos, sym, cell)[1] < 1e-10  # antiphase tilts keep inversion
    assert an.tilt_summary(pos, sym, cell)["tilt_all"] > 2.0
    if orientation == "001" and phase == "I4mcm":  # a0a0c-: rotation about the film normal only
        t = an.tilt_summary(pos, sym, cell, normal=np.cross(cell[0], cell[1]))
        assert t["tilt_in"] == pytest.approx(4.0, abs=0.1) and t["tilt_out"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("phase,group", [("Pm-3m", "P4/mmm"), ("I4mcm", "I4/mcm"), ("Imma", "Imma"), ("C2c", "C2/c")])
def test_nonpolar_starts_have_their_space_group(phase, group):
    """At the 3.90 A grid point (+0.008 %), each [001] reference has the space group it is named after
    (Pm-3m becomes P4/mmm under strain), at the tolerance FixSymmetry uses to hold it."""
    spglib = pytest.importorskip("spglib")
    pos, sym, cell = cells.film_cell("001", 3.9, tilt=phase, tilt_deg=4.0)
    pos, cell, _ = cells.strain_cell(pos, cell, "001", 3.9 * 1.00008)
    numbers = [{"Sr": 38, "Ti": 22, "O": 8}[s] for s in sym]
    frac = pos @ np.linalg.inv(cell)
    assert spglib.get_spacegroup((cell, frac, numbers), symprec=1e-4).split()[0] == group


@pytest.mark.parametrize("orientation", ["001", "111"])
def test_strain_cell(orientation):
    pos, sym, cell = cells.film_cell(orientation, 3.9)
    p2, c2, eps = cells.strain_cell(pos, cell, orientation, 3.9 * 1.02)
    assert eps == pytest.approx(0.02)
    assert an.inplane_pc_lattice(c2, orientation) == pytest.approx(3.9 * 1.02, rel=1e-12)
    assert c2[2, 2] < cell[2, 2]  # Poisson guess shortens a3 under tension
    frac1 = pos @ np.linalg.inv(cell)
    frac2 = p2 @ np.linalg.inv(c2)
    assert np.allclose(frac1, frac2)


def test_seed_directions_are_orthonormal_where_expected():
    _, _, cell = cells.film_cell("001", 3.9)
    d = cells.seed_directions(cell, "001")
    assert np.allclose(d["polar_normal"], [0, 0, 1])
    assert abs(np.dot(d["polar_110"], d["polar_100"])) == pytest.approx(np.cos(np.pi / 4))
    _, _, cell = cells.film_cell("111", 3.9)
    d = cells.seed_directions(cell, "111")
    assert abs(np.dot(d["polar_1-10"], d["polar_11-2"])) < 1e-12
    assert abs(np.dot(d["polar_normal"], d["polar_11-2"])) < 1e-12


def test_polar_seed_and_rattle():
    pos, sym, cell = cells.film_cell("111", 3.9)
    n = cells.seed_directions(cell, "111")["polar_normal"]
    p = cells.polar_seed(pos, sym, n, amp=0.04)
    s = an.polar_summary(p, sym, cell)
    assert s["ti_disp_polar"] == pytest.approx(0.04, abs=1e-9)
    r1, r2 = cells.rattle(pos, 0.01, seed=1), cells.rattle(pos, 0.01, seed=1)
    assert np.allclose(r1, r2) and not np.allclose(r1, pos)


def _eig_from_u(u, sym):
    m = np.array([MASS[s] for s in sym])
    e = u * np.sqrt(m)[:, None]
    return e / np.linalg.norm(e), m


def test_mode_character_labels():
    pos, sym, cell = an.cubic_srtio3()
    kw = {"positions": pos, "cell": cell}
    # acoustic (Born charges sum to zero)
    e, m = _eig_from_u(np.tile([0, 0, 1.0], (5, 1)), sym)
    assert an.mode_character(e, m, sym, **kw)["label"] == "acoustic"
    # Slater-like polar mode: Ti up, O down
    u = np.zeros((5, 3)); u[1, 2] = 1.0; u[2:, 2] = -0.5
    e, m = _eig_from_u(u, sym)
    c = an.mode_character(e, m, sym, **kw)
    assert c["label"] == "polar" and c["polar"] > 0.9
    # Axe mode: apical O (index 2, bond along z) against the two equatorial O, sum of O motion zero.
    # Nominal charges call this nonpolar; Born charges correctly call it polar.
    u = np.zeros((5, 3)); u[2, 2] = -2.0; u[3:, 2] = 1.0
    e, m = _eig_from_u(u, sym)
    assert an.mode_character(e, m, sym, **kw)["label"] == "polar"
    # Gamma-point silent-like O deformation (perpendicular to bonds, no dipole): not a tilt
    u = np.zeros((5, 3)); u[3, 2] = 1.0; u[4, 2] = -1.0
    e, m = _eig_from_u(u, sym)
    c = an.mode_character(e, m, sym, **kw)
    assert c["label"] == "other" and c["rotation"] < 0.05
    # a0a0c- tilt pattern in the 20-atom cell: only O moves, as rigid cage rotations
    from tests.test_analysis import _rotated_20_atom
    p0, s20, c20 = an.supercell(*an.cubic_srtio3(), [[1, 1, 0], [-1, 1, 0], [0, 0, 2]])
    p1, _, _ = _rotated_20_atom(3.0)
    u = an.min_image(p1 - p0, c20)
    e, m = _eig_from_u(u, s20)
    c = an.mode_character(e, m, s20, positions=p0, cell=c20)
    assert c["label"] == "tilt" and c["polar"] < 0.05 and c["rotation"] > 0.95


DFPT = """ Eigenvectors and eigenvalues of the dynamical matrix
 ----------------------------------------------------


   1 f  =   16.000000 THz   100.530965 2PiTHz  533.702560 cm-1    66.170000 meV
             X         Y         Z           dx          dy          dz
      0.000000  0.000000  0.000000     0.000000    0.000000    0.100000
      1.950000  1.950000  1.950000     0.000000    0.000000    0.200000

   2 f/i=    2.000000 THz    12.566371 2PiTHz   66.712820 cm-1     8.270000 meV
             X         Y         Z           dx          dy          dz
      0.000000  0.000000  0.000000     0.300000    0.000000    0.000000
      1.950000  1.950000  1.950000    -0.400000    0.000000    0.000000

 Eigenvectors after division by SQRT(mass)
   1 f  =   99.000000 THz   100.530965 2PiTHz  999.000000 cm-1    66.170000 meV
"""


NWRITE3_TAIL = """
 Eigenvectors and eigenvalues of the dynamical matrix
 ----------------------------------------------------
   1 f  =   77.000000 THz   100.530965 2PiTHz  777.000000 cm-1    66.170000 meV
             X         Y         Z           dx          dy          dz
      0.000000  0.000000  0.000000     9.000000    9.000000    9.000000
"""


def test_parse_outcar_phonons():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "OUTCAR"
        p.write_text(DFPT)
        f, v = va.parse_outcar_phonons(p)
    assert f == [533.70256, -66.71282]
    assert np.allclose(v[0], [[0, 0, 0.1], [0, 0, 0.2]])
    assert np.allclose(v[1], [[0.3, 0, 0], [-0.4, 0, 0]])
    # NWRITE = 3 prints a second, mass-divided block under the same heading: it must be ignored
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "OUTCAR"
        p.write_text(DFPT + NWRITE3_TAIL)
        f, v = va.parse_outcar_phonons(p)
    assert f == [533.70256, -66.71282]
