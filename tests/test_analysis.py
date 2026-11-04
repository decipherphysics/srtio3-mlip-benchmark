import numpy as np
import pytest

from sto import analysis as an
from sto import config


def test_cubic_is_centred_and_untilted():
    pos, sym, cell = an.cubic_srtio3()
    p = an.polar_summary(pos, sym, cell)
    assert p["ti_disp_polar"] < 1e-10
    t = an.tilt_summary(pos, sym, cell)
    assert abs(t["tilt_all"]) < 1e-8


def test_polar_displacement_recovered():
    pos, sym, cell = an.cubic_srtio3()
    pos = pos.copy()
    pos[1, 2] += 0.05  # Ti up by 0.05 A
    p = an.polar_summary(pos, sym, cell)
    assert p["ti_disp_z"] == pytest.approx(0.05, abs=1e-10)
    assert p["ti_disp_polar"] == pytest.approx(0.05, abs=1e-10)


def _rotated_20_atom(phi_deg=3.0):
    """sqrt2 x sqrt2 x 2 cell with an a0a0c- rotation of the O octahedra by phi about z."""
    pos, sym, cell = an.cubic_srtio3()
    pos, sym, cell = an.supercell(pos, sym, cell, [[1, 1, 0], [-1, 1, 0], [0, 0, 2]])
    pos = pos.copy()
    ti = [i for i, s in enumerate(sym) if s == "Ti"]
    o = [i for i, s in enumerate(sym) if s == "O"]
    phi = np.radians(phi_deg)
    a = 3.905
    for j in o:
        # equatorial O (same z as a Ti layer): rotate about the nearest Ti; sign alternates checkerboard and along z
        vecs = an.min_image(pos[ti] - pos[j], cell)
        d = np.linalg.norm(vecs, axis=1)
        if abs(vecs[np.argmin(d)][2]) > 0.1:  # apical O, leave alone
            continue
        k = ti[np.argmin(d)]
        r = an.min_image(pos[j] - pos[k], cell)
        # sign from the Ti position: checkerboard in-plane and alternating along z
        f = np.round(pos[k] / (a / 2)).astype(int)
        sign = (-1) ** ((f[0] + f[1]) // 2 + (f[2] - 1) // 2)
        c, s = np.cos(sign * phi), np.sin(sign * phi)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pos[j] = pos[k] + rot @ r
    return pos, sym, cell


def test_supercell_counts():
    pos, sym, cell = an.cubic_srtio3()
    p2, s2, c2 = an.supercell(pos, sym, cell, [[1, 1, 0], [-1, 1, 0], [0, 0, 2]])
    assert len(p2) == 20 and s2.count("Ti") == 4 and s2.count("O") == 12


def test_centrosymmetrize_removes_polar_keeps_cubic():
    pos, sym, cell = an.cubic_srtio3()
    pos, sym, cell = an.supercell(pos, sym, cell, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    pos = pos.copy()
    for i, s in enumerate(sym):
        if s == "Ti":
            pos[i, 2] += 0.06
        if s == "O":
            pos[i, 2] -= 0.02
    assert an.polar_summary(pos, sym, cell)["ti_disp_polar"] > 0.05
    new, rms = an.centrosymmetrize(pos, sym, cell)
    assert an.polar_summary(new, sym, cell)["ti_disp_polar"] < 1e-6
    assert rms > 0


def test_tetragonality_001_and_111():
    pos, sym, cell = an.cubic_srtio3(a=3.9)
    cell_t = cell.copy()
    cell_t[2, 2] *= 1.02
    t = an.tetragonality(cell_t, 1, "001")
    assert t["c_over_a"] == pytest.approx(1.02, rel=1e-9)
    # [111]-oriented hexagonal setting: a1=[1,-1,0], a2=[0,1,-1], a3=[1,1,1] (3 formula units)
    a = 3.9
    hexcell = a * np.array([[1, -1, 0], [0, 1, -1], [1, 1, 1]], float)
    t = an.tetragonality(hexcell, 3, "111")
    assert t["a_inplane"] == pytest.approx(a, rel=1e-9)
    assert t["c_over_a"] == pytest.approx(1.0, rel=1e-9)


def test_offsets_single_formula():
    comps = [{"Sr": 4, "Ti": 4, "O": 12}, {"Sr": 8, "Ti": 8, "O": 24}]
    e_ref = np.array([-160.0, -320.5])
    e_model = e_ref + np.array([4 * 1.5, 8 * 1.5]) + np.array([0.0, 0.01])
    mu, resid = an.fit_offsets(e_model, e_ref, comps)
    per_fu = mu["Sr"] + mu["Ti"] + 3 * mu["O"]
    assert per_fu == pytest.approx(1.5, abs=2e-3)
    assert np.abs(resid).max() < 0.01


def test_critical_strain():
    strains = [-2, -1, -0.5, 0, 0.5, 1, 2]
    gains = [5, 1, 0, 0, 0, 0.8, 4]
    disps = [0.05, 0.02, 0, 0, 0, 0.01, 0.04]
    c = an.critical_strain(strains, gains, disps)
    assert c == {"tensile": 1.0, "compressive": 1.0}


def test_critical_strain_ignores_stray_point_and_handles_zero():
    s = [0, 0.5, 1.0, 1.5, 2.0]
    g = [0, 1, 0, 2, 3]  # stray polar point at 0.5, nonpolar at 1.0
    d = [0, 0.02, 0, 0.02, 0.03]
    assert an.critical_strain(s, g, d)["tensile"] == 1.5
    allp = an.critical_strain([-1, 0, 1], [1, 1, 1], [0.02, 0.02, 0.02])
    assert allp == {"tensile": 0.0, "compressive": 0.0}
    out = an.critical_strain([-1, 0, 1, 2], [1, 0, 1, 0], [0.02, 0, 0.02, 0])
    assert np.isnan(out["tensile"]) and out["compressive"] == 1.0


def test_critical_strain_counts_the_grid_point_near_zero_as_unstrained():
    """The grid point at a = 3.90 A is +0.008 % from the DFT a0: the unstrained point, not a tensile one."""
    s = config.STRAINS_PCT
    d = [0.1] * len(s)
    assert an.critical_strain(s, [5.0] * len(s), d) == {"tensile": 0.0, "compressive": 0.0}
    gains = [46.7, 21.4, 7.7, 1.7, 1.2, 11.3, 38.9, 94.0, 154.0]  # the DFT gains, 3.70 ... 4.10 A
    c = an.critical_strain(s, gains, d, e_tol=2.0)
    assert c["tensile"] == pytest.approx(s[5]) and c["compressive"] == pytest.approx(-s[2])


def test_polar_shift_is_not_a_tilt():
    """Pure Ti off-centring bends Ti-O-Ti but must not register as an octahedral tilt."""
    pos, sym, cell = an.supercell(*an.cubic_srtio3(), [[1, 1, 0], [-1, 1, 0], [0, 0, 2]])
    pos = pos.copy()
    for i, s in enumerate(sym):
        if s == "Ti":
            pos[i] += [0.0, 0.0, 0.10]
    t = an.tilt_summary(pos, sym, cell)
    assert abs(t["tilt_all"]) < 1e-6
    # and a tilt plus a polar shift still reads as the tilt alone
    pos, sym, cell = _rotated_20_atom(3.0)
    pos = pos.copy()
    for i, s in enumerate(sym):
        if s == "Ti":
            pos[i] += [0.05, 0.05, 0.0]
    assert an.tilt_summary(pos, sym, cell)["tilt_in"] == pytest.approx(3.0, abs=0.05)


def _polar_shift(pos, sym, cell, ti_shift, o_shift_by_bond):
    """Uniform polar distortion: every Ti moved by ti_shift, every O by the shift for its bond axis (x, y, z)."""
    pos = np.asarray(pos, float)
    ti = [i for i, s in enumerate(sym) if s == "Ti"]
    out = pos.copy()
    for i, s in enumerate(sym):
        if s == "Ti":
            out[i] += ti_shift
        elif s == "O":
            v = an.min_image(pos[ti] - pos[i], cell)
            out[i] += o_shift_by_bond[int(np.argmax(np.abs(v[np.argmin(np.linalg.norm(v, axis=1))])))]
    return out


def test_polar_o_shift_is_not_a_tilt():
    """Polar distortions move the O on x, y and z bonds by different amounts. That must not read as a
    tilt, alone (P4mm-type: cage centroids would read 0.8 deg on in-plane bonds) or on top of a tilt."""
    untilted = an.supercell(*an.cubic_srtio3(), [[1, 1, 0], [-1, 1, 0], [0, 0, 2]])
    p4mm = ([0, 0, 0.10], [[0, 0, -0.02], [0, 0, -0.02], [0, 0, 0.06]])
    along_110 = ([0.07, 0.07, 0], [[-0.03, -0.01, 0], [-0.01, -0.03, 0], [0.04, 0.04, 0]])
    for (pos, sym, cell), shift in ((untilted, p4mm), (_rotated_20_atom(3.0), p4mm), (_rotated_20_atom(3.0), along_110)):
        before = an.tilt_summary(pos, sym, cell, normal=(0, 0, 1))
        after = an.tilt_summary(_polar_shift(pos, sym, cell, *shift), sym, cell, normal=(0, 0, 1))
        for k in ("tilt_in", "tilt_out"):
            assert after[k] == pytest.approx(before[k], abs=1e-6)


def test_tilt_angle_recovered():
    pos, sym, cell = _rotated_20_atom(3.0)
    t = an.tilt_summary(pos, sym, cell, normal=(0, 0, 1))
    assert t["tilt_in"] == pytest.approx(3.0, abs=0.05)
    assert t["tilt_out"] == pytest.approx(0.0, abs=1e-6)  # apical O are not moved, so their axes stay along z


def test_centrosymmetrize_keeps_tilts():
    pos, sym, cell = _rotated_20_atom(4.0)
    pos = pos.copy()
    for i, s in enumerate(sym):
        if s == "Ti":
            pos[i, 0] += 0.04
            pos[i, 1] += 0.04
    new, _ = an.centrosymmetrize(pos, sym, cell)
    assert an.polar_summary(new, sym, cell)["ti_disp_polar"] < 1e-6
    assert an.tilt_summary(new, sym, cell)["tilt_in"] == pytest.approx(4.0, abs=0.1)
