import tempfile
from pathlib import Path

from sto import vasp_audit as va

OUTCAR = """ POTCAR:    PAW_PBE Sr_sv 07Sep2000
   TITEL  = PAW_PBE Sr_sv 07Sep2000
   TITEL  = PAW_PBE Ti_pv 07Sep2000
   TITEL  = PAW_PBE O 08Apr2002
   number of dos      NEDOS =    301   number of ions     NIONS =     20
  free  energy   TOTEN  =      -158.12345678 eV
  energy  without entropy=     -158.12000000  energy(sigma->0) =     -158.12100000
  free  energy   TOTEN  =      -158.20000000 eV
  energy  without entropy=     -158.19900000  energy(sigma->0) =     -158.19950000
 reached required accuracy - stopping structural energy minimisation
"""

INCAR = """SYSTEM = SrTiO3 strained
ENCUT = 600 ; ISPIN = 1
GGA = PS   # PBEsol
IBRION = 2
NSW = 60
ISIF = 2 ! fixed cell
"""

POSCAR = """SrTiO3
1.0
 5.5 0 0
 0 5.5 0
 0 0 7.8
Sr Ti O
4 4 12
Direct
"""


def _make_run(root: Path, name: str):
    d = root / name
    d.mkdir(parents=True)
    (d / "OUTCAR").write_text(OUTCAR)
    (d / "INCAR").write_text(INCAR)
    (d / "CONTCAR").write_text(POSCAR)
    (d / "KPOINTS").write_text("auto\n0\nGamma\n 6 6 4\n")
    return d


def test_audit_run_reads_everything():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_run(root, "SrTiO3/001/strain_+1.5")
        runs = va.find_runs(root)
        assert len(runs) == 1
        r = va.audit_run(runs[0])
        assert r["functional"] == "PBEsol"
        assert r["encut"] == "600" and r["ispin"] == "1"
        assert r["natoms"] == 20 and r["formula"] == "Sr4 Ti4 O12"
        assert r["final_free_energy"] == -158.2
        assert r["final_energy_sigma0"] == -158.1995
        assert r["n_ionic_steps"] == 2 and r["converged"] is True
        assert r["orientation"] == "001" and r["strain_percent"] == "+1.5"
        assert r["kpoints"] == "Gamma 6 6 4"
        assert r["potcars"].count("PAW_PBE") == 3


def test_functional_labels():
    assert va.functional_label({"GGA": "PE"}, "") == "PBE"
    assert va.functional_label({}, "PAW_PBE Ti") .startswith("PBE")
    assert va.functional_label({}, "PAW Ti 08Apr2002").startswith("LDA")
    assert va.functional_label({"METAGGA": "R2SCAN"}, "") == "r2SCAN"


def test_strain_guesses():
    assert va.guess_orientation_and_strain("runs/111/eps-0.5") == ("111", "-0.5")
    assert va.guess_orientation_and_strain("runs/001/strain_1p0") == ("001", "1.0")
    assert va.guess_orientation_and_strain("runs/bulk") == ("", "")
    assert va.guess_orientation_and_strain("vasp_STO_runs/Epistrain/001/3.90/Cc_1/DE") == ("001", "0.008")
    assert va.guess_orientation_and_strain("vasp_STO_runs/Epistrain/111/3.70/R3c") == ("111", "-5.121")
    assert va.guess_orientation_and_strain("vasp_STO_runs/Relaxation/Pm-3m") == ("bulk", "")


def test_run_type_and_include():
    assert va.run_type({"IBRION": "2", "NSW": "50"}) == "relax"
    assert va.run_type({"IBRION": "8"}) == "phonon"
    assert va.run_type({}) == "static"
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_run(Path(tmp), "x/001/strain_0.0")
        r = va.audit_run(d)
        assert r["run_type"] == "relax" and r["include"] == "y" and r["is_srtio3"]
        (d / "OUTCAR").write_text(OUTCAR.replace(" reached required accuracy", " not yet"))
        assert va.audit_run(d)["include"] == "n"


def test_poscar_symbols():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "POSCAR"
        p.write_text(POSCAR)
        s = va.parse_poscar_symbols(p)
    assert s[:4] == ["Sr"] * 4 and s[4:8] == ["Ti"] * 4 and len(s) == 20
