# data

These files are made from the raw VASP runs by `scripts/00_audit.py` and `scripts/01_convert.py`. The raw runs
themselves are not included.

| File | Contents |
|---|---|
| `dft_frames.extxyz` | Every ionic step of the 79 runs (1824 frames), with `REF_energy` (eV, the VASP free energy), `REF_forces` (eV/Å) and `REF_stress` (eV/Å^3, ASE sign convention). `run_path` gives the folder of each run in the archive. |
| `dft_final.csv` | The final structure of each run: Ti off-centring, tilts, c/a and the cubic lattice parameter implied by the strain. |
| `dft_a0.json` | The DFT cubic lattice parameter implied by the strain labels (used with `--strain-ref dft`). |

`audit.csv` (one row per VASP run, made by `00_audit.py`) is not included, since it lists local file paths.
