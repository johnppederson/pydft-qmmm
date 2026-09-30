Example Case 4 (VASP)
=====================

Compute energy and QM forces for the first water (atoms 0–2) in a
15.636 Å box of 123 SPC/E waters with `interface="vasp"`. The QM water
is centered and whole solvent molecules are wrapped once. Waters within
3 Å of the QM centroid form region II.

The default FFT embedding uses:

| Region | Electrostatic contribution |
|---|---|
| I | Periodic VASP QM Hamiltonian |
| II | Full periodic Gaussian-smeared Coulomb field |
| III | Region III-only reciprocal PME field |

I/II charges are removed before PME summation. An empty region III gives
zero reciprocal embedding. The plugin couples the field to electrons
and valence nuclei, using spline interpolation for nuclear energies and
forces. The coupling applies `PMEExcluded` as an energy-only correction;
its classical forces are already excluded.

Run
---

Requires a VASP binary built with `-DPLUGINS`, a PBE POTCAR library,
and a Python environment compatible with that binary containing
PyDFT-QMMM, OpenMM, NumPy, and helPME-py.

From this directory:

```bash
export VASP_PP_PATH=/path/to/potpaw_PBE.64
export PYTHONHOME=/path/to/conda/env
export PATH="$PYTHONHOME/bin:$PATH"
export PYDFT_QMMM_VASP_COMMAND="mpirun -np 1 /path/to/vasp_std"
python run.py
```

For the site-specific H200 template, edit the account, modules, Python
environment, VASP binary, and POTCAR paths in `submit.slurm`, then run
`sbatch submit.slurm` from this directory. It loads this repository via
`PYTHONPATH` and selects spline nuclear interpolation.

Settings and output
-------------------

- PBE/PAW, Gamma point, `ENCUT=400` eV, `EDIFF=1e-8` eV.
- `ISYM=0` and `LREAL=False`; embedding requires symmetry off.
- VASP fine grid 108³ (0.145 Å spacing), Gaussian width 0.3 Å.
- PME grid 40³; OpenMM real-space cutoff 7 Å.
- `MMHamiltonian.pme_alpha=5.0` nm⁻¹ and
  `QMMMHamiltonian.pme_alpha=0.5` Å⁻¹ specify the **same α**.

Converge the grids, cutoff, and Gaussian width for the target system.
The script prints energy components, region sizes, and QM forces.
VASP files are written to `vasp_workdir/`; `PLUGIN_FIRED.txt` confirms
embedding ran, and `PLUGIN_ERROR.txt` records callback failures.

For Gaussian-basis alternatives, see [case 5](../case_5_pyscf) for
`pyscf-mol` and [case 8](../case_8_pyscf_pbc) for `pyscf-pbc`.
