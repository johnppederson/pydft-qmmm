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

Persistent VASP (experimental)
-----------------------------

To keep orbitals and charge-density prediction in memory across QM/MM steps,
pass these options to `QMHamiltonian(interface="vasp", ...)`:

```python
persistent=True,
persistent_steps=11,       # initial geometry + 10 MD steps
persistent_timeout=3600.,  # timeout per SCF / driver exchange, seconds
incar={"ISYM": 0, "IWAVPR": 11},
```

The interface sets `IBRION=12`, `PLUGINS/STRUCTURE=True`, and `NSW` to the
number of force evaluations. PyDFT-QMMM still integrates the full system;
VASP receives the next QM geometry through its Python structure callback.
The driver rewrites MM charges and PME sources before releasing that callback,
which invalidates the cached external potential. Changes in MM coordinates,
charges, and the II/III partition are supported, even when QM atoms do not
move. Changing the cell, QM membership/species, or electronic settings requires
a new calculator. Use a separate working directory for each live calculator.

`IWAVPR=11` predicts charge density; `12` also predicts orbitals and requires
more memory. `0` retains in-memory reuse without prediction and is useful as
a control. These settings do not remove the computational cost of the FFT
and embedding grids. Validate energies, forces, and convergence before judging
speed. The protocol returns completed ionic energies and forces; it does not
certify electronic convergence, so inspect OUTCAR as with the restart path.

Call `calculator.potential.close()` in a `finally` block (for a composite
QM/MM calculator, use `calculator.calculators[0].potential.close()`). After
exactly `persistent_steps` evaluations VASP exits normally and writes its
final outputs. Closing earlier terminates the process group and does not
guarantee a final WAVECAR. A timeout or process failure aborts the session;
create a new calculator to restart. Energy/force requests for unchanged
system state reuse the usual calculator cache and do not count as new steps.

`persistent_check.py` compares three embedded-water frames against separate
VASP runs for `IWAVPR=0,11,12`, including a frame that moves only MM atoms and
one that changes the partition. Run it on a compute node with the same
environment as `run.py`. Set `CHECK_DIR` to select its output directory.
The result and comparison JSON files contain energies, forces, and timings.

References: [VASP structure plugin](https://vasp.at/wiki/PLUGINS/STRUCTURE)
and [IWAVPR](https://vasp.at/wiki/IWAVPR).
