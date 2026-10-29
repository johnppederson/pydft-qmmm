Example Case 5 (VASP)
=====================

Summary
-------

Calculate the energy and forces of one QM water in a 15.636 Å box of
123 SPC/E waters using VASP, PBE/PAW, and QM/MM/PME coupling.
The first water is QM; waters within 6 Å form subsystem II, and the
remaining MM waters form subsystem III.

Requires a VASP binary built with `-DPLUGINS`, a PBE POTCAR library,
and a compatible Python environment containing PyDFT-QMMM and
`helpme-py`.

How to Run
----------

From this directory, configure VASP for your allocated resources:

```bash
export VASP_PP_PATH=/path/to/potpaw_PBE.64
export PYTHONHOME=/path/to/conda/env
export PATH="$PYTHONHOME/bin:$PATH"
export PYDFT_QMMM_VASP_COMMAND="mpirun -np 1 /path/to/vasp_std"
python case_5_vasp.py
```

What to Expect
--------------

The script prints total and component energies, subsystem sizes, and
QM forces. VASP inputs and outputs go to `vasp_workdir/`.

The example uses Gamma-point sampling, `ENCUT=400` eV, `EDIFF=1e-8` eV,
a 108³ VASP grid, and a 40³ PME grid. Symmetry is disabled for embedding.
Converge the grids, cutoff, and Gaussian width for the target system.
