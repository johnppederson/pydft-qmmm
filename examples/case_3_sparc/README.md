Example Case 3 (SPARC)
======================

Summary
-------

Calculate the energy and forces of one QM water in a 12 Å box of four
SPC/E waters using SPARC, PBE, and electrostatic QM/MM/PME coupling.

Requires `sparc-x-api`, `helpme-py`, and a SPARC binary with the QM/MM
embedding implementation.
Set `SPARC_PSP_PATH` to the pseudopotential library if needed.

How to Run
----------

From this directory, set the engine command for your allocated resources:

```bash
export ASE_SPARC_COMMAND="mpirun -np 8 /path/to/sparc"
python case_3_pme.py --workdir sparc_pme_workdir
```

Add `--prepare-only` to check the setup without running SPARC. For
multi-node runs, use a working directory on shared storage.

What to Expect
--------------

The script prints the total energy, subsystem sizes, and QM forces.
SPARC inputs and outputs go to the selected working directory.
The target grid spacing is 0.12 Bohr, with an SCF tolerance of 1e-8.
Converge the grid for the system before running production calculations.
