Example Case 4 (PySCF)
======================

Summary
-------

These scripts calculate QM/MM/PME energies and forces with PySCF.
The molecular example uses `pyscf-mol` on the CPU; the periodic example
uses `pyscf-pbc` on a GPU. Each script uses the inputs in this directory.

| Script | QM system | Method |
| --- | --- | --- |
| `case_4_mol.py` | Water in SPC/E water | Molecular PBE/def2-SVP |
| `case_4_pbc.py` | Water in a 12 Å periodic cell | Periodic PBE/GTH-DZVP on GPU |

How to Run
----------

Install PySCF and the optional PME dependencies:

```bash
python -m pip install 'pydft_qmmm[pyscf]' 'pydft_qmmm[qmmm-pme]'
```

Run the desired script from this directory on a compute node:

```bash
python case_4_mol.py
python case_4_pbc.py
```

Run the periodic example on a GPU compute node with GPU4PySCF, CuPy,
and a compatible CUDA runtime. Set `device="cpu"` to run it on the CPU.

What to Expect
--------------

The scripts print energy components and QM forces.

The molecular QM wavefunction is not periodic; its MM environment is.
The periodic example uses the simulation box as the QM cell, a GTH
pseudopotential, and Gamma-point sampling. Converge `grid_level` for
molecular calculations and `ke_cutoff` or `mesh` for periodic forces.
