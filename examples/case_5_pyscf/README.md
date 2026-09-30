Example Case 5 (pyscf-mol)
==========================

Compute the energy and forces of one QM water in SPC/E water with
`interface="pyscf-mol"`, PBE/def2-SVP, and `grid_level=5`.
The wavefunction is molecular; the MM environment is periodic.

Waters within 8 Å of the QM centroid form region II and enter as point
charges. Region III enters through PME. The default conservative
coupling removes QM force-field charges from OpenMM.

Run from this directory after installing the repository with its
`pyscf` and `qmmm-pme` extras:

```bash
python case_5_pme.py
```

The script prints total and component energies, region sizes, and QM
forces. `PySCF` contains the electronic and near-field nuclear embedding;
`PMENuclear` adds the reciprocal nuclear term. These output component
names differ from the engine selection name `pyscf-mol`.

Increase `grid_level` to check molecular quadrature convergence.
Set `device="gpu"` for GPU4PySCF with CUDA. For a periodic QM wavefunction,
use [case 8](../case_8_pyscf_pbc) with `pyscf-pbc`.
