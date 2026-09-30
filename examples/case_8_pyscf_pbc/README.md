Example Case 8 (pyscf-pbc)
=========================

Compute one QM water's energy and forces in a 12 Å box of 27 SPC/E
waters with `interface="pyscf-pbc"`. The simulation box is the QM cell.
Waters within 4 Å of the QM centroid form region II.

| Region | Electrostatic contribution |
|---|---|
| I | Periodic QM Hamiltonian |
| II | Full periodic Gaussian-smeared Coulomb field |
| III | Region III-only reciprocal PME field |

Remove I/II sources before PME summation; local erf exclusions would
leave their image contributions. Empty III gives zero reciprocal field.
Nuclear energies and reaction forces use the same grid interpolation.

Run
---

Install the repository with its `pyscf` and `qmmm-pme` extras, then run
from this directory:

```bash
python case_8_pbc.py
```

Set `device="gpu"` for GPU4PySCF with CUDA. For a molecular wavefunction
in the periodic MM environment, use `pyscf-mol`; see [case 5](../case_5_pyscf).

Settings
--------

- Use a GTH `pseudo` and matching basis; `ecp` is unsupported.
- Set `ke_cutoff` (Hartree) or `mesh` instead of `grid_level`.
- Converge the FFT grid against forces. Energy convergence alone is insufficient.
- Larger boxes increase QM cost. Sampling is one k-point; stress is unsupported.

This example uses `gth-dzvp` and `ke_cutoff=200`. For the bare water
cell, the reported net-force component changes from 92.6 to −0.748
kJ/mol/Å between cutoffs 80 and 200, illustrating the grid sensitivity.
