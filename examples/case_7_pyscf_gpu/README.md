Example Case 7 (pyscf-mol on a GPU)
==================================

Summary
-------
This case serves as an example of running the QM region on a GPU
through [GPU4PySCF](https://github.com/pyscf/gpu4pyscf).  It applies
the QM/MM/PME coupling of example case 5 to the larger system of
example case 1: a chloromethane-chloride complex in 2000 TIP3P waters,
6006 atoms in a 39.3 Angstrom box, with the six-atom MCL complex as the
QM region at PBE0/6-31G.  Waters whose centroid lies within 8.0
Angstroms form subsystem II and enter the QM Hamiltonian as point
charges; everything beyond reaches it through the reciprocal sum.

Select the GPU with:

```python
from pydft_qmmm import QMHamiltonian

qm = QMHamiltonian(
    interface="pyscf-mol", basis="6-31G", functional="PBE0",
    charge=-1, multiplicity=1, device="gpu",
)
```

How to Run
----------
Run the example from a GPU compute node with CUDA, MKL, CuPy, and
GPU4PySCF available in the shell environment:

```bash
cd examples/case_7_pyscf_gpu
python case_7_gpu.py
```

If GPU4PySCF is used from a source checkout, add that checkout to
`PYTHONPATH` before running the script.  The environment must use a
CuPy build matching the loaded CUDA runtime.

What to Expect
--------------
The script prints energy components, timing, and QM forces. The
`PySCF` output component uses the `pyscf-mol` engine. A reference run
on a Tesla V100 produced:

```
=== device: gpu
total energy      -2598680.493960 kJ/mol   (12.0 s)
  PySCF           -2529617.729254
  OpenMM            -77644.982256
  PMENuclear          8582.217549

subsystem I        6 atoms
subsystem II     216 atoms
subsystem III   5784 atoms
```
