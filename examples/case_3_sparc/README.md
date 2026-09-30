# SPARC water examples

The first water (atoms 0–2) is QM; the other three waters are SPC/E MM
in a 12 Å periodic cell. Both SPARC examples calculate one energy and
force evaluation using PBE, a target spacing of 0.12 Bohr (189³), SCF
tolerance 1e-8, and density mixing without a preconditioner (parameter 0.3).
The interface accepts `h` in Å, so the scripts convert from Bohr explicitly.

| Script | Coupling |
|---|---|
| `sparc/case_3_sparc_api.py` | Mechanical QM/MM |
| `sparc_electrostatic/case_3_sparc_electrostatic.py` | Electrostatic QM/MM/PME |

Install this repository and `sparc-x-api`; QM/MM/PME also needs `helpme-py`.
Use a SPARC binary containing the QM/MM embedding implementation for the
electrostatic example. Set `SPARC_PSP_PATH` if pseudopotentials are not
already available to SPARC-X-API.

From this directory, check the setup without launching SPARC:

```bash
python sparc_electrostatic/case_3_sparc_electrostatic.py --prepare-only
python sparc/case_3_sparc_api.py --prepare-only
```

Run on allocated compute resources. For an existing four-node Slurm
allocation with 24 ranks per node:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMM_CPU_THREADS=1
export ASE_SPARC_COMMAND="srun --exclusive -N 4 -n 96 /path/to/SPARC/lib/sparc"
python sparc_electrostatic/case_3_sparc_electrostatic.py --workdir sparc_pme_workdir
python sparc/case_3_sparc_api.py --workdir sparc_mechanical_workdir
```

On a workstation, use `mpirun -np 8 /path/to/SPARC/lib/sparc` instead.
Working directories must be on shared storage for multi-node runs.
Input paths are resolved relative to each script, so the examples can be
launched from another directory. Energies and QM forces print to stdout;
SPARC inputs and outputs go to `--workdir`. Preparation builds the
calculator and partition but does not write the external grid or run SCF.

The `psi4/` directory retains a short molecular mechanical-embedding
MD example. Its basis and boundary conditions differ from SPARC's;
it is not a numerical reference for periodic SPARC energies or forces.
