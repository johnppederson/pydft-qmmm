<p align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/johnppederson/pydft-qmmm/blob/master/docs/_media/pydft-qmmm-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/johnppederson/pydft-qmmm/blob/master/docs/_media/pydft-qmmm-light.svg">
  <img alt="The PyDFT-QMMM logo." width=75% src="https://github.com/johnppederson/pydft-qmmm/blob/master/docs/_media/pydft-qmmm-light.svg">
<picture>

</p>

PyDFT-QMMM: A Modular Framework for DFT-QM/MM Simulation
========================================================

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=fff)
[![License](https://img.shields.io/badge/license-LGPL_2.1-blue.svg)](https://opensource.org/license/lgpl-2-1)

[![Build](https://github.com/johnppederson/pydft-qmmm/actions/workflows/test_and_coverage.yml/badge.svg)](https://github.com/johnppederson/pydft-qmmm/actions)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/johnppederson/f0e19ee2c0a71030ace6067046a59ff1/raw/pydft_qmmm.json)](https://github.com/johnppederson/pydft-qmmm/actions)
<!-- [![Deployment](https://github.com/johnppederson/pydft-qmmm/actions/workflows/build_and_deploy.yml/badge.svg)](https://pypi.org/project/pydft-qmmm/) -->
[![Docs](https://github.com/johnppederson/pydft-qmmm/actions/workflows/docs.yml/badge.svg)](https://johnppederson.com/pydft-qmmm/)

</p>

Introduction
------------

PyDFT-QMMM implements QM/MM dynamics for several different PBC QM/MM
approaches, including the QM/MM/Cutoff and
[QM/MM/PME](https://doi.org/10.1063/5.0087386) methods.  Visit our
[website](https://johnppederson.com/pydft-qmmm/) for full documentation.

Requirements
------------
* Python >= 3.10
* [NumPy](https://github.com/numpy/numpy)
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).
* [OpenMM](https://github.com/openmm/openmm)
  [(OpenMM licenses)](https://github.com/openmm/openmm/blob/master/docs-source/licenses/Licenses.txt).
* A QM engine, such as the default [Psi4](https://github.com/psi4/psi4) >= 1.10
  [(LGPL-3.0 license)](https://opensource.org/license/LGPL-3-0).

### Configured environments

| Engine | Engine source |
| --- | --- |
| Psi4 |  Psi4 1.11 |
| PySCF-mol | PySCF 2.14; GPU4PySCF for GPU calculations |
| PySCF-pbc | PySCF 2.14 periodic solver; GPU4PySCF for GPU calculations |
| VASP | External VASP 6.6.1 executable plus the Python plugin |
| SPARC-QMMM | sparc-x-api plus an external SPARC-QMMM executable. **⚠ Under development.** |

The environment needs these important packages:

* Python 3.10.20
* Psi4 1.11
* OpenMM 8.5.2
* NumPy 2.2.6
* helPME-py 0.2.2
* QCEngine and QCElemental
* ASE 3.29.0
* Pint 0.24.4

#### Required for QM/MM/PME
* [helPME-py](https://github.com/johnppederson/helpmy-py) required for evaluating
  a PME potential on an arbitrary set of coordinates
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).

#### Required for Enhanced Sampling
* [PLUMED](https://github.com/plumed/plumed2) required for enhanced sampling
  [(LGPL-3.0 license)](https://opensource.org/license/LGPL-3-0).

#### Required for Optimization
* [geomeTRIC](https://github.com/leeping/geomeTRIC) required for optimization
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).

#### Required for Expanded MM Input File Types
* [ParmEd](https://github.com/ParmEd/ParmEd) required for reading GROMACS, AMBER,
  and CHARMM forcefields/topologies [(LGPL-2.1 license)](https://opensource.org/license/LGPL-2-1).

#### Required for Testing
* [pytest](https://github.com/pytest-dev/pytest) required for performing tests
  [(MIT license)](https://opensource.org/license/MIT).
* [pytest-cov](https://github.com/pytest-dev/pytest-cov) required for performing coverage
  analysis [(MIT license)](https://opensource.org/license/MIT).

#### Required for Development
* [pre-commit](https://github.com/pre-commit/pre-commit) required for maintaining typing and
  formatting standards [(MIT license)](https://opensource.org/license/MIT).

#### Required for Documentation
* [Sphinx](https://github.com/sphinx-doc/sphinx) required for generating documentation
  [(BSD-2-clause license)](https://opensource.org/licenses/BSD-2-Clause).
* [sphinx-autodoc-typehints](https://github.com/tox-dev/sphinx-autodoc-typehints) required for
  generating type hints [(MIT license)](https://opensource.org/licenses/MIT).
* [Furo](https://github.com/pradyunsg/furo) theme for Sphinx documentation
  [(MIT license)](https://opensource.org/license/MIT).

Installation
------------

PyDFT-QMMM can be installed directly from github using ``pip``:

```bash
python -m pip install git+https://github.com/johnppederson/pydft-qmmm
```

Alternatively, you can clone the repository and install using ``pip``:

```bash
git clone https://github.com/johnppederson/pydft-qmmm
cd pydft-qmmm
pip install .
```


Reusable MD setup
-----------------

`MMHamiltonian` accepts OpenMM setup options directly. For XML force fields,
`lj_mixing="geometric"` uses geometric LJ sigma and reads `NBFixPair` entries
under `NonbondedForce`. Optional `lj_lorentz_classes` use arithmetic sigma for
pairs involving those classes. `lj_zero_classes` sets pair-table epsilon to
zero; existing 1-4 exceptions retain their energy scaling. These options do not
change the default force-field behavior.

```python
from pydft_qmmm import MMHamiltonian
from pydft_qmmm.utils import Subsystem

mm = MMHamiltonian(
    forcefield=["forcefield.xml", "residues.xml"],
    subsystem=Subsystem.II,  # fixed direct embedding; default is III
    lj_mixing="geometric",
    nonbonded_exclusions=("resname AUA", "resname AUC"),
)
```

Each exclusion query defines a group. Nonbonded pairs within each group are
excluded; pairs between groups and bonded terms are retained. Exclusions are
applied before OpenMM contexts and QM/MM exclusions are constructed.

The existing velocity helper can keep selected atoms at rest and rescale only
the mobile atoms. Use the same selection with the `Stationary` plugin during MD.
After building the calculator, ensure mobile atoms have their physical masses.

```python
from pydft_qmmm import Simulation, generate_velocities
from pydft_qmmm.plugins import Stationary

frozen = sorted(system.select("resname AUC"))
system.velocities = generate_velocities(
    system.masses, 300, 10101, frozen=frozen, rescale=True,
)
stationary = Stationary("atom " + " ".join(map(str, frozen)))
with Simulation(
    system=system, calculator=calculator, integrator=integrator,
    plugins=[stationary], center=False, wrap=False,
    output_directory="output/", dcd_write_interval=1,
    dcd_timestep=integrator.timestep,
) as simulation:
    simulation.run_dynamics(10)
```

The context manager closes engine processes and output files on normal exit or
an exception. The usual DCD, CSV, and log output is retained. Centering and
wrapping default to enabled; disable them for a fixed QM-cell origin.

For VASP, `QMHamiltonian(interface="vasp", check_convergence=True, ...)` checks
EDIFF convergence in `OUTCAR`. A persistent run checks all SCFs after its final
evaluation and normal process exit. Set `persistent_steps` to the MD step count
plus one for the initial energy/force evaluation. An early stop closes the
process but does not perform the final convergence check.
