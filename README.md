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
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/johnppederson/f0e19ee2c0a71030ace6067046a59ff1/raw/pydft_qmmmm.json)](https://github.com/johnppederson/pydft-qmmm/actions)
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
* [Psi4](https://github.com/psi4/psi4) >= 1.10
  [(LGPL-3.0 license)](https://opensource.org/license/LGPL-3-0).

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
