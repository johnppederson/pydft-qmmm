<p align="center">
<img width=75% src="https://github.com/johnppederson/pydft-qmmm/blob/master/docs/media/pydft_qmmm.svg">
</p>

PyDFT-QMMM: A Modular Framework for DFT-QM/MM Simulation
========================================================

<p align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
[![License](https://img.shields.io/badge/license-LGPL_2.1-blue.svg)](https://opensource.org/license/lgpl-2-1)

</p>

Introduction
------------

PyDFT-QMMM implements QM/MM dynamics for several different PBC QM/MM
approaches, including the QM/MM/Cutoff and
[QM/MM/PME](https://doi.org/10.1063/5.0087386) methods.  Visit our
website for full documentation.

Requirements
------------

* Python >= 3.9
* Numpy
* OpenMM
* Psi4 >= 1.9

### Optional

#### Requirements for QM/MM/PME

* numba (used by QM/MM/PME utilities)
* [Psi4](https://github.com/johnppederson/psi4) (Psi4 implementing QM/MM/PME, must compile from source)
* [OpenMM](https://github.com/johnppederson/openmm) (OpenMM implementing QM/MM/PME, must compile from source)

#### Requirements for Plumed Plugin

* plumed (used for enhanced sampling simulations)

#### Requirements for GROMACS, AMBER, and CHARMM forcefields/topologies

* ParmEd (used for reading molecular mechanics topology files)

Installation
------------

The latest development version of the code can be installed using the
following command:

```bash
python -m pip install git+https://github.com/johnppederson/pydft-qmmm
```

To-Do
-----
- [ ] Add controls for logging QM and MM engine outputs through interfaces
- [ ] Allow for QM/MM/PME using default versions of QM and MM engines
