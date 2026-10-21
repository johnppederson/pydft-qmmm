.. _`sec:getting_started`:

===============
Getting Started
===============

Installation
============

PyDFT-QMMM can be installed directly from github using ``pip``:

.. code-block:: bash

    $ python -m pip install git+https://github.com/johnppederson/pydft-qmmm

Alternatively, you can clone the repository and install using ``pip``:

.. code-block:: bash

    $ git clone https://github.com/johnppederson/pydft-qmmm
    $ cd pydft-qmmm
    $ pip install .

Optional QM engines
===================

Psi4 is the default QM engine. VASP, SPARC, and PySCF are also supported.
Select VASP with ``interface="vasp"`` or SPARC with ``interface="sparc"``
in ``QMHamiltonian``; each requires its external engine installation.

VASP requires a build with Python plugins. Use ``simulation.run_dynamics(10)``
to run ten dynamics steps. The simulation counts the starting geometry,
keeps VASP alive throughout the run, and closes it afterward. No separate
VASP mode or step count is needed. Calling ``calculator.calculate()`` alone
performs one energy/force evaluation and lets VASP finish normally.

To use PySCF, install:

.. code-block:: bash

    $ python -m pip install 'pydft_qmmm[pyscf]'

Choose ``interface="pyscf-mol"`` for a molecular QM region or
``interface="pyscf-pbc"`` for a periodic, Gamma-point calculation.
Both support a periodic MM environment.

.. code-block:: python

    qm = QMHamiltonian(
        interface="pyscf-mol",
        basis="def2-svp",
        functional="PBE",
        charge=0,
        multiplicity=1,
    )

Examples
========

Several cases are provided in the example suite.  The CLI and Python API
are demonstrated in :example:`0`.  Enhanced sampling functionality with
Plumed is demonstrated in :example:`1`. Using AMBER or GROMACS forcefield
parameter files is demonstrated in :example:`2`.

Templates
=========

Templates for writing third-party plugins and interfaces are provided in
the templates folder in the project root directory.
