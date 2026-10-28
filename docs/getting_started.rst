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

VASP dynamics and electrostatic embedding require a build with Python plugins.
Use ``simulation.run_dynamics(10)`` to run ten dynamics steps. The simulation
counts the starting geometry,
keeps VASP alive throughout the run, and closes it afterward. No separate
VASP mode or step count is needed. Calling ``calculator.calculate()`` alone
uses static VASP (``IBRION=-1, NSW=0``) for one energy/force evaluation.

Set ordinary electronic options such as ``EDIFF`` in ``incar``. The interface
leaves ``NBANDS``, ``WEIMIN``, and ``EBREAK`` to VASP unless you explicitly
provide them. During dynamics it sets ``IBRION=12`` and ``EDIFFG=0`` to let
PyDFT-QMMM control the geometries and number of steps. Single points retain
VASP's static electronic defaults. VASP derives different electronic defaults
in MD mode, so SP/MD comparisons should check energy and force convergence
with respect to ``EDIFF``.

VASP PME defaults
-----------------

With VASP, ``QMMMHamiltonian()`` now uses conservative PME-only embedding:
MM atoms start in region III, with no region II partition. An omitted PME
grid follows VASP's runtime grid, with order-6 splines. helPME builds the
field with ``compute_P_rec_grid()``; matching diagonal cells also use the
direct-grid force transpose. Install ``pydft_qmmm[qmmm-pme]`` for helpme-py
0.3.1 or newer.

Explicit PME grids, spline orders, and widths remain unchanged. VASP and
SPARC default to ``pme_alpha=2.357`` Å⁻¹, giving sigma approximately 0.3 Å.
``embedding_sigma`` separately sets the Gaussian FFT width. Other engines
continue to inherit OpenMM's splitting parameter when alpha is omitted:

.. code-block:: python

    coupling = QMMMHamiltonian()  # VASP: PME alpha = 2.357 Å⁻¹

Comparison routes remain available:

.. code-block:: python

    # Region II inside 6 Å, PME region III outside.
    coupling = QMMMHamiltonian(cutoff=6.)

    # FFT only: assign MM atoms to II and disable long-range coupling.
    from pydft_qmmm.utils import Subsystem
    mm = MMHamiltonian(
        interface="openmm", subsystem=Subsystem.II, forcefield=forcefield,
    )
    coupling = QMMMHamiltonian("electrostatic", "none", partition=None)

Explicit unequal-cell mappings and grids that helPME must pad retain
arbitrary-point evaluation. SPARC also uses the native implementation: when PME is enabled and no PME
grid is supplied, it uses ``fd_grid`` with order-6 splines. Its coupling and
partition defaults stay unchanged. PySCF-PBC, KSCED, and molecular engines
retain their independent arbitrary-point PME paths and existing defaults. The archived implementation is not a runtime dependency.

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
