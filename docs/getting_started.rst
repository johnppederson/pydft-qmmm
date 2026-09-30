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

The Psi4 interface is the default for :class:`QMHamiltonian`.  A PySCF
interface is also bundled, and is installed with the ``pyscf`` extra.
Select ``interface="pyscf-mol"`` for molecular calculations or
``interface="pyscf-pbc"`` for periodic calculations.

Install the optional dependencies with:

.. code-block:: bash

    $ python -m pip install 'pydft_qmmm[pyscf,qmmm-pme]'

The supported model is a **molecular** QM region embedded in a
periodically replicated, fixed-charge MM environment.  The wavefunction
is restricted Kohn-Sham for a singlet and unrestricted Kohn-Sham for any
higher multiplicity.  This is not a :mod:`pyscf.pbc` periodic
wavefunction: there are no k-points, and the periodicity enters only
through the electrostatic environment that OpenMM and the PME machinery
provide.

The ``pyscf-pbc`` interface uses the simulation box as the QM cell.
It requires a GTH ``pseudo`` and a matching basis; ``ecp`` is unsupported.
Set ``ke_cutoff`` (Hartree) or ``mesh`` for the uniform FFT grid.
Sampling is a single k-point.

.. code-block:: python

    qm = QMHamiltonian(
        interface="pyscf-pbc",
        basis="gth-dzvp",
        pseudo="gth-pbe",
        functional="pbe",
        charge=0,
        multiplicity=1,
        ke_cutoff=200.0,
    )

Larger boxes increase QM cost. Converge ``ke_cutoff`` against forces;
energy convergence alone can leave large gradient errors.

For a molecular QM region:

.. code-block:: python

    qm = QMHamiltonian(
        interface="pyscf-mol",
        basis="def2-svp",
        functional="PBE0",
        charge=0,
        multiplicity=1,
        conv_tol=1e-9,
        max_cycle=100,
        grid_level=3,
    )
    mm = MMHamiltonian(forcefield=["forcefield.xml"])
    coupling = QMMMHamiltonian(
        "electrostatic",
        "electrostatic",
        coupling_mode="conservative",  # default
    )
    total = qm[qm_slice] + mm[mm_slice] + coupling

Subsystem II enters the QM Hamiltonian as :mod:`pyscf.qmmm` point
charges.  Subsystem III and the periodic images enter as a reciprocal
potential, sampled on the solver's own quadrature grid and contracted
into a one-electron operator.  Forces are analytic on both the QM atoms
and the MM sites.

The default conservative mode removes QM fixed-charge electrostatics
from both the OpenMM energy and its forces before adding the molecular
QM/MM terms.  ``coupling_mode="force"`` retains the historical
directional force mixing for compatibility, emits a warning, and is not
suitable when forces must be derivatives of the reported energy.

The solver follows the wavefunction unless one is named: Kohn-Sham
when a ``functional`` is given and Hartree-Fock otherwise, restricted
for a closed shell and unrestricted above it.  Naming ``method``
overrides that, and accepts ``rhf``, ``uhf``, ``rohf``, ``rks``,
``uks``, or ``roks``.  A Kohn-Sham method without a functional, a
Hartree-Fock method with one, and a restricted method on an open shell
are each refused rather than quietly reinterpreted.

``density_fit=True`` builds the Coulomb and exchange matrices from a
fitted auxiliary basis instead of the exact four-center integrals,
optionally with an explicit ``auxbasis``.  ``device="gpu"`` runs the
same calculation through GPU4PySCF; the embedding operators are still
built on the host and moved to the device, so the two agree to SCF
precision.

.. code-block:: python

    qm = QMHamiltonian(
        interface="pyscf-mol",
        basis="def2-svp",
        charge=0,
        multiplicity=1,
        method="rohf",
        density_fit=True,
        device="gpu",
    )

Unlike Psi4, PySCF does not infer an effective core potential from
the orbital basis, so a basis parameterized against one has to be paired
with it explicitly:

.. code-block:: python

    qm = QMHamiltonian(
        interface="pyscf-mol",
        basis="def2-svp",
        ecp="def2-svp",
        functional="PBE",
        charge=-1,
        multiplicity=1,
    )

Omitting ``ecp`` for such a basis is refused rather than silently
computed, because PySCF would otherwise place every electron, cores
included, into a valence-only basis and converge to a meaningless
energy.

An effective core potential also changes what the QM/MM/PME nuclear
term must use.  The wavefunction carries only the valence electrons, so
the matching nuclear charge is the atomic number less the electrons the
potential replaces -- 25 rather than 53 for iodine in a def2 basis.
Interfaces report that through :meth:`QMInterface.nuclear_charges`, and
the coupling Hamiltonian asks rather than assuming, so a QM region does
not enter the reciprocal sum with the wrong net charge.

``grid_level`` sets that quadrature.  The grid moves with the QM atoms,
and the terms that generates are omitted for the same reason PySCF omits
them from the exchange-correlation gradient by default, so the level
also sets how closely the analytic QM force tracks the energy: level 3
leaves roughly 0.07 kJ/mol/Angstrom on a single embedded water, and
level 5 roughly 0.003.

Examples
========

Several cases are provided in the example suite.  The CLI and Python API
are demonstrated in :example:`0`.  Enhanced sampling functionality with
Plumed is demonstrated in :example:`1`. Using AMBER or GROMACS forcefield
parameter files is demonstrated in :example:`2`.  QM/MM/PME with PySCF
as the QM engine is demonstrated in :example:`case 5 <5_pyscf>`, and a
heavy-atom QM region whose basis carries an effective core potential in
:example:`case 6 <6_pyscf_ecp>`.  :example:`Case 7 <7_pyscf_gpu>` repeats
case 5 on a GPU and compares the two devices.  :example:`Case 8
<8_pyscf_pbc>` uses a periodic ``pyscf.pbc`` wavefunction whose cell is
the simulation box.

Templates
=========

Templates for writing third-party plugins and interfaces are provided in
the templates folder in the project root directory.
