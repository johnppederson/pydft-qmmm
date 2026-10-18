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

The ``pyscf-mol`` interface uses a molecular QM region embedded in a
periodically replicated, fixed-charge MM environment.  The QM wavefunction
has no periodic boundary conditions or k-points.  Periodicity enters through
the electrostatic environment that OpenMM and the PME machinery provide.

The ``pyscf-pbc`` interface uses a periodic QM cell embedded in a periodically
replicated, fixed-charge MM environment.  Both use the same simulation box
and periodic boundary conditions.  The QM calculation uses :mod:`pyscf.pbc`
with Gamma-point sampling.
It requires a GTH ``pseudo`` and a matching basis; ``ecp`` is unsupported.
Set ``ke_cutoff`` (Hartree) or ``mesh`` for the uniform FFT grid.

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

To embed active QM atoms in a frozen QM density, add ``ksced`` to the same
periodic interface. This requires the ``pyscf-ksced`` plugin with analytic
gradients, installed or available through ``PYSCF_EXT_PATH``.

.. code-block:: python

    qm = QMHamiltonian(
        interface="pyscf-pbc",
        basis="gth-dzvp",
        pseudo="gth-pbe",
        functional="PBE",
        charge=0,
        multiplicity=1,
        mesh=(41, 41, 41),
        ksced={
            "active_atoms": [0, 1, 2],
            "frozen_atoms": [3, 4, 5],
            "basis_mode": "M",
            "t_nad": "LDA_K_TF",
        },
    )
    calculator = qm[0:3].build_calculator(system)
    result = calculator.calculate()

The active and frozen selections must be nonempty and disjoint. Each can be
a list of system atom indices or a selection string. Only active A belongs
to subsystem I. Frozen B stays in the MM region and is selected separately
as the KSCED environment. This keeps B–C interactions in the MM force field;
there is no frozen-density B–C electrostatic term.
The outer ``charge`` and ``multiplicity`` describe active A. Frozen B defaults
to charge 0 and multiplicity 1; set ``frozen_charge`` and
``frozen_multiplicity`` inside ``ksced`` to change them.

B is converged once and reused as A moves. The returned energy is E_ainb,
without B's constant self energy. Analytic A forces are placed in full-system
atom order; B rows are zero. With electrostatic embedding enabled, the energy
includes A–C electronic and nuclear coupling, and C receives reaction forces.
B's positions and the simulation box must
stay fixed. The interface does not constrain an integrator automatically.

This option supports Gamma-point ``basis_mode="M"` with GTH pseudopotentials,
FFTDF and pure LDA/GGA XC and kinetic functionals, on CPU or GPU using the
existing ``device`` option. Adding ``sigma`` and ``smearing_method="fermi"``
to the outer options applies fixed-electron-number smearing to A; its reported
energy is then the free energy differentiated by the forces. The interface's
``configure_electrostatic_embedding(True)`` enables the existing periodic
Gaussian charge field of region II. Added ``PMEElectronicPotential`` fields
supply region III reciprocal electrostatics. Both source sets exclude frozen B,
independently of B's II/III assignment. These fields couple only to A's density
and valence nuclear charges. Arbitrary added fields are rejected because their
source exclusions and force derivatives cannot be guaranteed.

Configure embedding before the first SCF. The Gaussian near field is periodic,
with its width set by ``embedding_sigma``; it is not an unsmoothed isolated
point-charge interaction. The returned components sum to the total energy.

``compute_components()`` reports additive terms in kJ/mol:

* KSCED Active Energy: A's own energy evaluated at the embedded density,
  including its internal nuclear repulsion, but excluding A-B and A-C terms.
* A-B Nuclear Interaction, A-B Electron-Nuclear Interaction, and
  A-B Hartree Interaction.
* Non-additive XC and Non-additive Kinetic.
* Electronic Entropy Correction: free energy minus internal energy (zero
  without smearing).
* A-C Near Electronic Embedding and A-C Near Nuclear Embedding.
* A-C Reciprocal Electronic Embedding and A-C Reciprocal Nuclear Embedding.

The near and reciprocal fields are evaluated at the same converged A density.
All component keys remain present when embedding is disabled or a field is
absent, with zero values for absent contributions. Components are cached until
positions, charges, elements, partitions, or the box change. Asking for them
neither reconverges B nor changes the energy or force operator. They replace
the former three aggregate component labels. B's constant self energy is
omitted; B-C MM energy must be reported by the MM calculator.

Full OpenMM assembly still requires excluding A–B force-field terms while
retaining B–C terms; this interface does not perform that setup or freeze B.

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
from both the OpenMM energy and its forces before adding the QM/MM
terms. This applies to molecular engines and periodic engines (VASP,
SPARC, and PySCF-PBC, including KSCED). QM/MM Lennard-Jones interactions
retain both their energy and forces, and the original system charges
remain available to the embedding engine. Conservative PME omits the
classical PME exclusion correction because the OpenMM QM charges are
already zero. Each engine keeps ownership of its existing nuclear and
electronic embedding terms. ``coupling_mode="force"`` retains the historical
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

Separate VASP and MM cells
-------------------------

For a slab, the VASP interface can use a shorter QM cell along z while
``System.box`` remains the full periodic MM cell. Pass ``qm_box`` (lattice
vectors as columns), ``qm_origin`` (Cartesian Angstrom), and
``embedding_grid_shape`` (the full MM Gaussian grid) to ``QMHamiltonian``.
Set ``NGXF``, ``NGYF``, and ``NGZF`` explicitly in ``incar`` so that the QM
and MM grids have identical spacing. The z origin must lie on that grid.
Only orthorhombic cells with identical x/y vectors are currently supported;
``persistent`` must be false. Both region I and region II must lie strictly
inside the QM cell's z faces. Coordinates in ``System`` remain global.

Enable slab Coulomb truncation through the INCAR keys
``KERNEL_TRUNCATION/LTRUNCATE=True``,
``KERNEL_TRUNCATION/IDIMENSIONALITY=2``,
``KERNEL_TRUNCATION/ISURFACE=3``, and
``KERNEL_TRUNCATION/LCOARSEN=True``. These settings affect VASP's QM
self-interactions. MM embedding retains the original MM periodic lattice.
The Gaussian field is restricted to the QM grid and its reaction force uses
the transpose of that restriction. PME uses the MM lattice and global QM
grid coordinates in both directions. Cell metadata is written to
``QM_CELL.json`` and checked against the actual VASP callback grid.

Converge the QM z height against a full-height truncated reference. Also
compare that reference against the original periodic model: truncation
changes the physical boundary condition. Dispersion image interactions
require a separate cell-size check.

Neutral water slabs with separate QM and MM cells
------------------------------------------------

For fixed-region VASP/OpenMM calculations, the optional
``QMMMHamiltonian("electrostatic", "electrostatic", partition=None,
slab_correction=True)`` adds the neutral Yeh--Berkowitz correction to
MM electrostatics and the matching QM/MM cross term. The correction is
disabled by default. Configure VASP with an explicit aligned ``qm_box``,
``qm_origin`` and ``embedding_grid_shape``, ``persistent=False``, and
``KERNEL_TRUNCATION/LTRUNCATE=True``, ``IDIMENSIONALITY=2``, ``ISURFACE=3``.
The last two tags also use the ``KERNEL_TRUNCATION/`` prefix.

The MM term is C M_MM² and the cross term is 2 C M_MM M_QM, with
C = 1/(2 ε₀ V_MM). M_QM includes the final electron density and ionic
valence charges. The electronic cross term enters the SCF potential;
ionic energy, ionic forces and MM reaction forces follow the same energy.
There is no extra QM self correction in this mode because VASP already truncates it.
The OpenMM wrapper evaluates the MM energy and its analytic derivative
using the current charges after QM exclusions; no new integrator or
MDCraft dependency is required.

This implementation requires separately neutral QM and MM regions,
orthorhombic cells, PME for MM, and continuous slab coordinates along z.
Keep the cell and region membership fixed. It does not resize the cell
or add vacuum. The approximation still needs vacuum-convergence checks;
it is not an exact two-periodic solver. No slab stress correction is
provided, so pressure-coupled dynamics are not supported. Water tests
with VASP kernel coarsening disabled passed the current finite-difference
tolerance before adding this correction; full coupled validation remains
necessary.

The QM grid neutrality check allows an absolute residual of 1e-4 electrons
(the water test showed a 2.16e-5 electron integration residual). The density
is never renormalized; larger discrepancies fail. ``SLAB_DIAGNOSTICS.json``
records the residual, tolerance and electron count by z plane.

For periodic QM with a full-height or smaller aligned QM cell, additionally set
``qm_slab_correction=True`` on the coupling Hamiltonian. Disable kernel
truncation and keep ``LDIPOL=False``. The plugin adds C_QM M_QM², updating
its potential -2 C_QM M_QM z at each SCF step. Its energy correction removes
the potential contribution already counted in the band energy. The
ionic force is -2 C_QM M_QM Z along z. Here C_QM = 1/(2 ε₀ V_QM);
the MM and cross terms retain C_MM = 1/(2 ε₀ V_MM). The total correction
is C_QM M_QM² + 2 C_MM M_QM M_MM + C_MM M_MM². For equal cell volumes
this reduces to C (M_QM + M_MM)². Place the QM cell faces in vacuum so the
linear potential's discontinuity does not cut through the QM density.
This mode requires full coupled finite-difference validation before MD.
