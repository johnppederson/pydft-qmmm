"""QM/MM mechanical-embedding example using Psi4 for the QM region.

Mirrors ``case_3_sparc`` but with Psi4 (PBE) as the QM engine, so the
two runs can be compared directly.  System: 4-water cluster in a 12 Å
periodic box; water 1 (atoms 0-2) is QM, waters 2-4 (atoms 3-11) are
SPC/E MM.  Mechanical embedding is used for both close- and long-range
coupling. SPARC also has a separate electrostatic-embedding example.
"""
from __future__ import annotations

from pydft_qmmm import *
from pydft_qmmm.plugins import SETTLE

system = System.load("water4.pdb")

system.velocities = generate_velocities(
    system.masses,
    300,
    10101,
)

# Psi4 uses a molecular basis; its numerical accuracy differs from
# the periodic real-space SPARC calculation.
qm = QMHamiltonian(
    interface="psi4",
    basis="def2-SVP",
    functional="PBE",
    charge=0,
    multiplicity=1,
    guess="read",
    scf_type="df",
    dft_spherical_points=302,
    dft_radial_points=75,
)

mm = MMHamiltonian(
    forcefield=["spce.xml", "spce_residues.xml"],
    nonbonded_method="CutoffPeriodic",
    nonbonded_cutoff=5.0,
)

# Mechanical embedding for both close- and long-range coupling, to
# match the SPARC run.
qmmm = QMMMHamiltonian("mechanical", "mechanical")

total = qm[:3] + mm[3:] + qmmm

integrator = VerletIntegrator(1)

settle = SETTLE()

simulation = Simulation(
    system=system,
    hamiltonian=total,
    integrator=integrator,
    plugins=[settle],
    output_directory="output_api/",
    log_decimal_places=6,
    csv_decimal_places=6,
)

simulation.run_dynamics(2)
