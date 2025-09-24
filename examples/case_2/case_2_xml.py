from __future__ import annotations

from pydft_qmmm import *
from pydft_qmmm.plugins import SETTLE

# Load system first.
system = System.load("spce.pdb")

# Generate velocities.
system.velocities = generate_velocities(
    system.masses,
    300,
    10101,
)

# Define QM Hamiltonian.
qm = QMHamiltonian(
    basis="def2-SVP",
    functional="PBE",
    charge=0,
    multiplicity=1,
    guess="read",
)

# Define MM Hamiltonian.
mm = MMHamiltonian(
    forcefield=["spcfw.xml",
                "spcfw_residues.xml"],
    nonbonded_method="PME",
    pme_gridnumber=30,
    pme_alpha=5.0,
)

# Define IXN Hamiltonian.
qmmm = QMMMHamiltonian("electrostatic", "cutoff")

# Define QM/MM Hamiltonian.
total = qm[:3] + mm[3:] + qmmm

# Define the integrator to use.
integrator = VerletIntegrator(1)

# Define simulation.
simulation = Simulation(
    system=system,
    hamiltonian=total,
    integrator=integrator,
    output_directory="output_xml/",
    log_decimal_places=6,
    csv_decimal_places=6,
)

simulation.run_dynamics(10)
