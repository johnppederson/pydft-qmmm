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
    basis_set="def2-SVP",
    functional="PBE",
    charge=0,
    spin=1,
)

# Define MM Hamiltonian.
mm = MMHamiltonian(
    "spcfw.prmtop",
    # "spcfw.parm7",
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

# Define the logger.
logger = Logger("output_prmtop/", system, decimal_places=6)

# Define simulation.
simulation = Simulation(
    system=system,
    hamiltonian=total,
    integrator=integrator,
    logger=logger,
)

simulation.run_dynamics(10)
