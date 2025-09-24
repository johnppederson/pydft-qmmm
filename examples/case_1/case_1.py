from __future__ import annotations

from pydft_qmmm import *
from pydft_qmmm.plugins import Plumed
from pydft_qmmm.plugins import SETTLE

# Load system first.
system = System.load("cmc.pdb")

# Generate velocities.
system.velocities = generate_velocities(
    system.masses,
    300,
    10101,
)

# Define QM Hamiltonian.
qm = QMHamiltonian(
    basis="6-31G",
    functional="PBE0",
    charge=-1,
    multiplicity=1,
    dft_spherical_points = 194,
    dft_radial_points = 50,
    guess="read",
)

# Define MM Hamiltonian.
mm = MMHamiltonian(
    forcefield=["tip3p_cmc_no_intra.xml",
                "tip3p_cmc_residues.xml"],
)

# Define IXN Hamiltonian.
qmmm = QMMMHamiltonian("electrostatic", "cutoff")

# Define QM/MM Hamiltonian
total = qm[:6] + mm[6:] + qmmm

# Define the integrator to use.
integrator = LangevinIntegrator(1, 300, 0.005)

# Define plugins.
settle = SETTLE(
    oh_distance=0.9572,
    hh_distance=1.5136,
)
plumed = Plumed(
    """
    UNITS LENGTH=A TIME=fs
    dccl1: DISTANCE ATOMS=1,2
    dccl2: DISTANCE ATOMS=1,3
    diff: CUSTOM ARG=dccl1,dccl2 FUNC=x-y PERIODIC=NO
    restraint: RESTRAINT ARG=diff AT=0.0 KAPPA=750.0

    ox: GROUP ATOMS=7-6006:3
    hy: GROUP ATOMS=7-6006 REMOVE=ox
    cn1: COORDINATION GROUPA=2 GROUPB=hy R_0=2.5
    cn2: COORDINATION GROUPA=3 GROUPB=hy R_0=2.5
    cndiff: CUSTOM ARG=cn1,cn2 FUNC=x-y PERIODIC=NO

    PRINT ARG=dccl1,dccl2,diff,restraint.bias,cn1,cn2,cndiff FILE=COLVAR
    FLUSH STRIDE=25
    """,
    "output/plumed.log",
)

# Define simulation.
simulation = Simulation(
    system=system,
    hamiltonian=total,
    integrator=integrator,
    plugins=[settle, plumed],
    output_directory="output/",
    log_decimal_places=3,
    csv_decimal_places=3,
)

# Run simulation.
simulation.run_dynamics(1000)
plumed.plumed.finalize()
