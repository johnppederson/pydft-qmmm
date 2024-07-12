#! /usr/bin/env python3
"""
QM/MM, a method to perform single-point QM/MM calculations using the
QM/MM/PME direct electrostatic QM/MM embedding method.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from qmmm_pme import *
from qmmm_pme.plugins import AtomEmbedding
from qmmm_pme.plugins import Stationary

# Load system first.
system = System(
    pdb_list=["../data/..."],
    topology_list=["../data/..."],
    forcefield_list=["../data/..."],
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
    pme_gridnumber=30,
)

# Define QM/MM Hamiltonian
qmmm = qm[:3] + mm[3:] | 14.0

# Define the integrator to use.
dynamics = VelocityVerlet(1, 300)

# Define the logger.
logger = Logger("./output/", system, dcd_write_interval=1, decimal_places=6)

# Define plugins.  The Stationary plugin will keep the electrode
# stationary, and the AtomEmbedding plugin will perform atom-wise
# analytical embedding in a QM/MM calculation.
stationary = Stationary(...)
atom_embedding = AtomEmbedding(...)

# Define simulation.
simulation = Simulation(
    system=system,
    hamiltonian=qmmm,
    dynamics=dynamics,
    logger=logger,
    plugins=[stationary, atom_embedding],
)

simulation.run_dynamics(10)
