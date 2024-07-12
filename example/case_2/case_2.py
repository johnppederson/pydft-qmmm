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
from qmmm_pme.plugins import SETTLE

# Load system first.
system = System(
    "../data/spce.pdb",
    "../data/spce.xml",
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
    "../data/spce.xml",
    "../data/spce_residues.xml",
)

# Define IXN Hamiltonian.
qmmm = QMMMHamiltonian("electrostatic", "cutoff")

# Define QM/MM Hamiltonian
total = qm[:3] + mm[3:] + qmmm

# Define the integrator to use.
integrator = VerletIntegrator(1)

# Define the logger.
logger = Logger("./output/", system, dcd_write_interval=1, decimal_places=6)

# Define plugins.
settle = SETTLE()

# Define simulation.
simulation = Simulation(
    system=system,
    hamiltonian=qmmm,
    dynamics=dynamics,
    logger=logger,
    plugins=[settle],
)

simulation.run_dynamics(10)
