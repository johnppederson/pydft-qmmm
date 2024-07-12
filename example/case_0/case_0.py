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

from pydft_qmmm import *

# Load system first.
system = System.load(
    "../data/bbh.pdb",
    "../data/bmim.xml",
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
    "../data/bmim.xml",
    "../data/bmim_residues.xml",
)

# Define IXN Hamiltonian.
qmmm = QMMMHamiltonian("electrostatic", "cutoff")

# Define QM/MM Hamiltonian
total = qm[2580:] + mm[:2580] + qmmm

# Define the integrator to use.
integrator = VerletIntegrator(1)

# Define the logger.
logger = Logger(
    "./output/",
    system,
    dcd_write_interval=1,
    decimal_places=6,
)

# Define simulation.
simulation = Simulation(
    system=system,
    integrator=integrator,
    hamiltonian=total,
    logger=logger,
)

simulation.run_dynamics(10)
