"""Periodic PySCF QM/MM single point in a 12 Å water box."""
from __future__ import annotations

from pydft_qmmm import *
from pydft_qmmm.plugins import CentroidPartition

# CRYST1 sets the periodic QM cell.
system = System.load("spce_small.pdb")

# The pseudopotential is required; converge ke_cutoff against forces.
qm = QMHamiltonian(
    interface="pyscf-pbc",
    basis="gth-dzvp",
    pseudo="gth-pbe",
    functional="pbe",
    charge=0,
    multiplicity=1,
    ke_cutoff=200.0,
    conv_tol=1e-10,
    # device="gpu",   # needs GPU4PySCF and a CUDA runtime
)

mm = MMHamiltonian(
    forcefield=["spce.xml", "spce_residues.xml"],
    nonbonded_method="PME",
    nonbonded_cutoff=5.0,
    pme_gridnumber=24,
    pme_alpha=5.0,
)

qmmm = QMMMHamiltonian(
    "electrostatic",
    "electrostatic",
    partition=CentroidPartition("all", 4.0),
)

# The first water is QM.
total = qm[:3] + mm[3:] + qmmm

calculator = total.build_calculator(system)

results = calculator.calculate()

print(f"total energy   {results.energy:18.6f} kJ/mol")
for name, energy in results.components.items():
    if not name.startswith("."):
        print(f"  {name:<28s} {energy:18.6f}")
print()
print(f"subsystem I   {len(system.select('subsystem I')):6d} atoms")
print(f"subsystem II  {len(system.select('subsystem II')):6d} atoms")
print(f"subsystem III {len(system.select('subsystem III')):6d} atoms")
print()
print("forces on the QM atoms (kJ/mol/A):")
for atom in sorted(system.select("subsystem I")):
    fx, fy, fz = results.forces[atom]
    print(f"  atom {atom} {fx:14.6f} {fy:14.6f} {fz:14.6f}")
print()
print(f"net force      {results.forces.sum(axis=0)}")
