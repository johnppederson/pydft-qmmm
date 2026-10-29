"""Molecular PySCF QM/MM/PME single point."""
from __future__ import annotations

from pydft_qmmm import *
from pydft_qmmm.plugins import CentroidPartition

system = System.load("spce.pdb")

# grid_level sets the XC and embedding quadrature.
qm = QMHamiltonian(
    interface="pyscf-mol",
    basis="def2-svp",
    functional="PBE",
    charge=0,
    multiplicity=1,
    conv_tol=1e-10,
    grid_level=5,
)

mm = MMHamiltonian(
    forcefield=["spce.xml", "spce_residues.xml"],
    nonbonded_method="PME",
    nonbonded_cutoff=14.0,
    pme_gridnumber=30,
    pme_alpha=5.0,
)

qmmm = QMMMHamiltonian(
    "electrostatic",
    "electrostatic",
    partition=CentroidPartition("all", 8.0),
)

total = qm[:3] + mm[3:] + qmmm

calculator = total.build_calculator(system)

results = calculator.calculate()

print(f"total energy   {results.energy:18.6f} kJ/mol")
for name, energy in results.components.items():
    if not name.startswith("."):
        print(f"  {name:<12s} {energy:18.6f}")
print()
print(f"subsystem I   {len(system.select('subsystem I')):6d} atoms")
print(f"subsystem II  {len(system.select('subsystem II')):6d} atoms")
print(f"subsystem III {len(system.select('subsystem III')):6d} atoms")
print()
print("forces on the QM atoms (kJ/mol/A):")
for atom in sorted(system.select("subsystem I")):
    fx, fy, fz = results.forces[atom]
    print(f"  atom {atom} {fx:14.6f} {fy:14.6f} {fz:14.6f}")
