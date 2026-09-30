"""VASP QM/MM/PME energy and forces for one water in SPC/E solvent."""

from __future__ import annotations

from pydft_qmmm import MMHamiltonian, QMHamiltonian, QMMMHamiltonian, System
from pydft_qmmm.plugins import CentroidPartition
from pydft_qmmm.utils import center_positions, wrap_positions

system = System.load("box_128_min.pdb")

system.positions = center_positions(system.positions, system.box, [0, 1, 2])
system.positions = wrap_positions(
    system.positions, system.box, system.residue_map,
)

qm = QMHamiltonian(
    interface="vasp",
    charge=0,
    kpts=(1, 1, 1),
    encut=400,
    ediff=1e-8,
    ismear=0,
    sigma=0.05,
    embedding_sigma=0.3,
    incar={
        "GGA": "PE", "PREC": "Accurate", "ISYM": 0, "LREAL": False,
        "NGXF": 108, "NGYF": 108, "NGZF": 108,
    },
)

mm = MMHamiltonian(
    forcefield=["spce.xml", "spce_residues.xml"],
    nonbonded_method="PME",
    nonbonded_cutoff=7.0,
    pme_gridnumber=(40, 40, 40),
    pme_alpha=5.0,  # nm⁻¹ in OpenMM
)

qmmm = QMMMHamiltonian(
    "electrostatic", "electrostatic",
    partition=CentroidPartition("all", 3.0),
    pme_gridnumber=(40, 40, 40),
    pme_alpha=0.5,  # Å⁻¹ in helPME
)

total = qm[:3] + mm[3:] + qmmm

calculator = total.build_calculator(system)

results = calculator.calculate()
print(f"total energy {results.energy:.6f} kJ/mol")
for name, energy in results.components.items():
    if not name.startswith("."):
        print(f"  {name:<16s} {energy:18.6f}")
for region in ("I", "II", "III"):
    print(f"subsystem {region}: {len(system.select('subsystem ' + region))} atoms")
print("QM forces (kJ/mol/Å):")
for atom in sorted(system.select("subsystem I")):
    fx, fy, fz = results.forces[atom]
    print(f"  atom {atom}: {fx:14.6f} {fy:14.6f} {fz:14.6f}")
