"""QM/MM mechanical energy and forces for a water in a 12 Å periodic cell."""
from __future__ import annotations

import argparse
from pathlib import Path

from pydft_qmmm import MMHamiltonian, QMHamiltonian, QMMMHamiltonian, System

BOHR_ANGSTROM = 0.529177210903
HERE = Path(__file__).resolve().parent


def build_calculator(workdir):
    system = System.load(str(HERE / "water4.pdb"))
    qm = QMHamiltonian(
        interface="sparc", charge=0, xc="pbe",
        # h is in Å. The interface pins FD_GRID from the cell dimensions.
        h=0.12 * BOHR_ANGSTROM,
        kpts=(1, 1, 1), tol_scf=1e-8,
        mixing_variable="density", mixing_precond="none", mixing_parameter=0.3,
        directory=str(workdir), embedding=False, embedding_sigma=0.3,
    )
    mm = MMHamiltonian(
        interface="openmm",
        forcefield=[str(HERE / "spce.xml"), str(HERE / "spce_residues.xml")],
        nonbonded_method="PME", nonbonded_cutoff=5.0,
        pme_gridnumber=(40, 40, 40), pme_alpha=5.0,  # nm⁻¹
    )
    qmmm = QMMMHamiltonian("mechanical", "mechanical", cutoff=2.5)
    total = qm[:3] + mm[3:] + qmmm
    calculator = total.build_calculator(system)
    qmmm.partition.generate_partition()
    return system, calculator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Build the calculator and check the partition without running SPARC.")
    parser.add_argument("--workdir", type=Path, default=Path("sparc_workdir"))
    args = parser.parse_args()
    system, calculator = build_calculator(args.workdir.resolve())
    print("SPARC target spacing: 0.12 Bohr; grid: 189³ for this 12 Å cell")
    for region in ("I", "II", "III"):
        print(f"subsystem {region}: {len(system.select('subsystem ' + region))} atoms")
    if args.prepare_only:
        print("Preparation complete; no SPARC calculation launched.")
        return
    results = calculator.calculate()
    print(f"Total energy: {results.energy:.6f} kJ/mol")
    print("QM forces (kJ/mol/Å):")
    for atom in sorted(system.select("subsystem I")):
        print(f"  atom {atom}: " + " ".join(f"{x:14.6f}" for x in results.forces[atom]))


if __name__ == "__main__":
    main()
