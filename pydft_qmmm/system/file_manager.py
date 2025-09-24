"""Utilities for reading and writing PDB files.
"""
from __future__ import annotations

__all__ = [
    "load_system",
    "write_to_pdb",
]

import pathlib
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.utils import ELEMENT_TO_MASS
from pydft_qmmm.utils import compute_lattice_constants
from pydft_qmmm.utils import compute_lattice_vectors
from pydft_qmmm.utils import check_array

from .atom import Atom

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pydft_qmmm import System


def load_system(*args: str) -> tuple[list[Atom], NDArray[np.float64]]:
    """Load system data from PDB files.

    Args:
        args: The PDB file or list of PDB files with position,
            element, name, residue, residue name, chain, and lattice
            vector data.

    Returns:
        A list of atom objects and the box vectors.
    """
    atoms = []
    boxes = []
    residue = 0
    for filename in args:
        tmp_atoms, tmp_box = _read_pdb(filename, residue)
        residue = tmp_atoms[-1].residue + 1
        atoms.extend(tmp_atoms)
        boxes.append(tmp_box)
    if len(boxes) == 1:
        return atoms, boxes[0]
    for i in range(len(boxes) - 1):
        if not np.allclose(boxes[i], boxes[i+1]):
            raise OSError(
                "Multiple PDB files have been loaded with different lattices.",
            )
    return atoms, boxes[-1]


def _read_pdb(
        pdb_file: str,
        residue: int = 0
) -> tuple[list[Atom], NDArray[np.float64]]:
    """Extract system data from a PDB file.

    Args:
        pdb_file: A PDB file with position, element, name, residue,
            residue name, chain, and lattice vector data.

    Returns:
        A list of atom objects and the box vectors.
    """
    atoms: list[Atom] = []
    box = np.zeros((3, 3))
    resnum_old = 0
    # Read the PDB file.
    with open(pdb_file) as fh:
        lines = fh.readlines()
    for line in lines:
        # Extract lattice information and construct box.
        if line.startswith("CRYST1"):
            a = float(line[6:15].strip())
            b = float(line[15:24].strip())
            c = float(line[24:33].strip())
            A = float(line[33:40].strip())
            B = float(line[40:47].strip())
            G = float(line[47:54].strip())
            box = compute_lattice_vectors(a, b, c, A, B, G)
        # Add an atom to the list of Atom objects.
        if line.startswith("ATOM") or line.startswith("HETATM"):
            name = line[12:16].strip()
            resname = line[17:21].strip()
            chain = line[21].strip()
            resnum = int(line[22:26])
            # Advance the residue counter if the residue name, residue
            # sequence number, or chain identifier changes once there is
            # at least one Atom in the list of Atom objects.
            if atoms:
                if (atoms[-1].residue_name != resname
                        or atoms[-1].chain != chain
                        or resnum_old != resnum):
                    residue += 1
            position = np.array(
                [float(line[30:38].strip()),
                 float(line[38:46].strip()),
                 float(line[46:54].strip())],
            )
            element = line[76:78].strip().lower().capitalize()
            mass = ELEMENT_TO_MASS.get(element, 0.)
            atom = Atom(
                position=position,
                mass=mass,
                residue=residue,
                element=element,
                name=name,
                residue_name=resname,
                chain=chain,
            )
            atoms.append(atom)
            resnum_old = resnum
    return atoms, box


def write_to_pdb(name: str, system: System) -> None:
    r"""Write a system to a PDB file.

    Args:
        name: The PDB file to which the system will be written.
        system: The system whose data will be written to a PDB file.
    """
    filename = pathlib.Path(name).with_suffix(".pdb")
    a, b, c, A, B, G = compute_lattice_constants(system.box)
    residues: dict[str, list[int]] = {}
    for i in range(len(system)):
        residue = residues.get(system.residue_names[i], [])
        if not system.residues[i] in residue:
            residue.append(system.residues[i])
        residues[system.residue_names[i]] = residue
    if check_array(system.positions):
        raise TypeError
    with open(filename, "w") as fh:
        fh.write(
            (f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{A:7.2f}{B:7.2f}"
             f"{G:7.2f} P 1           1 \n"),
        )
        for i, _ in enumerate(system.names):
            residue = residues[system.residue_names[i]]
            resid = residue.index(system.residues[i]) + 1
            line = "HETATM"
            line += f"{i+1:5d}  "
            line += f"{system.names[i]:4s}"
            line += f"{system.residue_names[i]:4s}"
            line += f"{system.chains[i]}{resid:4d}    "
            line += f"{system.positions[i, 0]:8.3f}"
            line += f"{system.positions[i, 1]:8.3f}"
            line += f"{system.positions[i, 2]:8.3f}"
            line += "  1.00  0.00          "
            line += f"{system.elements[i]:2s}  \n"
            fh.write(line)
        fh.write("END")
