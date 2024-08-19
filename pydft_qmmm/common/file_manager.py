"""A utility for reading and writing files.
"""
from __future__ import annotations

import array
import os
import struct
from typing import TYPE_CHECKING

import numpy as np

from .atom import Atom
from .constants import ELEMENT_TO_MASS
from .utils import compute_lattice_constants
from .utils import compute_lattice_vectors

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pydft_qmmm import System


_WORKING_DIRECTORY = os.getcwd() + "/"


def _parse_name(name: str, ext: str = "") -> str:
    """Check file name for correct directory path and extension.

    Args:
        name: The directory/name of the file.
        ext: The desired extension of the file.

    Returns:
        The full path of the file.
    """
    filename = ""
    if not name.startswith(_WORKING_DIRECTORY):
        filename = _WORKING_DIRECTORY
    filename += name
    if not name.endswith(ext):
        filename += "." + ext
    return filename


def _check_ext(filename: str, ext: str) -> None:
    """Ensure that a filename has the correct extension.

    Args:
        filename: The name of the file.
        ext: The desired extension of the file.
    """
    if not filename.endswith(ext):
        bad_ext = filename.split(".")[-1]
        raise ValueError(
            (
                f"Got extension'{bad_ext}' from file '{filename}'.  "
                + f"Expected extension '{ext}'"
            ),
        )


def _check_array(value: NDArray[np.float64], key: str = "positions") -> None:
    if np.isnan(value).any():
        raise ValueError(
            f"Array '{key}' contains NaN values.",
        )
    if np.isinf(value).any():
        raise ValueError(
            f"Array '{key}' contains Inf values.",
        )


def load_system(*args: str) -> tuple[list[Atom], NDArray[np.float64]]:
    """Load files necessary to generate a system.

    Args:
        *args: The directory or list of directories containing
            PDB files with position, element, name, residue, residue
            name, and lattice vector data.

    Returns:
        Data required to create a System object.
    """
    atoms = []
    boxes = []
    for filename in args:
        tmp_atoms, tmp_box = _read_pdb(filename)
        atoms.extend(tmp_atoms)
        boxes.append(tmp_box)
    if len(boxes) > 1:
        for i in range(len(boxes) - 1):
            if not np.allclose(boxes[i], boxes[i+1]):
                raise OSError(
                    (
                        "Multiple PDB files have been loaded with "
                        "different box vectors."
                    ),
                )
    return atoms, boxes[-1]


def _read_pdb(pdb_file: str) -> tuple[list[Atom], NDArray[np.float64]]:
    """Extract system data from a PDB file.

    Args:
        pdb_file: The directory or list of directories containing
            PDB files with position, element, name, residue, residue
            name, and lattice vector data.

    Returns:
        Data required to create a System object.
    """
    atoms: list[Atom] = []
    box = np.array(
        [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ],
    )
    with open(pdb_file) as fh:
        lines = fh.readlines()
    offset = 0
    for line in lines:
        if line.startswith("CRYST1"):
            a = float(line[6:15].strip())
            b = float(line[15:24].strip())
            c = float(line[24:33].strip())
            alpha = float(line[33:40].strip())
            beta = float(line[40:47].strip())
            gamma = float(line[47:54].strip())
            box = compute_lattice_vectors(
                a, b, c,
                alpha, beta, gamma,
            )
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # num = int(line[6:11].strip())
            name = line[12:16].strip()
            resname = line[17:21].strip()
            resnum = int(line[22:26])
            if len(atoms) == 0:
                offset = -resnum
            else:
                if (
                    atoms[-1].residue > resnum + offset
                    or atoms[-1].residue < resnum + offset
                    or atoms[-1].residue_name != resname
                ):
                    offset = atoms[-1].residue - resnum + 1
            position = np.array([
                float(line[30:38].strip()),
                float(line[38:46].strip()),
                float(line[46:54].strip()),
            ])
            element = line[76:78].strip()
            mass = ELEMENT_TO_MASS.get(element, 0.)
            atom = Atom(
                position,
                np.array([0., 0., 0.]),
                np.array([0., 0., 0.]),
                mass,
                0.,
                resnum + offset,
                element,
                name,
                resname,
            )
            atoms.append(atom)
    return atoms, box


def write_to_pdb(name: str, system: System) -> None:
    r"""Write system to PDB file.

    Args:
        name: The directory and filename to write the PDB file to.
        system: The system whose data will be written to a PDB file.

    .. note:: Based on PDB writer from OpenMM.
    """
    filename = _parse_name(name, ext="pdb")
    (
        len_a, len_b, len_c,
        alpha, beta, gamma,
    ) = compute_lattice_constants(system.box)
    residues: dict[str, list[int]] = {}
    for i in range(len(system)):
        residue = residues.get(system.residue_names[i], [])
        if not system.residues[i] in residue:
            residue.append(system.residues[i])
        residues[system.residue_names[i]] = residue
    _check_array(system.positions)
    with open(filename, "w") as fh:
        fh.write(
            (
                f"CRYST1{len_a:9.3f}{len_b:9.3f}{len_c:9.3f}"
                + f"{alpha:7.2f}{beta:7.2f}"
                + f"{gamma:7.2f} P 1           1 \n"
            ),
        )
        for i, _ in enumerate(system.names):
            residue = residues[system.residue_names[i]]
            resid = residue.index(system.residues[i]) + 1
            line = "HETATM"
            line += f"{i+1:5d}  "
            line += f"{system.names[i]:4s}"
            line += f"{system.residue_names[i]:4s}"
            line += f"A{resid:4d}    "
            line += f"{system.positions[i, 0]:8.3f}"
            line += f"{system.positions[i, 1]:8.3f}"
            line += f"{system.positions[i, 2]:8.3f}"
            line += "  1.00  0.00          "
            line += f"{system.elements[i]:2s}  \n"
            fh.write(line)
        fh.write("END")


def start_dcd(
        name: str,
        write_interval: int,
        num_particles: int,
        timestep: int | float,
) -> None:
    r"""Start writing to a DCD file.

    Args:
        name: The directory and filename to write the DCD file to.
        write_interval: The interval between successive DCD
            writes, in simulation steps.
        num_particles: The number of atoms in the system.
        timestep: The timestep (:math:`\mathrm{fs}`) used to perform
            simulation.

    .. note:: Based on DCD writer from OpenMM.
    """
    filename = _parse_name(name, ext="dcd")
    with open(filename, "wb") as fh:
        header = struct.pack(
            "<i4c9if", 84, b"C", b"O", b"R", b"D",
            0, 0, write_interval, 0, 0, 0, 0, 0, 0, timestep,
        )
        header += struct.pack(
            "<13i", 1, 0, 0, 0, 0, 0, 0, 0, 0, 24,
            84, 164, 2,
        )
        header += struct.pack("<80s", b"Created by PyDFT-QMMM")
        header += struct.pack("<80s", b"Created now")
        header += struct.pack("<4i", 164, 4, num_particles, 4)
        fh.write(header)


def write_to_dcd(
        name: str,
        write_interval: int,
        system: System,
        frame: int,
        offset: NDArray[np.float64] | None = None,
) -> None:
    r"""Write data to an existing DCD file.

    Args:
        name: The directory and filename to write the DCD file to.
        write_interval: The interval between successive DCD
            writes, in simulation steps.
        system: The system whose data will be written to a DCD file.
        frame: The current frame of the simulation.
        offset: A translation applied to positions before they are
            recorded.

    .. note:: Based on DCD writer from OpenMM.
    """
    filename = _parse_name(name, ext="dcd")
    (
        len_a, len_b, len_c,
        alpha, beta, gamma,
    ) = compute_lattice_constants(system.box)
    positions = system.positions.base.copy()
    if offset is not None:
        positions += offset
    _check_array(positions)
    with open(filename, "r+b") as fh:
        fh.seek(8, os.SEEK_SET)
        fh.write(struct.pack("<i", frame//write_interval))
        fh.seek(20, os.SEEK_SET)
        fh.write(struct.pack("<i", frame))
        fh.seek(0, os.SEEK_END)
        fh.write(
            struct.pack(
                "<i6di", 48, len_a, gamma, len_b, beta,
                alpha, len_c, 48,
            ),
        )
        num = struct.pack("<i", 4*len(system))
        for i in range(3):
            fh.write(num)
            coordinate = array.array(
                "f", (position[i] for position in positions),
            )
            coordinate.tofile(fh)
            fh.write(num)
        fh.flush()


def start_log(name: str) -> None:
    """Start writing to a log file.

    Args:
        name: The directory and filename to write the log file to.
    """
    filename = _parse_name(name, ext="log")
    with open(filename, "w") as fh:
        fh.write(f"{' PyDFT-QMMM Logger ':=^72}\n")


def write_to_log(name: str, lines: str, frame: int) -> None:
    """Write data to an existing log file.

    Args:
        name: The directory and filename to write the log file to.
        lines: A multi-line string which will be written to the
            log file.
        frame: The current frame of the simulation.
    """
    filename = _parse_name(name, ext="log")
    with open(filename, "a") as fh:
        fh.write(f"{' Frame ' + f'{frame:0>6}' + ' ':-^72}\n")
        fh.write(lines + "\n")
        fh.flush()


def end_log(name: str) -> None:
    """Terminate an existing log file.

    Args:
        name: The directory and filename to write the log file to.
    """
    filename = _parse_name(name, ext="log")
    with open(filename, "a") as fh:
        fh.write(f"{' End of Log ':=^72}")


def start_csv(name: str, header: str) -> None:
    """Start writing to a CSV file.

    Args:
        name: The directory and filename to write the CSV file to.
        header: The header for the CSV file.
    """
    filename = _parse_name(name, ext="csv")
    with open(filename, "w") as fh:
        fh.write(header + "\n")


def write_to_csv(
        name: str,
        line: str,
        header: str | None = None,
) -> None:
    """Write data to an existing CSV file.

    Args:
        name: The directory and filename to write the CSV file to.
        line: A string which will be written to the CSV file.
        header: The header for the CSV file.
    """
    filename = _parse_name(name, ext="csv")
    if header:
        with open(filename) as fh:
            lines = fh.readlines()
        with open(filename, "w") as fh:
            lines[0] = header + "\n"
            fh.writelines(lines)
    with open(filename, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
