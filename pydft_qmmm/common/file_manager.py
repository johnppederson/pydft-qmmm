#! /usr/bin/env python3
"""A module defining the class and functions needed to load and write
files.
"""
from __future__ import annotations

import array
import os
import struct
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import numpy as np

from .utils import compute_lattice_constants

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray
    from pydft_qmmm.system import ObservedArray
    from pydft_qmmm.system import array_float
    from pydft_qmmm.system import array_int
    from pydft_qmmm.system import array_str


class FileManager:
    """A class to load and generate inputs and outputs.

    :param working_directory: The current working directory.
    """

    def __init__(self, working_directory: str = "./") -> None:
        self._working_directory = working_directory
        if not os.path.isdir(working_directory):
            os.makedirs(working_directory)

    def load(
            self,
            pdb_file: list[str] | str,
            forcefield_file: list[str] | str | None = None,
    ) -> tuple[
        list[list[float]],
        list[int],
        list[str],
        list[str],
        list[str],
        list[float],
        list[float],
        list[list[float]],
    ]:
        """Load files necessary to generate a system.

        :param pdb_list: |pdb_list|
        :param forcefield_list: |forcefield_list|
        :return: Data for the :class:`State` and :class:`Topology`
            record classes.
        """
        # Check the file extensions and add them to the :class:`Files`
        # record.
        if isinstance(pdb_file, str):
            _check_ext(pdb_file, "pdb")
            pdb_list = [pdb_file]
        elif isinstance(pdb_file, list):
            for fh in pdb_file:
                _check_ext(fh, "pdb")
            pdb_list = pdb_file
        else:
            raise TypeError
        if isinstance(forcefield_file, str):
            _check_ext(forcefield_file, "xml")
            forcefield_list = [forcefield_file]
        elif isinstance(forcefield_file, list):
            for fh in forcefield_file:
                _check_ext(fh, "xml")
            forcefield_list = forcefield_file
        else:
            raise TypeError
        # Generate and return :class:`System` data using OpenMM.
        system_info = _get_atom_data(
            pdb_list,
            forcefield_list,
        )
        return system_info

    def write_to_pdb(
            self,
            name: str,
            positions: NDArray[np.float64] | ObservedArray[Any, array_float],
            box: NDArray[np.float64] | ObservedArray[Any, array_float],
            molecules: list[int] | ObservedArray[Any, array_int],
            molecule_names: list[str] | ObservedArray[Any, array_str],
            elements: list[str] | ObservedArray[Any, array_str],
            names: list[str] | ObservedArray[Any, array_str],
    ) -> None:
        """Utility to write PDB files with :class:`State` current
        coordinates.

        .. note:: Based on PDB writer from OpenMM.
        """
        filename = self._parse_name(name, ext="pdb")
        (
            len_a, len_b, len_c,
            alpha, beta, gamma,
        ) = compute_lattice_constants(box)
        with open(filename, "w") as fh:
            fh.write(
                (
                    f"CRYST1{len_a:9.3f}{len_b:9.3f}{len_c:9.3f}"
                    + f"{alpha:7.2f}{beta:7.2f}"
                    + f"{gamma:7.2f} P 1           1 \n"
                ),
            )
            for i, atom in enumerate(names):
                line = "HETATM"
                line += f"{i+1:5d}  "
                line += f"{name:4s}"
                line += f"{molecule_names[i]:4s}"
                line += f"A{molecules[i]+1:4d}    "
                line += f"{positions[i,0]:8.3f}"
                line += f"{positions[i,1]:8.3f}"
                line += f"{positions[i,2]:8.3f}"
                line += "  1.00  0.00           "
                line += f"{elements[i]:2s}  \n"
                fh.write(line)
            fh.write("END")

    def start_dcd(
            self,
            name: str,
            write_interval: int,
            num_particles: int,
            timestep: int | float,
    ) -> None:
        """Utility to start writing a DCD file.

        :param name: The directory/name of DCD file to be written.
        :param write_interval: The interval between successive writes
            to the DCD file.
        :param num_particles: The number of particles in the
            :class:`System`.
        :param timestep: |timestep|

        .. note:: Based on DCD writer from OpenMM.
        """
        filename = self._parse_name(name, ext="dcd")
        with open(filename, "wb") as fh:
            header = struct.pack(
                "<i4c9if", 84, b"C", b"O", b"R", b"D",
                0, 0, write_interval, 0, 0, 0, 0, 0, 0, timestep,
            )
            header += struct.pack(
                "<13i", 1, 0, 0, 0, 0, 0, 0, 0, 0, 24,
                84, 164, 2,
            )
            header += struct.pack("<80s", b"Created by QM/MM/PME")
            header += struct.pack("<80s", b"Created now")
            header += struct.pack("<4i", 164, 4, num_particles, 4)
            fh.write(header)

    def write_to_dcd(
            self,
            name: str,
            write_interval: int,
            num_particles: int,
            positions: NDArray[np.float64] | ObservedArray[Any, array_float],
            box: NDArray[np.float64] | ObservedArray[Any, array_float],
            frame: int,
    ) -> None:
        """Write data to an existing DCD file.

        :param name: The directory/name of DCD file to be written.
        :param write_interval: The interval between successive writes
            to the DCD file.
        :param num_particles: The number of particles in the
            :class:`System`.
        :param positions: |positions|
        :param box: |box|
        :param frame: |frame|

        .. note:: Based on DCD writer from OpenMM.
        """
        filename = self._parse_name(name, ext="dcd")
        (
            len_a, len_b, len_c,
            alpha, beta, gamma,
        ) = compute_lattice_constants(box)
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
            num = struct.pack("<i", 4*num_particles)
            for i in range(3):
                fh.write(num)
                coordinate = array.array(
                    "f", (position[i] for position in positions),
                )
                coordinate.tofile(fh)
                fh.write(num)
            fh.flush()

    def start_log(
            self,
            name: str,
    ) -> None:
        """Utility to start writing a log file.

        :param name: The directory/name of log file to be written.
        """
        filename = self._parse_name(name, ext="log")
        with open(filename, "w") as fh:
            fh.write(f"{' QM/MM/PME Logger ':=^72}\n")

    def write_to_log(
            self,
            name: str,
            lines: str,
            frame: int,
    ) -> None:
        """Write data to an existing log file.

        :param name: The directory/name of log file to be written.
        :param lines: The lines to be written to the log file.
        :param frame: |frame|
        """
        filename = self._parse_name(name, ext="log")
        with open(filename, "a") as fh:
            fh.write(f"{' Frame ' + f'{frame:0>6}' + ' ':-^72}\n")
            fh.write(lines + "\n")
            fh.flush()

    def end_log(
            self,
            name: str,
    ) -> None:
        """Terminate an existing log file.

        :param name: The directory/name of log file to be terminated.
        """
        filename = self._parse_name(name, ext="log")
        with open(filename, "a") as fh:
            fh.write(f"{' End of Log ':=^72}")

    def start_csv(
            self,
            name: str,
            header: str,
    ) -> None:
        """Utility to start writing a CSV file.

        :param name: The directory/name of CSV file to be written.
        :param header: The header for the CSV file.
        """
        filename = self._parse_name(name, ext="csv")
        with open(filename, "w") as fh:
            fh.write(header + "\n")

    def write_to_csv(
            self,
            name: str,
            line: str,
            header: str | None = None,
    ) -> None:
        """Write data to an existing CSV file.

        :param name: The directory/name of CSV file to be written.
        :param lines: The lines to be written to the CSV file.
        """
        filename = self._parse_name(name, ext="csv")
        if header:
            with open(filename) as fh:
                lines = fh.readlines()
            with open(filename, "w") as fh:
                lines[0] = header + "\n"
                fh.writelines(lines)
        with open(filename, "a") as fh:
            fh.write(line + "\n")
            fh.flush()

    def _parse_name(self, name: str, ext: str = "") -> str:
        """Ensure that the given name has the correct directory path
        and extension.

        :param name: The directory/name of the file to be written.
        :param ext: The desired extension of the file to be written.
        """
        filename = ""
        if not name.startswith(self._working_directory):
            filename = self._working_directory
        filename += name
        if not name.endswith(ext):
            filename += "." + ext
        return filename


def _check_ext(filename: str, ext: str) -> None:
    """Ensure that the given filename has the correct extension.

    :param filename: The name of the file.
    :param ext: The desired extension.
    """
    if not filename.endswith(ext):
        bad_ext = filename.split(".")[-1]
        raise ValueError(
            (
                f"Got extension'{bad_ext}' from file '{filename}'.  "
                + f"Expected extension '{ext}'"
            ),
        )


def _get_atom_data(
        pdb_list: list[str],
        forcefield_list: list[str] | None = None,
) -> tuple[
    list[list[float]],
    list[int],
    list[str],
    list[str],
    list[str],
    list[float],
    list[float],
    list[list[float]],
]:
    """Extract :class:`State` and :class:`Topology` data from PDB and
    XML files using OpenMM.

    :param pdb_list: |pdb_list|
    :param topology_list: |topology_list|
    :param forcefield_list: |forcefield_list|
    :return: Data for the :class:`State` and :class:`Topology`
        record classes.
    """
    positions = []
    molecules: list[int] = []
    elements = []
    names = []
    molecule_names = []
    box = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    for pdb in pdb_list:
        with open(pdb) as fh:
            lines = fh.readlines()
        offset = -1
        for line in lines:
            if line.startswith("CRYST1"):
                box[0][0] = float(line[6:15].strip())
                box[1][1] = float(line[15:24].strip())
                box[2][2] = float(line[24:33].strip())
            if line.startswith("ATOM") or line.startswith("HETATM"):
                names.append(line[12:16].strip())
                molecule_names.append(line[17:21].strip())
                number = int(line[22:26].strip()) + offset
                if len(molecules) > 0:
                    if number < molecules[-1]:
                        offset += molecules[-1] - number + 1
                        number += molecules[-1] - number + 1
                molecules.append(number)
                positions.append(
                    [
                        float(line[30:38].strip()),
                        float(line[38:46].strip()),
                        float(line[46:54].strip()),
                    ],
                )
                elements.append(line[76:78].strip())
    masses = [0.]*len(names)
    charges = [0.]*len(names)
    if forcefield_list is None:
        forcefield_list = []
    for xml in forcefield_list:
        tree = ET.parse(xml)
        root = tree.getroot()
        for i, name in enumerate(names):
            residues = root.find("Residues")
            if not isinstance(residues, ET.Element):
                raise OSError
            atom = residues.find(f".//Atom[@name='{name}']")
            if not isinstance(atom, ET.Element):
                raise OSError
            type_ = atom.attrib["type"]
            atoms = root.find("AtomTypes")
            if not isinstance(atoms, ET.Element):
                raise OSError
            atom = atoms.find(f".//Type[@name='{type_}']")
            if isinstance(atom, ET.Element):
                masses[i] = float(atom.attrib["mass"])
            nonbonded = root.find("NonbondedForce")
            if not isinstance(nonbonded, ET.Element):
                raise OSError
            atom = nonbonded.find(f".//Atom[@type='{type_}']")
            if isinstance(atom, ET.Element):
                charges[i] = float(atom.attrib["charge"])
    return (
        positions, molecules, elements, names,
        molecule_names, masses, charges, box,
    )
