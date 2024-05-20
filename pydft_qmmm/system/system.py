#! /usr/bin/env python3
"""A module defining the :class:`System` class.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import overload
from typing import TYPE_CHECKING

import numpy as np

from .atom import _SystemAtom
from .atom import Atom
from .variable import ArrayValue
from .variable import ObservedArray
from pydft_qmmm.common import FileManager
from pydft_qmmm.common import Subsystem


if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any
    from numpy.typing import NDArray
    from .variable import array_float
    from .variable import array_int
    from .variable import array_str
    from .variable import array_obj


class System(Sequence[_SystemAtom]):
    """An wrapper class designed to generate and hold :class:`State`
    and :class:`Topology` record objects.

    :param pdb_list: |pdb_list|
    :param topology_list: |topology_list|
    :param forcefield_list: |forcefield_list|
    """
    _positions: ObservedArray[Any, array_float]
    _velocities: ObservedArray[Any, array_float]
    _forces: ObservedArray[Any, array_float]
    _masses: ObservedArray[Any, array_float]
    _charges: ObservedArray[Any, array_float]
    _molecules: ObservedArray[Any, array_int]
    _box: ObservedArray[Any, array_float]
    _elements: ObservedArray[Any, array_str]
    _names: ObservedArray[Any, array_str]
    _molecule_names: ObservedArray[Any, array_str]
    _subsystems: ObservedArray[Any, array_obj]

    def __init__(
            self,
            atoms: list[Atom],
            box: NDArray[np.float64] = np.zeros((3, 3)),
    ) -> None:
        self._atoms = atoms
        self._system_atoms = self._aggregate(atoms)
        self._box = ObservedArray(box)

    def __len__(self) -> int:
        return len(self._atoms)

    @overload
    def __getitem__(self, key: int) -> _SystemAtom: ...
    @overload
    def __getitem__(self, key: slice) -> Sequence[_SystemAtom]: ...

    def __getitem__(
            self,
            key: int | slice,
    ) -> _SystemAtom | Sequence[_SystemAtom]:
        return self._system_atoms[key]

    def __contains__(self, atom: object) -> bool:
        if isinstance(atom, Atom):
            return atom in self._atoms
        elif isinstance(atom, _SystemAtom):
            return atom in self._system_atoms
        else:
            raise TypeError

    def __iter__(self) -> Iterator[_SystemAtom]:
        yield from self._system_atoms

    def __reversed__(self) -> Iterator[_SystemAtom]:
        yield from self._system_atoms[::-1]

    def index(self, atom: Any, start: int = 0, stop: int = -1) -> int:
        if isinstance(atom, Atom):
            return self._atoms.index(atom, start, stop)
        elif isinstance(atom, _SystemAtom):
            return self._system_atoms.index(atom, start, stop)
        else:
            raise TypeError

    def _aggregate(self, atoms: list[Atom]) -> list[_SystemAtom]:
        n = len(atoms)
        # Initialize aggregate data containers.
        positions = np.zeros((n, 3), dtype=np.float64)
        velocities = np.zeros((n, 3), dtype=np.float64)
        forces = np.zeros((n, 3), dtype=np.float64)
        masses = np.zeros(n, dtype=np.float64)
        charges = np.zeros(n, dtype=np.float64)
        molecules = np.zeros(n, dtype=np.int32)
        elements = np.zeros(n, dtype="<U2")
        names = np.zeros(n, dtype="<U4")
        molecule_names = np.zeros(n, dtype="<U4")
        subsystems = np.zeros(n, dtype="O")
        # Populate aggregate data.
        for i, atom in enumerate(atoms):
            positions[i, :] = atom.position
            velocities[i, :] = atom.velocity
            forces[i, :] = atom.force
            masses[i] = atom.mass
            charges[i] = atom.charge
            molecules[i] = atom.molecule
            elements[i] = atom.element
            names[i] = atom.name
            molecule_names[i] = atom.molecule_name
            subsystems[i] = atom.subsystem
        # Populate basic private datatypes.
        self._positions = ObservedArray(positions)
        self._velocities = ObservedArray(velocities)
        self._forces = ObservedArray(forces)
        self._masses = ObservedArray(masses)
        self._charges = ObservedArray(charges)
        self._molecules = ObservedArray(molecules)
        self._elements = ObservedArray(elements)
        self._names = ObservedArray(names)
        self._molecule_names = ObservedArray(molecule_names)
        self._subsystems = ObservedArray(subsystems)
        # Generate _SystemAtom objects.
        atom_list = []
        for i in range(n):
            atom_list.append(
                _SystemAtom(
                    self._positions[i],
                    self._velocities[i],
                    self._forces[i],
                    ArrayValue(self._masses, i),
                    ArrayValue(self._charges, i),
                    ArrayValue(self._molecules, i),
                    ArrayValue(self._elements, i),
                    ArrayValue(self._names, i),
                    ArrayValue(self._molecule_names, i),
                    ArrayValue(self._subsystems, i),
                ),
            )
        return atom_list

    @staticmethod
    def load(
            pdb_file: list[str] | str,
            forcefield_file: list[str] | str | None = None,
    ) -> System:
        fm = FileManager()
        system_info = fm.load(pdb_file, forcefield_file)
        box = np.array(system_info[-1])
        atom_info = zip(*system_info[0:-1])
        atoms = []
        for pos, mol, el, name, mol_name, mass, charge in atom_info:
            atoms.append(
                Atom(
                    position=np.array(pos),
                    molecule=mol,
                    element=el,
                    name=name,
                    molecule_name=mol_name,
                    mass=mass,
                    charge=charge,
                ),
            )
        return System(atoms, box)

    @property
    def positions(self) -> ObservedArray[
        Any,
        array_float,
    ]: return self._positions

    @positions.setter
    def positions(self, positions: NDArray[np.float64]) -> None:
        self._positions[:] = positions

    @property
    def velocities(self) -> ObservedArray[
        Any,
        array_float,
    ]: return self._velocities

    @velocities.setter
    def velocities(self, velocities: NDArray[np.float64]) -> None:
        self._velocities[:] = velocities

    @property
    def forces(self) -> ObservedArray[Any, array_float]: return self._forces

    @forces.setter
    def forces(self, forces: NDArray[np.float64]) -> None:
        self._forces[:] = forces

    @property
    def masses(self) -> ObservedArray[Any, array_float]: return self._masses

    @masses.setter
    def masses(self, masses: NDArray[np.float64]) -> None:
        self._masses[:] = masses

    @property
    def charges(self) -> ObservedArray[Any, array_float]: return self._charges

    @charges.setter
    def charges(self, charges: NDArray[np.float64]) -> None:
        self._charges[:] = charges

    @property
    def molecules(self) -> ObservedArray[Any, array_int]: return self._molecules

    @molecules.setter
    def molecules(self, molecules: NDArray[np.int32]) -> None:
        self._molecules[:] = molecules

    @property
    def box(self) -> ObservedArray[Any, array_float]: return self._box

    @box.setter
    def box(self, box: NDArray[np.float64]) -> None:
        self._box[:] = box

    @property
    def elements(self) -> ObservedArray[Any, array_str]: return self._elements

    @elements.setter
    def elements(self, elements: np.ndarray[Any, np.dtype[np.str_]]) -> None:
        self._elements[:] = elements

    @property
    def names(self) -> ObservedArray[Any, array_str]: return self._names

    @names.setter
    def names(self, names: np.ndarray[Any, np.dtype[np.str_]]) -> None:
        self._names[:] = names

    @property
    def molecule_names(
        self,
    ) -> ObservedArray[Any, array_str]: return self._molecule_names

    @molecule_names.setter
    def molecule_names(
            self,
            molecule_names: np.ndarray[Any, np.dtype[np.str_]],
    ) -> None:
        self._molecule_names[:] = molecule_names

    @property
    def subsystems(self) -> ObservedArray[Any, array_obj]:
        return self._subsystems

    @subsystems.setter
    def subsystems(
            self,
            subsystems: np.ndarray[Any, np.dtype[np.object_]],
    ) -> None:
        self._subsystems[:] = subsystems

    @property
    def molecule_map(self) -> dict[int, frozenset[int]]:
        return {
            i: frozenset({j for j, m in enumerate(self._molecules) if m == i})
            for i in set(self._molecules)
        }

    @property
    def subsystem_map(self) -> dict[Subsystem, frozenset[int]]:
        return {
            i: frozenset({j for j, s in enumerate(self._subsystems) if s == i})
            for i in (Subsystem.I, Subsystem.II, Subsystem.III)
        }

    @property
    def qm_region(self) -> frozenset[int]:
        return frozenset(
            {
                j for j, s in enumerate(self._subsystems)
                if s == Subsystem.I
            },
        )

    @property
    def mm_region(self) -> frozenset[int]:
        return frozenset(
            {
                j for j, s in enumerate(self._subsystems)
                if s in (Subsystem.II, Subsystem.III)
            },
        )

    # def count(self, atom: Atom) -> int:

    # def append(self, atom: Atom) -> None:
    #     super().append(atom)
    #     atom_list = self._aggregate(self.data)
    #     self._data = atom_list

    # def extend(self, atoms: list[Atom]) -> None:
    #     super().extend(atoms)
    #     atom_list = self._aggregate(self.data)
    #     self._data = atom_list

    # def insert(self, index: int, atom: Atom) -> None:
    #     super().insert(index, atom)
    #     atom_list = self._aggregate(self.data)
    #     self._data = atom_list

    # def clear(self) -> None:
    #     super().clear()
    #     atom_list = self._aggregate(self.data)
    #     self._data = atom_list

    # def remove(self, atom: Atom) -> None:
    #     super().remove(atom)
    #     atom_list = self._aggregate(self.data)
    #     self._data = atom_list

    # def reverse(self) -> None:
    #     super().reverse()
    #     atom_list = self._aggregate(self.data)
    #     self._data = atom_list

    # def sort(self, reverse: bool):  # , key: Callable?
    #     raise NotImplementedError("Cannot sort atoms")
