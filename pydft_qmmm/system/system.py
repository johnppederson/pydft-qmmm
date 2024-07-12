"""A container for all atom and residue data in a given system.
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
from pydft_qmmm.common import decompose
from pydft_qmmm.common import FileManager
from pydft_qmmm.common import interpret
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
    r"""A container for all atom and residue data in a given system.

    Args:
        atoms: The list of atoms that will comprise the system.
        box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of the box
            containing the system.

    Attributes:
        _positions: The positions (:math:`\mathrm{\mathring{A}}`) of the atoms
            within the system.
        _velocities: The velocities (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) of
            the atoms.
        _forces: The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            acting on the atoms.
        _masses: The masses (:math:`\mathrm{AMU}`) of the atoms.
        _charges: The partial charges (:math:`e`) of the atoms.
        _residues: The indices of residues to which the atoms belong.
        _box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of the box
            containing the system.
        _elements: The element symbols of the atoms.
        _names: The names (type) of the atoms, as in a PDB file.
        _residue_names: The names of the residues to which the atom
            belongs.
        _subsystems: The subsystems of which the atoms are a part.
    """
    _positions: ObservedArray[Any, array_float]
    _velocities: ObservedArray[Any, array_float]
    _forces: ObservedArray[Any, array_float]
    _masses: ObservedArray[Any, array_float]
    _charges: ObservedArray[Any, array_float]
    _residues: ObservedArray[Any, array_int]
    _box: ObservedArray[Any, array_float]
    _elements: ObservedArray[Any, array_str]
    _names: ObservedArray[Any, array_str]
    _residue_names: ObservedArray[Any, array_str]
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
        """Get the number of atoms in the system.

        Returns:
            The number of atoms in the system.
        """
        return len(self._atoms)

    @overload
    def __getitem__(self, key: int) -> _SystemAtom: ...
    @overload
    def __getitem__(self, key: slice) -> Sequence[_SystemAtom]: ...

    def __getitem__(
            self,
            key: int | slice,
    ) -> _SystemAtom | Sequence[_SystemAtom]:
        """Get the specified atom or slice of atoms.

        Returns:
            An atom or list of atoms.
        """
        return self._system_atoms[key]

    def __contains__(self, atom: object) -> bool:
        """Check if an atom is in the system.

        Args:
            atom: An atom object.

        Returns:
            A boolean specifying whether the atom is in the system
            or not.
        """
        if isinstance(atom, Atom):
            return atom in self._atoms
        elif isinstance(atom, _SystemAtom):
            return atom in self._system_atoms
        else:
            raise TypeError

    def __iter__(self) -> Iterator[_SystemAtom]:
        """Iterate over atoms in the system.

        Returns:
            An iterator for atoms in the system.
        """
        yield from self._system_atoms

    def __reversed__(self) -> Iterator[_SystemAtom]:
        """Iterate over atoms in the system in reversed order.

        Returns:
            An iterator for atoms in the system in reversed order.
        """
        yield from self._system_atoms[::-1]

    def index(self, atom: Any, start: int = 0, stop: int = -1) -> int:
        """Find the first index where the atom object is found.

        Args:
            atom: An atom object.
            start: The index at which to begin the search.
            stop: The index at which to conclude the search.
        """
        if isinstance(atom, Atom):
            return self._atoms.index(atom, start, stop)
        elif isinstance(atom, _SystemAtom):
            return self._system_atoms.index(atom, start, stop)
        else:
            raise TypeError

    def _aggregate(self, atoms: list[Atom]) -> list[_SystemAtom]:
        """Create an internal representation of the atoms of the system.

        Args:
            atoms: A list of atoms in the system.

        Returns:
            A list of atoms in the system which has been collated and
            used to generate observed arrays for system data.
        """
        n = len(atoms)
        # Initialize aggregate data containers.
        positions = np.zeros((n, 3), dtype=np.float64)
        velocities = np.zeros((n, 3), dtype=np.float64)
        forces = np.zeros((n, 3), dtype=np.float64)
        masses = np.zeros(n, dtype=np.float64)
        charges = np.zeros(n, dtype=np.float64)
        residues = np.zeros(n, dtype=np.int32)
        elements = np.zeros(n, dtype="<U2")
        names = np.zeros(n, dtype="<U4")
        residue_names = np.zeros(n, dtype="<U4")
        subsystems = np.zeros(n, dtype="O")
        # Populate aggregate data.
        for i, atom in enumerate(atoms):
            positions[i, :] = atom.position
            velocities[i, :] = atom.velocity
            forces[i, :] = atom.force
            masses[i] = atom.mass
            charges[i] = atom.charge
            residues[i] = atom.residue
            elements[i] = atom.element
            names[i] = atom.name
            residue_names[i] = atom.residue_name
            subsystems[i] = atom.subsystem
        # Populate basic private datatypes.
        self._positions = ObservedArray(positions)
        self._velocities = ObservedArray(velocities)
        self._forces = ObservedArray(forces)
        self._masses = ObservedArray(masses)
        self._charges = ObservedArray(charges)
        self._residues = ObservedArray(residues)
        self._elements = ObservedArray(elements)
        self._names = ObservedArray(names)
        self._residue_names = ObservedArray(residue_names)
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
                    ArrayValue(self._residues, i),
                    ArrayValue(self._elements, i),
                    ArrayValue(self._names, i),
                    ArrayValue(self._residue_names, i),
                    ArrayValue(self._subsystems, i),
                ),
            )
        return atom_list

    @staticmethod
    def load(
            pdb_file: list[str] | str,
            forcefield_file: list[str] | str | None = None,
    ) -> System:
        """Load a system from PDB and FF XML files.

        Args:
            pdb_file: The directory or list of directories containing
                PDB files with position, element, name, residue, residue
                name, and lattice vector data.
            forcefield_file: The directory or list of directories
                containing FF XML files with mass and charge data.

        Returns:
            The system generated from the data in the PDB and FF XML
            files.
        """
        fm = FileManager()
        system_info = fm.load(pdb_file, forcefield_file)
        box = np.array(system_info[-1])
        atom_info = zip(*system_info[0:-1])
        atoms = []
        for pos, mol, el, name, mol_name, mass, charge in atom_info:
            atoms.append(
                Atom(
                    position=np.array(pos),
                    residue=mol,
                    element=el,
                    name=name,
                    residue_name=mol_name,
                    mass=mass,
                    charge=charge,
                ),
            )
        return System(atoms, box)

    def select(self, query: str) -> frozenset[int]:
        """Convert a VMD-like selection query into a set of atom indices.

        Args:
            query: The VMD-like selection query.

        Returns:
            A set of indices of atoms in the system representing the
            query.
        """
        line = decompose(query)
        selection = interpret(line, self)
        return selection

    @property
    def positions(self) -> ObservedArray[
        Any,
        array_float,
    ]:
        r"""The positions (:math:`\mathrm{\mathring{A}}`) of the atoms."""
        return self._positions

    @positions.setter
    def positions(self, positions: NDArray[np.float64]) -> None:
        self._positions[:] = positions

    @property
    def velocities(self) -> ObservedArray[
        Any,
        array_float,
    ]:
        r"""The velocities (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) of the
        atoms.
        """
        return self._velocities

    @velocities.setter
    def velocities(self, velocities: NDArray[np.float64]) -> None:
        self._velocities[:] = velocities

    @property
    def forces(self) -> ObservedArray[Any, array_float]:
        r"""The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) on
        the atoms.
        """
        return self._forces

    @forces.setter
    def forces(self, forces: NDArray[np.float64]) -> None:
        self._forces[:] = forces

    @property
    def masses(self) -> ObservedArray[Any, array_float]:
        r"""The masses (:math:`\mathrm{AMU}`) of the atoms."""
        return self._masses

    @masses.setter
    def masses(self, masses: NDArray[np.float64]) -> None:
        self._masses[:] = masses

    @property
    def charges(self) -> ObservedArray[Any, array_float]:
        """The partial charges (:math:`e`) of the atoms."""
        return self._charges

    @charges.setter
    def charges(self, charges: NDArray[np.float64]) -> None:
        self._charges[:] = charges

    @property
    def residues(self) -> ObservedArray[Any, array_int]:
        """The indices of the residues to which atoms belong."""
        return self._residues

    @residues.setter
    def residues(self, residues: NDArray[np.int32]) -> None:
        self._residues[:] = residues

    @property
    def box(self) -> ObservedArray[Any, array_float]:
        r"""The lattice vectors (:math:`\mathrm{\mathring{A}}`) of the
        system.
        """
        return self._box

    @box.setter
    def box(self, box: NDArray[np.float64]) -> None:
        self._box[:] = box

    @property
    def elements(self) -> ObservedArray[Any, array_str]:
        """The element symbols of the atoms."""
        return self._elements

    @elements.setter
    def elements(self, elements: np.ndarray[Any, np.dtype[np.str_]]) -> None:
        self._elements[:] = elements

    @property
    def names(self) -> ObservedArray[Any, array_str]:
        """The names (type) of the atoms, as in a PDB file."""
        return self._names

    @names.setter
    def names(self, names: np.ndarray[Any, np.dtype[np.str_]]) -> None:
        self._names[:] = names

    @property
    def residue_names(
        self,
    ) -> ObservedArray[Any, array_str]:
        """The names of residues to which atoms belong."""
        return self._residue_names

    @residue_names.setter
    def residue_names(
            self,
            residue_names: np.ndarray[Any, np.dtype[np.str_]],
    ) -> None:
        self._residue_names[:] = residue_names

    @property
    def subsystems(self) -> ObservedArray[Any, array_obj]:
        """The subsystems of which atoms are a part."""
        return self._subsystems

    @subsystems.setter
    def subsystems(
            self,
            subsystems: np.ndarray[Any, np.dtype[np.object_]],
    ) -> None:
        self._subsystems[:] = subsystems

    @property
    def residue_map(self) -> dict[int, frozenset[int]]:
        """The set of atom indices corresponding to a residue index."""
        return {
            i: frozenset({j for j, m in enumerate(self._residues) if m == i})
            for i in set(self._residues)
        }

    @property
    def subsystem_map(self) -> dict[Subsystem, frozenset[int]]:
        """The set of atom indices corresponding to a subsystem."""
        return {
            i: frozenset({j for j, s in enumerate(self._subsystems) if s == i})
            for i in (Subsystem.I, Subsystem.II, Subsystem.III)
        }

    @property
    def qm_region(self) -> frozenset[int]:
        """The set of atom indices with membership in Subsystem I."""
        return frozenset(
            {
                j for j, s in enumerate(self._subsystems)
                if s == Subsystem.I
            },
        )

    @property
    def mm_region(self) -> frozenset[int]:
        """The set of atom indices with membership in Subsystems II & III."""
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
