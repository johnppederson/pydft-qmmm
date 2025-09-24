"""A container for all atom data in a given system.
"""
from __future__ import annotations

__all__ = ["System"]

from collections.abc import Sequence
from dataclasses import field
from functools import cached_property
from typing import overload
from typing import TYPE_CHECKING

import numpy as np

from .atom import _SystemAtom
from .variable import ArrayValue
from .variable import ObservedArray
from .variable import observed_class
from .variable import array_float
from .variable import array_int
from .variable import array_str
from .variable import array_obj
from .selection_utils import decompose
from .selection_utils import interpret
from .selection_utils import FAST_KEYWORDS
from .selection_utils import SLOW_KEYWORDS
from .file_manager import load_system
from pydft_qmmm.utils import system_cache

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any
    from collections.abc import Callable
    from numpy.typing import NDArray
    from .atom import Atom


def _del_attr(self: System, attr: str) -> Callable[[Any], None]:
    """Make a function that deletes a system attribute when called.

    Args:
        self: The system object.
        attr: The attribute to delete when the returned function is
            called.

    Returns:
        A function that deletes the given attribute from the system
        when called.
    """
    def wrapper(value: Any) -> None:
        if hasattr(self, attr):
            delattr(self, attr)
    return wrapper


@observed_class
class System(Sequence[_SystemAtom]):
    r"""A container for all atom and residue data in a given system.

    Args:
        atoms: The list of atoms that will comprise the system.
        box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of the
            box containing the system.
    """
    positions: ObservedArray[Any, array_float] = field(
        default_factory=lambda: ObservedArray(np.empty((0, 3), float)),
    )
    velocities: ObservedArray[Any, array_float] = field(
        default_factory=lambda: ObservedArray(np.empty((0, 3), float)),
    )
    forces: ObservedArray[Any, array_float] = field(
        default_factory=lambda: ObservedArray(np.empty((0, 3), float)),
    )
    masses: ObservedArray[Any, array_float] = field(
        default_factory=lambda: ObservedArray(np.empty(0, float)),
    )
    charges: ObservedArray[Any, array_float] = field(
        default_factory=lambda: ObservedArray(np.empty(0, float)),
    )
    residues: ObservedArray[Any, array_int] = field(
        default_factory=lambda: ObservedArray(np.empty(0, int)),
    )
    elements: ObservedArray[Any, array_str] = field(
        default_factory=lambda: ObservedArray(np.empty(0, str)),
    )
    names: ObservedArray[Any, array_str] = field(
        default_factory=lambda: ObservedArray(np.empty(0, str)),
    )
    residue_names: ObservedArray[Any, array_str] = field(
        default_factory=lambda: ObservedArray(np.empty(0, str)),
    )
    chains: ObservedArray[Any, array_str] = field(
        default_factory=lambda: ObservedArray(np.empty(0, str)),
    )
    subsystems: ObservedArray[Any, array_obj] = field(
        default_factory=lambda: ObservedArray(np.empty(0, object)),
    )
    box: ObservedArray[Any, array_float] = field(
        default_factory=lambda: ObservedArray(np.empty((0, 3), float)),
    )

    def __init__(
            self,
            atoms: list[Atom],
            box: NDArray[np.float64] = np.zeros((3, 3)),
    ) -> None:
        for name, field_ in getattr(self, "__dataclass_fields__").items():
            setattr(self, "_" + name, field_.default_factory())
        self._setup(atoms)
        self._box: ObservedArray[Any, array_float] = ObservedArray(box)
        # Delete residue_map cached property if residues changes.
        self.residues.register_notifier(_del_attr(self, "residue_map"))

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
    @overload
    def __getitem__(self, key: str) -> Sequence[_SystemAtom]: ...

    def __getitem__(
            self,
            key: int | slice | str,
    ) -> _SystemAtom | Sequence[_SystemAtom]:
        """Get the specified atom or slice of atoms.

        Returns:
            An atom or list of atoms.
        """
        if isinstance(key, str):
            return [self._system_atoms[i] for i in sorted(self.select(key))]
        return self._system_atoms[key]

    def __contains__(self, atom: object) -> bool:
        """Check if an atom is in the system.

        Args:
            atom: An atom object.

        Returns:
            Whether the atom is in the system or not.
        """
        if isinstance(atom, Atom):
            return atom in self._atoms
        elif isinstance(atom, _SystemAtom):
            return atom in self._system_atoms
        return False

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

    def _setup(self, atoms: list[Atom]) -> None:
        """Create an internal representation of the atoms of the system.

        Args:
            atoms: A list of atoms in the system.
        """
        # Populate ObservedArray objects.
        for name in getattr(self, "__dataclass_fields__"):
            if name == "box":
                continue
            temp = getattr(self, name)
            for atom in atoms:
                atom_value = getattr(
                    atom,
                    (
                        "velocity" if name == "velocities"
                        else "mass" if name == "masses" else name[:-1]
                    ),
                )
                temp = np.concatenate((temp, np.array([atom_value])))
            setattr(self, "_" + name, ObservedArray(temp))
        # Generate _SystemAtom objects.
        system_atoms = []
        for i, atom in enumerate(atoms):
            kwargs: dict[str, Any] = dict()
            for name in getattr(self, "__dataclass_fields__"):
                if name == "box":
                    continue
                data = getattr(self, name)
                name = (
                    "velocity" if name == "velocities"
                    else "mass" if name == "masses" else name[:-1]
                )
                if isinstance(data[i], np.ndarray):
                    kwargs[name] = data[i]
                else:
                    kwargs[name] = ArrayValue(data, i)
            system_atoms.append(_SystemAtom(**kwargs))
        self._atoms = atoms
        self._system_atoms = system_atoms

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

    def count(self, atom: Any) -> int:
        """Count how many times an atom object is in the system.

        Args:
            atom: An atom object.
        """
        if isinstance(atom, Atom):
            return self._atoms.count(atom)
        elif isinstance(atom, _SystemAtom):
            return self._system_atoms.count(atom)
        else:
            raise TypeError

    @staticmethod
    def load(*args: str) -> System:
        """Load a system from PDB files.

        Args:
            args: The PDB file or list of PDB files with position,
                element, name, residue, residue, name, and lattice
                vector data.

        Returns:
            The system generated from the data in the PDB files.
        """
        atoms, box = load_system(*args)
        return System(atoms, box)

    def select(self, query: str) -> frozenset[int]:
        """Convert a VMD-like selection query into a set of atom indices.

        Args:
            query: The VMD-like selection query.

        Returns:
            A set of indices of atoms in the system referenced by the
            query.
        """
        if any(x in query.lower() for x in FAST_KEYWORDS.keys() | "within"):
            return self._fast_select(query)
        return self._slow_select(query)

    def _fast_select(self, query: str) -> frozenset[int]:
        """Convert a VMD-like selection query into a set of atom indices.

        Args:
            query: The VMD-like selection query.

        Returns:
            A set of indices of atoms in the system referenced by the
            query.
        """
        line = decompose(query)
        selection = interpret(line, self)
        return selection

    @system_cache(*(x[0] for x in SLOW_KEYWORDS.values()), obj_is_system=True)
    def _slow_select(self, query: str) -> frozenset[int]:
        """Convert a VMD-like selection query into a set of atom indices.

        Args:
            query: The VMD-like selection query.

        Returns:
            A set of indices of atoms in the system referenced by the
            query.
        """
        line = decompose(query)
        selection = interpret(line, self)
        return selection

    @cached_property
    def residue_map(self) -> dict[int, frozenset[int]]:
        """The set of atom indices corresponding to a residue index."""
        residue_map: dict[int, frozenset[int]] = {}
        for i, residue in enumerate(self.residues):
            resid = residue_map.get(residue, frozenset())
            residue_map[residue] = resid | {i}
        return residue_map
