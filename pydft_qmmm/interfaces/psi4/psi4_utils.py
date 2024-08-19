"""Functionality for building the Psi4 interface and some helper classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import psi4.core
from numpy.typing import NDArray

from .psi4_interface import Psi4Interface
from pydft_qmmm.common import BOHR_PER_ANGSTROM

if TYPE_CHECKING:
    from pydft_qmmm.interfaces import QMSettings


class Psi4Context:
    r"""A wrapper class for managing Psi4 Geometry object generation.

    Args:
        atoms: The indices of atoms that are treated at the QM level
            of theory.
        embedding: The indices of atoms that are electrostatically
            embedded.
        elements: The element symbols of the atoms in the system.
        positions: The positions (:math:`\mathrm{\mathring{A}}`) of the atoms
            within the system.
        charges: The partial charges (:math:`e`) of the atoms in the
            system.
        charge: The net charge (:math:`e`) of the system represented
            at the QM level of theory.
        spin: The net spin of the system represented by the QM
            level of theory.
    """

    def __init__(
            self,
            atoms: set[int],
            embedding: set[int],
            elements: list[str],
            positions: NDArray[np.float64],
            charges: NDArray[np.float64],
            charge: int,
            spin: int,
    ) -> None:
        self.atoms = atoms
        self.embedding = embedding
        self.elements = elements
        self.positions = positions
        self.charges = charges
        self.charge = charge
        self.spin = spin
        self.do_embedding = True

    @lru_cache
    def generate_molecule(self) -> psi4.core.Molecule:
        """Create the Geometry object for Psi4 calculations.

        Returns:
            The Psi4 Geometry object, which contains the positions,
            net charge, and net spin of atoms treated at the QM level
            of theory.
        """
        geometrystring = """\n"""
        for atom in sorted(self.atoms):
            position = self.positions[atom]
            element = self.elements[atom]
            geometrystring = (
                geometrystring
                + str(element) + " "
                + str(position[0]) + " "
                + str(position[1]) + " "
                + str(position[2]) + "\n"
            )
        geometrystring += str(self.charge) + " "
        geometrystring += str(self.spin) + "\n"
        # geometrystring += "symmetry c1\n"
        geometrystring += "noreorient\nnocom\n"
        return psi4.geometry(geometrystring)

    def generate_external_potential(self) -> list[list[int | list[int]]] | None:
        r"""Create the data structures read by Psi4 to perform embedding.

        Returns:
            The list of coordinates (:math:`\mathrm{a.u.}`) and charges
            (:math:`e`) that will be read by Psi4 during calculations
            and electrostatically embedded.
        """
        external_potential = []
        for i in sorted(self.embedding):
            external_potential.append(
                [
                    self.charges[i],
                    [
                        self.positions[i, 0] * BOHR_PER_ANGSTROM,
                        self.positions[i, 1] * BOHR_PER_ANGSTROM,
                        self.positions[i, 2] * BOHR_PER_ANGSTROM,
                    ],
                ],
            )
        if not external_potential:
            return None
        if not self.do_embedding:
            return None
        return external_potential

    def update_positions(self, positions: NDArray[np.float64]) -> None:
        r"""Set the atomic positions used by Psi4.

        Args:
            positions: The positions (:math:`\mathrm{\mathring{A}}`) of the
                atoms within the system.
        """
        self.positions = positions
        self.generate_molecule.cache_clear()

    def update_charges(self, charges: NDArray[np.float64]) -> None:
        """Set the atomic partial charges used by Psi4 for embedding.

        Args:
            charges: The partial charges (:math:`e`) of the atoms.
        """
        self.charges = charges
        self.generate_molecule.cache_clear()

    def update_embedding(self, embedding: set[int]) -> None:
        """Set the atoms are electrostatically embedded.

        Args:
            embedding: The indices of atoms that are electrostatically
                embedded.
        """
        self.embedding = embedding


@dataclass(frozen=True)
class Psi4Options:
    """An immutable wrapper class for storing Psi4 global options.

    Args:
        basis: The name of the basis set to be used by Psi4.
        dft_spherical_points: The number of spherical Lebedev points
            to use in the DFT quadrature.
        dft_radial_points: The number of radial points to use in the
            DFT quadrature.
        scf_type: The name of the type of SCF to perform, as in
            the JK build algorithms as in Psi4.
        scf__reference: The name of the reference to use, including
            restricted Kohn-Sham or unrestricted Kohn-Sham.
        scf__guess: The name of the algorithm used to generate the
            initial guess at the start of an SCF procedure.
    """
    basis: str
    dft_spherical_points: int
    dft_radial_points: int
    scf_type: str
    scf__reference: str
    scf__guess: str


def psi4_interface_factory(settings: QMSettings) -> Psi4Interface:
    """Build the interface to Psi4 given the settings.

    Args:
        settings: The settings used to build the Psi4 interface.

    Returns:
        The Psi4 interface.
    """
    basis = settings.basis_set
    if "assign" not in settings.basis_set:
        basis = "assign " + settings.basis_set.strip()
    psi4.basis_helper(basis, name="default")
    options = _build_options(settings)
    functional = settings.functional
    context = _build_context(settings)
    wrapper = Psi4Interface(
        settings, options, functional, context,
    )
    # Register observer functions.
    settings.system.charges.register_notifier(wrapper.update_charges)
    settings.system.positions.register_notifier(wrapper.update_positions)
    settings.system.subsystems.register_notifier(wrapper.update_subsystems)
    return wrapper


def _build_options(settings: QMSettings) -> Psi4Options:
    """Build the Psi4Options object.

    Args:
        settings: The settings used to build the Psi4 interface.

    Returns:
        The global options used by Psi4 in each calculation.
    """
    options = Psi4Options(
        "default",
        settings.quadrature_spherical,
        settings.quadrature_radial,
        settings.scf_type,
        "uks" if settings.spin > 1 else "rks",
        "read" if settings.read_guess else "auto",
    )
    return options


def _build_context(settings: QMSettings) -> Psi4Context:
    """Build the Psi4Context object.

    Args:
        settings: The settings used to build the Psi4 interface.

    Returns:
        The geometry and embedding manager for Psi4.
    """
    context = Psi4Context(
        set(settings.system.select("subsystem I")),
        set(),
        list(settings.system.elements),
        settings.system.positions,
        settings.system.charges,
        settings.charge,
        settings.spin,
    )
    return context
