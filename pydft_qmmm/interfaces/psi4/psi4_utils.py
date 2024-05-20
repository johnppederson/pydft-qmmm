#! /usr/bin/env python3
"""A module to define the :class:`Psi4Interface` class.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import psi4.core
from numpy.typing import NDArray

from .psi4_interface import Psi4Interface
from pydft_qmmm.common.units import BOHR_PER_ANGSTROM

if TYPE_CHECKING:
    from pydft_qmmm.interfaces import QMSettings


class Psi4Context:
    """A wrapper class for managing Psi4 Geometry object generation.

    :param atoms: |qm_atoms|
    :param embedding: |ae_atoms|
    :param elements: |elements|
    :param positions: |positions|
    :param charge: |charge|
    :param spin: |spin|
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

        :return: The Psi4 Geometry object.
        """
        geometrystring = """\n"""
        for atom in self.atoms:
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
        geometrystring += "symmetry c1\n"
        geometrystring += "noreorient\nnocom\n"
        return psi4.geometry(geometrystring)

    def generate_external_potential(self) -> list[list[int | list[int]]] | None:
        """Create the Geometry object for Psi4 calculations.

        :return: The Psi4 Geometry object.
        """
        external_potential = []
        for i in self.embedding:
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
        """Update the atom positions for Psi4.

        :param positions: |positions|
        """
        self.positions = positions
        self.generate_molecule.cache_clear()

    def update_charges(self, charges: NDArray[np.float64]) -> None:
        """Update the atom positions for Psi4.

        :param positions: |positions|
        """
        self.charges = charges
        self.generate_molecule.cache_clear()

    def update_embedding(self, embedding: set[int]) -> None:
        """Update the analytic embedding atoms for Psi4.

        :param embedding: |ae_atoms|
        """
        self.embedding = embedding


@dataclass(frozen=True)
class Psi4Options:
    """An immutable wrapper class for storing Psi4 global options.

    :param basis: |basis_set|
    :param dft_spherical_points: |quadrature_spherical|
    :param dft_radial_points: |quadrature_radial|
    :param scf_type: |scf_type|
    :param scf__reference: The restricted or unrestricted Kohn-Sham SCF.
    :param scf__guess: The type of guess to use for the Psi4.
        calculation.
    """
    basis: str
    dft_spherical_points: int
    dft_radial_points: int
    scf_type: str
    scf__reference: str
    scf__guess: str


def psi4_interface_factory(settings: QMSettings) -> Psi4Interface:
    """A function which constructs the :class:`Psi4Interface` for a QM
    system.

    :param settings: The :class:`QMSettings` object to build the
        QM system interface from.
    :return: The :class:`Psi4Interface` for the QM system.
    """
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
    """Build the :class:`Psi4Options` object.

    :param settings: The :class:`QMSettings` object to build from.
    :return: The :class:`Psi4Options` object built from the given
        settings.
    """
    options = Psi4Options(
        settings.basis_set,
        settings.quadrature_spherical,
        settings.quadrature_radial,
        settings.scf_type,
        "uks" if settings.spin > 1 else "rks",
        "read" if settings.read_guess else "auto",
    )
    return options


def _build_context(settings: QMSettings) -> Psi4Context:
    """Build the :class:`Psi4Context` object.

    :param settings: The :class:`QMSettings` object to build from.
    :return: The :class:`Psi4Context` object built from the given
        settings.
    """
    context = Psi4Context(
        set(settings.system.qm_region),
        set(),
        list(settings.system.elements),
        settings.system.positions,
        settings.system.charges,
        settings.charge,
        settings.spin,
    )
    return context
