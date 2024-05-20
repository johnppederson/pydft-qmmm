#! /usr/bin/env python3
"""A module defining the :class:`QMMMHamiltonian` class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .hamiltonian import CouplingHamiltonian
from pydft_qmmm.calculators import InterfaceCalculator
from pydft_qmmm.common import Subsystem
from pydft_qmmm.common import TheoryLevel
from pydft_qmmm.interfaces import MMInterface
from pydft_qmmm.interfaces import QMInterface


if TYPE_CHECKING:
    from pydft_qmmm import System
    from pydft_qmmm.calculators import Calculator


_DEFAULT_FORCE_MATRIX = {
    Subsystem.I: {
        Subsystem.I: TheoryLevel.QM,
        Subsystem.II: TheoryLevel.NO,
        Subsystem.III: TheoryLevel.NO,
    },
    Subsystem.II: {
        Subsystem.I: TheoryLevel.NO,
        Subsystem.II: TheoryLevel.MM,
        Subsystem.III: TheoryLevel.MM,
    },
    Subsystem.III: {
        Subsystem.I: TheoryLevel.NO,
        Subsystem.II: TheoryLevel.MM,
        Subsystem.III: TheoryLevel.MM,
    },
}


_CLOSE_EMBEDDING = {
    "mechanical": (TheoryLevel.MM, TheoryLevel.MM),
    "electrostatic": (TheoryLevel.QM, TheoryLevel.QM),
    "none": (TheoryLevel.NO, TheoryLevel.NO),
}


_LONG_EMBEDDING = {
    "mechanical": (TheoryLevel.MM, TheoryLevel.MM),
    "electrostatic": (TheoryLevel.QM, TheoryLevel.MM),
    "cutoff": (TheoryLevel.NO, TheoryLevel.MM),
    "none": (TheoryLevel.NO, TheoryLevel.NO),
}


_SUPPORTED_EMBEDDING = [
    ("none", "none"),
    ("mechanical", "none"),
    ("mechanical", "cutoff"),
    ("mechanical", "mechanical"),
    ("electrostatic", "none"),
    ("electrostatic", "cutoff"),
    ("electrostatic", "mechanical"),
    ("electrostatic", "electrostatic"),
]


@dataclass
class QMMMHamiltonian(CouplingHamiltonian):
    """A wrapper class storing settings for QMMM calculations.

    :param qm_hamiltonian: |hamiltonian| for calculations on the QM
        subsystem.
    :param mm_hamiltonian: |hamiltonian| for calculations on the MM
        subsystem.
    :param embedding_cutoff: |embedding_cutoff|
    """
    close_range: str = "electrostatic"
    long_range: str = "cutoff"
    embedding_cutoff: float | int = 14.

    def __post_init__(self) -> None:
        if (self.close_range, self.long_range) not in _SUPPORTED_EMBEDDING:
            raise TypeError("...")
        self.force_matrix = _DEFAULT_FORCE_MATRIX.copy()
        # Adjust I-II interaction.
        I_II, II_I = _CLOSE_EMBEDDING[self.close_range]
        self.force_matrix[Subsystem.I][Subsystem.II] = I_II
        self.force_matrix[Subsystem.II][Subsystem.I] = II_I
        # Adjust I-III interaction.
        I_III, III_I = _LONG_EMBEDDING[self.long_range]
        self.force_matrix[Subsystem.I][Subsystem.III] = I_III
        self.force_matrix[Subsystem.III][Subsystem.I] = III_I

    def modify_calculator(
            self,
            calculator: Calculator,
            system: System,
    ) -> None:
        if isinstance(calculator, InterfaceCalculator):
            if isinstance(calculator.interface, MMInterface):
                self.modify_mm_interface(calculator.interface, system)
            if isinstance(calculator.interface, QMInterface):
                self.modify_qm_interface(calculator.interface, system)

    def modify_mm_interface(
            self,
            interface: MMInterface,
            system: System,
    ) -> None:
        qm_atoms = system.qm_region
        mm_atoms = system.mm_region
        atoms = qm_atoms.union(mm_atoms)
        interface.zero_intramolecular(qm_atoms)
        if (
            self.force_matrix[Subsystem.I][Subsystem.III]
            == self.force_matrix[Subsystem.III][Subsystem.I]
        ):
            if (
                self.force_matrix[Subsystem.I][Subsystem.III]
                == TheoryLevel.NO
            ):
                if (
                    self.force_matrix[Subsystem.I][Subsystem.II]
                    == TheoryLevel.NO
                ):
                    interface.zero_intermolecular(qm_atoms)
                elif (
                    self.force_matrix[Subsystem.I][Subsystem.II]
                    == TheoryLevel.MM
                ):
                    interface.add_real_elst(qm_atoms)
                interface.zero_charges(qm_atoms)
            elif (
                self.force_matrix[Subsystem.I][Subsystem.III]
                == TheoryLevel.MM
            ):
                if (
                    self.force_matrix[Subsystem.I][Subsystem.II]
                    == TheoryLevel.QM
                ):
                    interface.add_real_elst(qm_atoms, -1)
            else:
                raise TypeError("...")
        else:
            interface.zero_forces(qm_atoms)
            inclusion = np.zeros((len(atoms), 3))
            inclusion[list(qm_atoms), :] = 1
            interface.add_non_elst(qm_atoms, inclusion=inclusion)
            if (
                self.force_matrix[Subsystem.I][Subsystem.III]
                == TheoryLevel.NO
            ):
                if (
                    self.force_matrix[Subsystem.I][Subsystem.II]
                    == TheoryLevel.MM
                ):
                    interface.add_real_elst(qm_atoms, 1, inclusion=inclusion)
                elif (
                    self.force_matrix[Subsystem.I][Subsystem.II]
                    == TheoryLevel.QM
                ):
                    interface.add_real_elst(qm_atoms, -1)
            elif (
                self.force_matrix[Subsystem.I][Subsystem.III]
                == TheoryLevel.QM
            ):
                interface.add_real_elst(qm_atoms, -1)
                ...
            else:
                raise TypeError("...")

    def modify_qm_interface(
            self,
            interface: QMInterface,
            system: System,
    ) -> None:
        if (
            self.force_matrix[Subsystem.I][Subsystem.II] == TheoryLevel.QM
            or self.force_matrix[Subsystem.II][Subsystem.I] == TheoryLevel.QM
        ):
            ...
        else:
            interface.disable_embedding()

    # def __or__(self, other: Any) -> Hamiltonian:
    #    if not isinstance(other, (int, float)):
    #        raise TypeError("...")
    #    self.embedding_cutoff = other
    #    return self

    def __str__(self) -> str:
        return "H^{QM/MM}"
