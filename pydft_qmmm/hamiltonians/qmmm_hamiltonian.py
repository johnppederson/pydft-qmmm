"""A Hamiltonian defining the inter-subsystem coupling in QM/MM.

Attributes:
    _DEFAULT_FORCE_MATRIX: The default force matrix, which has no
        level of theory coupling subsystem I to subsystems II and III.
    _CLOSE_EMBEDDING: The levels of theory for I-II and II-I forces for
        different close-range embedding schemes.
    _LONG_EMBEDDING: The levels of theory for I-III and III-I forces for
        different long-range embedding schemes.
    _SUPPORTED_EMBEDDING: Allowed pairs of close-range and long-range
        embedding schemes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

from .hamiltonian import CouplingHamiltonian
from pydft_qmmm.calculators import InterfaceCalculator
from pydft_qmmm.common import lazy_load
from pydft_qmmm.common import Subsystem
from pydft_qmmm.common import TheoryLevel
from pydft_qmmm.interfaces import MMInterface
from pydft_qmmm.interfaces import QMInterface


if TYPE_CHECKING:
    from pydft_qmmm import System
    from pydft_qmmm.calculators import Calculator
    from pydft_qmmm.calculators import CompositeCalculator


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
    r"""A Hamiltonian defining inter-subsystem coupling in QM/MM.

    Args:
        close_range: The name of the embedding procedure for
            close-range (I-II) interactions.
        long_range: The name of the embedding procedure for
            long-range (I-III) interactions.
        cutoff: The cutoff distance (:math:`\mathrm{\mathring{A}}`) at
            which to partition a system into subsystems II and III.
    """
    close_range: str = "electrostatic"
    long_range: str = "cutoff"
    cutoff: float | int = 14.

    def __post_init__(self) -> None:
        """Generate the force matrix for the selected embedding scheme.
        """
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
        """Modify a calculator to represent the coupling.

        Args:
            calculator: A calculator which is defined in part by the
                system.
            system: The system that will be used to modify the
                calculator.
        """
        if isinstance(calculator, InterfaceCalculator):
            if isinstance(calculator.interface, MMInterface):
                self.modify_mm_interface(calculator.interface, system)
            if isinstance(calculator.interface, QMInterface):
                self.modify_qm_interface(calculator.interface, system)

    def modify_composite(
            self,
            calculator: CompositeCalculator,
            system: System,
    ) -> None:
        """Modify a composite calculator to represent the coupling.

        Args:
            calculator: A composite calculator which is defined in part
                by the system.
            system: The system that will be used to modify the
                calculator.
        """
        calculator.cutoff = self.cutoff
        if (
            self.force_matrix[Subsystem.I][Subsystem.III]
            == TheoryLevel.QM
        ):
            plugin = lazy_load("pydft_qmmm.plugins.pme")
            calculator.register_plugin(plugin.PME())

    def modify_mm_interface(
            self,
            interface: MMInterface,
            system: System,
    ) -> None:
        """Modify an MM interface to reflect the selected embedding.

        Args:
            interface: The MM interface representing part of the system.
            system: The system that will be used to modify the
                interface.
        """
        qm_atoms = system.select("subsystem I")
        mm_atoms = system.select("subsystem II or subsystem III")
        atoms = qm_atoms | mm_atoms
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
                    warn(
                        (
                            "I-II Mechanical with I-III None embedding is "
                            + "known to produce unstable trajectories."
                        ),
                        RuntimeWarning,
                    )
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
            inclusion[sorted(qm_atoms), :] = 1
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
            else:
                raise TypeError("...")

    def modify_qm_interface(
            self,
            interface: QMInterface,
            system: System,
    ) -> None:
        """Modify a QM interface to reflect the selected embedding.

        Args:
            interface: The QM interface representing part of the system.
            system: The system that will be used to modify the
                interface.
        """
        if (
            self.force_matrix[Subsystem.I][Subsystem.II] == TheoryLevel.QM
            or self.force_matrix[Subsystem.II][Subsystem.I] == TheoryLevel.QM
        ):
            ...
        else:
            interface.disable_embedding()

    def __str__(self) -> str:
        """Create a LATEX string representation of the Hamiltonian.

        Returns:
            The string representation of the Hamiltonian.
        """
        return "H^{QM/MM}"
