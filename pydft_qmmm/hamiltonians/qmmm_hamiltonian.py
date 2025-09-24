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

__all__ = ["QMMMHamiltonian"]

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

from .hamiltonian import CouplingHamiltonian
from pydft_qmmm.calculators import PotentialCalculator
from pydft_qmmm.utils import Subsystem
from pydft_qmmm.utils import TheoryLevel
from pydft_qmmm.interfaces import MMInterface
from pydft_qmmm.interfaces import QMInterface
from pydft_qmmm.plugins import CentroidPartition
from pydft_qmmm.utils import compute_lattice_constants

if TYPE_CHECKING:
    from pydft_qmmm import System
    from pydft_qmmm.calculators import CompositeCalculator
    from pydft_qmmm.calculators import PartitionPlugin


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


class QMMMHamiltonian(CouplingHamiltonian):
    r"""A Hamiltonian defining inter-subsystem coupling in QM/MM.

    Args:
        close_range: The name of the embedding procedure for
            close-range (I-II) interactions.
        long_range: The name of the embedding procedure for
            long-range (I-III) interactions.
        partition: The partition plugin to use for generating subsystem
            II, if any.
        cutoff: The cutoff distance (:math:`\mathrm{\mathring{A}}`) at
            which to partition a system into subsystems II and III.
            This keyword, if specified, will overwrite the cutoff
            distance for a partition plugin object provided to the
            partition keyword.
        pme_alpha: The Ewald-summation Gaussian width parameter
            (:math:`\mathrm{\mathring{A}^{-1}}`) for QM/MM/PME.
        pme_gridnumber: The number of grid points to include along each
            lattice edge for QM/MM/PME.
        pme_spline_order: The order of splines used on the FFT grid for
            QM/MM/PME.
    """

    def __init__(
            self,
            close_range: str = "electrostatic",
            long_range: str = "cutoff",
            partition: PartitionPlugin | None = CentroidPartition("all", 14.),
            cutoff: int | float | None = None,
            pme_alpha: int | float | None = None,
            pme_gridnumber: int | tuple[int, int, int] | None = None,
            pme_spline_order: int | None = None,
    ) -> None:
        if (close_range, long_range) not in _SUPPORTED_EMBEDDING:
            raise TypeError  # Todo: Make this informative.
        self.force_matrix = _DEFAULT_FORCE_MATRIX.copy()
        self.partition = partition
        self.cutoff = cutoff
        self.pme_alpha = pme_alpha
        self.pme_gridnumber = pme_gridnumber
        self.pme_spline_order = pme_spline_order
        # Adjust I-II interaction.
        I_II, II_I = _CLOSE_EMBEDDING[close_range]
        self.force_matrix[Subsystem.I][Subsystem.II] = I_II
        self.force_matrix[Subsystem.II][Subsystem.I] = II_I
        # Adjust I-III interaction.
        I_III, III_I = _LONG_EMBEDDING[long_range]
        self.force_matrix[Subsystem.I][Subsystem.III] = I_III
        self.force_matrix[Subsystem.III][Subsystem.I] = III_I

    def modify_calculator(
            self,
            calculator: CompositeCalculator,
            system: System,
    ) -> None:
        """Modify a composite calculator to include the coupling.

        Args:
            calculator: A composite calculator which is defined in part
                by the system.
            system: The system corresponding to the calculator.
        """
        if self.partition is not None:
            if self.cutoff is not None:
                self.partition.cutoff = self.cutoff
            calculator.register_plugin(self.partition)
        for calc in calculator.calculators:
            if isinstance(calc, PotentialCalculator):
                if isinstance(calc.potential, MMInterface):
                    mm_interface = calc.potential
                if isinstance(calc.potential, QMInterface):
                    qm_interface = calc.potential
        if (
            self.force_matrix[Subsystem.I][Subsystem.III]
            == TheoryLevel.QM
        ):
            from pydft_qmmm.potentials.pme_potential import (
                PMEExcludedPotential,
                PMENuclearPotential,
                PMEElectronicPotential,
            )
            (
                pme_alpha,
                pme_gridnumber,
                pme_spline_order,
            ) = mm_interface.get_pme_parameters()
            advisable_gridnumber = np.ceil(
                compute_lattice_constants(system.box)[0:3],
                dtype=int,
                casting='unsafe',
            ) * 2
            if self.pme_alpha is None:
                self.pme_alpha = pme_alpha
            if self.pme_gridnumber is None:
                self.pme_gridnumber = tuple(advisable_gridnumber.tolist())
            elif isinstance(self.pme_gridnumber, int):
                self.pme_gridnumber = (self.pme_gridnumber,)*3
            if self.pme_spline_order is None:
                self.pme_spline_order = pme_spline_order
            if any(self.pme_gridnumber < advisable_gridnumber):
                warn(
                    (
                        "QM/MM/PME grids with spacings coarser than 0.5 Å"
                        " are known to result in significant numerical"
                        " error.  The minimal advisable gridnumbers are:\n"
                        f"{advisable_gridnumber}"
                    ),
                    RuntimeWarning,
                )
            pme_nuclei = PMENuclearPotential(
                system,
                self.pme_alpha,
                self.pme_gridnumber,
                self.pme_spline_order,
            )
            pme_exclusion = PMEExcludedPotential(
                system,
                pme_alpha,
                pme_gridnumber,
                pme_spline_order,
            )
            calculator.calculators.extend(
                [
                    PotentialCalculator(system, pme_nuclei),
                    PotentialCalculator(system, pme_exclusion),
                ],
            )
            pme_electrons = PMEElectronicPotential(
                system,
                self.pme_alpha,
                self.pme_gridnumber,
                self.pme_spline_order,
            )
            qm_interface.add_electronic_potential(pme_electrons)
        self.apply_exclusions(mm_interface, system)

    def apply_exclusions(
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
                            "known to produce unstable trajectories."
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

    def __str__(self) -> str:
        """Create a LATEX string representation of the Hamiltonian.

        Returns:
            The string representation of the Hamiltonian.
        """
        return "H^{QM/MM}"
