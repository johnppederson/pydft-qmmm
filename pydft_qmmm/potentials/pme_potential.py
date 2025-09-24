"""Classes for introducing arbitrary potentials to QM calculations.
"""
from __future__ import annotations

__all__ = [
    "HelPMEPyInterface",
    "PMEElectronicPotential",
    "PMENuclearPotential",
    "PMEExcludedPotential",
]

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.utils import DependencyImportError
from pydft_qmmm.utils import compute_lattice_constants
from pydft_qmmm.utils import ELEMENT_TO_MASS
from pydft_qmmm.utils import KJMOL_PER_EH
from pydft_qmmm.utils import system_cache

from .potential import ElectronicPotential
from .potential import AtomicPotential

try:
    import helpme_py
except ImportError:
    raise DependencyImportError(
        "helPME-py",
        "performing QM/MM/PME calculations",
        "https://github.com/johnppederson/helpme-py",
    )

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pydft_qmmm import System


@dataclass(frozen=True)
class HelPMEPyInterface:
    r"""A mix-in for initializing and storing PME settings for a system.

    Args:
        system: The system which will be tied to the helPME-py
            interface.
        pme_gridnumber: The number of grid points to include along each
            lattice edge in PME summation.
        pme_alpha: The Gaussian width parameter in Ewald summation
            (:math:`\mathrm{\mathring{A}^{-1}}`).
        pme_spline_order: The order of splines to use in the
            interpolation on the FFT grid.

    Attributes:
        pme: The helPME-py PME object.
    """
    system: System
    pme_alpha: float
    pme_gridnumber: tuple[int, int, int]
    pme_spline_order: int
    pme: helpme_py.PMEInstanceD = field(
        default_factory=helpme_py.PMEInstanceD,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Perform setup for the helPME-py PME object."""
        self.pme.setup(
            1,
            self.pme_alpha,
            self.pme_spline_order,
            *self.pme_gridnumber,
            1389.3545764438198,  # Todo: Add to constants.
            0,
        )
        self.update_box(self.system.box)
        self.system.box.register_notifier(self.update_box)

    def update_box(self, box: NDArray[np.float64]) -> None:
        r"""Set the lattice vectors used by the helPME-py object.

        Args:
            box: The lattice vectors (:math:`\mathrm{\mathring{A}}`) of
                the box containing the system.
        """
        self.pme.set_lattice_vectors(
            *compute_lattice_constants(box),
            helpme_py.LatticeType.XAligned,
        )


class PMEElectronicPotential(ElectronicPotential, HelPMEPyInterface):
    r"""Representation of a PME electrostatic potential on the electrons.

    Args:
        system: The system which will be tied to the helPME-py
            interface.
        pme_gridnumber: The number of grid points to include along each
            lattice edge in PME summation.
        pme_alpha: The Gaussian width parameter in Ewald summation
            (:math:`\mathrm{\mathring{A}^{-1}}`).
        pme_spline_order: The order of splines to use in the
            interpolation on the FFT grid.

    Attributes:
        pme: The helPME-py PME object.
    """

    def compute_potential(
            self,
            coordinates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Calculate the PME potential at arbitrary coordinates.

        Args:
            coordinates: An array of coordinates
                (:math:`\mathrm{\mathring{A}}`) at which to calculate
                the PME potential.

        Returns:
            An array of the PME potential
            (:math:`\mathrm{kJ\;mol^{-1}\;e^{-1}}`),
            corresponding to the provided coordinates.
        """
        potential = np.zeros((len(coordinates), 1))
        excluded = sorted(self.system.select("not subsystem III"))
        self.pme.compute_P_rec(
            0,
            helpme_py.MatrixD(self.system.charges.reshape(-1, 1)),
            helpme_py.MatrixD(self.system.positions),
            helpme_py.MatrixD(coordinates),
            0,
            helpme_py.MatrixD(potential),
        )
        self.pme.compute_P_adj(
            0,
            helpme_py.MatrixD(self.system.charges[excluded].reshape(-1, 1)),
            helpme_py.MatrixD(self.system.positions[excluded, :]),
            helpme_py.MatrixD(coordinates),
            helpme_py.MatrixD(potential),
            False,
        )
        return -potential / KJMOL_PER_EH


class PMENuclearPotential(AtomicPotential, HelPMEPyInterface):
    r"""Representation of a PME electrostatic potential on the nuclei.

    Args:
        system: The system which will be tied to the helPME-py
            interface.
        pme_gridnumber: The number of grid points to include along each
            lattice edge in PME summation.
        pme_alpha: The Gaussian width parameter in Ewald summation
            (:math:`\mathrm{\mathring{A}^{-1}}`).
        pme_spline_order: The order of splines to use in the
            interpolation on the FFT grid.

    Attributes:
        pme: The helPME-py PME object.
    """

    def compute_energy(self) -> float:
        r"""Compute the system energy associated with the potential.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the potential.
        """
        nuclear_charges = self.get_nuclear_charges()
        potential = self.compute_potential_and_derivs()
        return float(potential[:, 0].dot(nuclear_charges))

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system induced by the potential.

        Returns:
            The forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) acting
            on atoms in the system.
        """
        nuclear_charges = self.get_nuclear_charges()
        potential = self.compute_potential_and_derivs()
        nuclear_forces = -(potential[:, 1:].T * nuclear_charges).T
        nuclei = sorted(self.system.select("subsystem I"))
        forces = np.zeros(self.system.positions.shape)
        forces[nuclei] = nuclear_forces
        return forces

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy associated with the potential.

        Returns:
            The components of the energy (:math:`\mathrm{kJ\;mol^{-1}}`)
            of the system.
        """
        components: dict[str, float] = {}
        return components

    def get_nuclear_charges(self) -> list[int]:
        """Get the nuclear charges of the Subsystem I atoms.

        Returns:
            The integer nuclear charges of the Subsystem I atoms.
        """
        nuclei = sorted(self.system.select("subsystem I"))
        elements = list(ELEMENT_TO_MASS.keys())
        nuclear_charges = [elements.index(self.system.elements[atom])
                           for atom in nuclei]
        return nuclear_charges

    @system_cache("subsystems", "positions", "charges")
    def compute_potential_and_derivs(self) -> NDArray[np.float64]:
        r"""Calculate the PME potential and derivatives at the nuclei.

        Returns:
            An array of the PME potential
            (:math:`\mathrm{kJ\;mol^{-1}\;e^{-1}}`) and its derivatives
            (:math:`\mathrm{kJ\;mol^{-1}\;e^{-1}\;\mathring{A}^{-1}}`),
            corresponding to the provided coordinates.
        """
        nuclei = sorted(self.system.select("subsystem I"))
        excluded = sorted(self.system.select("not subsystem III"))
        coordinates = self.system.positions[nuclei]
        potential = np.zeros((len(coordinates), 4))
        self.pme.compute_P_rec(
            0,
            helpme_py.MatrixD(self.system.charges.reshape(-1, 1)),
            helpme_py.MatrixD(self.system.positions),
            helpme_py.MatrixD(coordinates),
            1,
            helpme_py.MatrixD(potential),
        )
        self.pme.compute_PDP_adj(
            0,
            helpme_py.MatrixD(self.system.charges[excluded].reshape(-1, 1)),
            helpme_py.MatrixD(self.system.positions[excluded, :]),
            helpme_py.MatrixD(coordinates),
            helpme_py.MatrixD(potential),
            False,
        )
        return potential


class PMEExcludedPotential(PMENuclearPotential):
    r"""Representation of a PME electrostatic potential on excluded atoms.

    Args:
        system: The system which will be tied to the helPME-py
            interface.
        pme_gridnumber: The number of grid points to include along each
            lattice edge in PME summation.
        pme_alpha: The Gaussian width parameter in Ewald summation
            (:math:`\mathrm{\mathring{A}^{-1}}`).
        pme_spline_order: The order of splines to use in the
            interpolation on the FFT grid.

    Attributes:
        pme: The helPME-py PME object.
    """

    def compute_energy(self) -> float:
        r"""Compute the system energy associated with the potential.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the potential.
        """
        nuclei = sorted(self.system.select("subsystem I"))
        charges = self.system.charges[nuclei]
        potential = self.compute_potential_and_derivs()
        return -float(potential[:, 0].dot(charges))

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system induced by the potential.

        Returns:
            The forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) acting
            on atoms in the system.
        """
        forces = np.zeros(self.system.positions.shape)
        return forces
