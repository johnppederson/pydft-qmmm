"""Classes for introducing arbitrary potentials to QM calculations.
"""
from __future__ import annotations

import os
from math import erfc

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
from pydft_qmmm.utils import atomic_number
from pydft_qmmm.utils import compute_lattice_constants
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
    from pydft_qmmm.interfaces import QMInterface


# Enable nearest-image PME exclusions with PYDFT_QMMM_PME_MINIMUM_IMAGE=1.
PME_MINIMUM_IMAGE_ENV = "PYDFT_QMMM_PME_MINIMUM_IMAGE"


def pme_minimum_image() -> bool:
    """Whether exclusions use the nearest-image separation."""
    return bool(os.environ.get(PME_MINIMUM_IMAGE_ENV, "").strip())


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

    def _source_potential_and_derivs(
            self,
            coordinates: NDArray[np.float64],
            weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Evaluate quadrature charges at the system atoms.

        This exchanges the sources and targets of ``compute_potential``.
        The reciprocal operator is symmetric under this exchange.

        Args:
            coordinates: An array of quadrature coordinates
                (:math:`\mathrm{\mathring{A}}`).
            weights: An array of signed charges (:math:`e`) at the
                quadrature coordinates.

        Returns:
            The potential (:math:`\mathrm{kJ\;mol^{-1}\;e^{-1}}`) and
            its derivatives
            (:math:`\mathrm{kJ\;mol^{-1}\;e^{-1}\;\mathring{A}^{-1}}`)
            at every atom in the system.
        """
        sources = np.ascontiguousarray(
            np.asarray(weights, dtype=float).reshape(-1, 1),
        )
        points = np.ascontiguousarray(
            np.asarray(coordinates, dtype=float).reshape(-1, 3),
        )
        targets = np.ascontiguousarray(
            np.asarray(self.system.positions, dtype=float),
        )
        potential = np.zeros((len(targets), 4))
        self.pme.compute_P_rec(
            0,
            helpme_py.MatrixD(sources),
            helpme_py.MatrixD(points),
            helpme_py.MatrixD(targets),
            1,
            helpme_py.MatrixD(potential),
        )
        excluded = sorted(self.system.select("not subsystem III"))
        if excluded:
            adjustment = np.zeros((len(excluded), 4))
            self.pme.compute_PDP_adj(
                0,
                helpme_py.MatrixD(sources),
                helpme_py.MatrixD(points),
                helpme_py.MatrixD(targets[excluded, :]),
                helpme_py.MatrixD(adjustment),
                pme_minimum_image(),
            )
            potential[excluded] += adjustment
        return potential

    def _source_forces(
            self,
            coordinates: NDArray[np.float64],
            weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Calculate the forces a set of source charges exerts.

        Args:
            coordinates: The source coordinates
                (:math:`\mathrm{\mathring{A}}`).
            weights: The source charges (:math:`e`).

        Returns:
            The forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            exerted on every atom in the system.
        """
        potential = self._source_potential_and_derivs(coordinates, weights)
        charges = np.asarray(self.system.charges).reshape(-1, 1)
        return -potential[:, 1:] * charges

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
            pme_minimum_image(),
        )
        return -potential / KJMOL_PER_EH

    def compute_source_potential(
            self,
            coordinates: NDArray[np.float64],
            weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Calculate the potential a quadrature charge density makes.

        Args:
            coordinates: An array of quadrature coordinates
                (:math:`\mathrm{\mathring{A}}`).
            weights: An array of signed charges (:math:`e`) at the
                quadrature coordinates.

        Returns:
            The potential (:math:`\mathrm{kJ\;mol^{-1}\;e^{-1}}`) at
            every atom in the system.
        """
        return self._source_potential_and_derivs(coordinates, weights)[:, 0]

    def compute_source_forces(
            self,
            coordinates: NDArray[np.float64],
            weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Calculate the PME forces exerted by a quadrature density.

        Args:
            coordinates: An array of quadrature coordinates
                (:math:`\mathrm{\mathring{A}}`).
            weights: An array of signed charges (:math:`e`) at the
                quadrature coordinates.

        Returns:
            The forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            exerted on every atom in the system.
        """
        return self._source_forces(coordinates, weights)


@dataclass(frozen=True)
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
        qm_interface: The QM interface providing effective nuclear
            charges. If omitted, atomic numbers are used.

    Attributes:
        pme: The helPME-py PME object.
    """
    qm_interface: QMInterface | None = None

    def source_charges(self) -> NDArray[np.float64]:
        r"""Get the effective nuclear charges of Subsystem I.

        Returns:
            The signed charges (:math:`e`) whose interaction with the
            PME field this potential represents.
        """
        if self.qm_interface is not None:
            return np.asarray(
                self.qm_interface.nuclear_charges(), dtype=float,
            )
        return np.asarray(self.get_nuclear_charges(), dtype=float)

    def compute_energy(self) -> float:
        r"""Compute the system energy associated with the potential.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the potential.
        """
        potential = self.compute_potential_and_derivs()
        return float(potential[:, 0].dot(self.source_charges()))

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute nuclear and reciprocal source forces.

        Returns:
            The forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) acting
            on atoms in the system.
        """
        charges = self.source_charges()
        nuclei = sorted(self.system.select("subsystem I"))
        forces = self._source_forces(
            self.system.positions[nuclei, :], charges,
        )
        potential = self.compute_potential_and_derivs()
        forces[nuclei] -= potential[:, 1:] * charges.reshape(-1, 1)
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
        """Get the all-electron nuclear charges of Subsystem I.

        Returns:
            The integer nuclear charges of the Subsystem I atoms.
        """
        nuclei = sorted(self.system.select("subsystem I"))
        return [
            atomic_number(self.system.elements[atom]) for atom in nuclei
        ]

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
            pme_minimum_image(),
        )
        return potential


@dataclass(frozen=True)
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

    # Force mixing removes the matching classical forces on subsystem I.
    # With include_forces=False, omit those correction forces; when
    # real_space_cutoff is set, retain the correction forces on MM atoms.
    # Without a cutoff, the correction remains energy only
    # (Pederson and McDaniel, JCP 161, 034103, Eq. 9).
    include_forces: bool = True
    # Periodic-engine force mixing removes the complete classical QM
    # electrostatic force on subsystem I. Its energy correction must include
    # the real-space remainder and count QM-QM periodic interactions only
    # once; subsystem II and III atoms keep its derivative.
    real_space_cutoff: float | None = None

    def compute_energy(self) -> float:
        """Remove retained classical QM electrostatics in force mixing."""
        energy = super().compute_energy()
        if self.real_space_cutoff is None:
            return energy
        if self.include_forces:
            raise ValueError("Periodic exclusion correction requires energy-only force mixing")
        nuclei = sorted(self.system.select("subsystem I"))
        positions = np.asarray(self.system.positions)
        charges = np.asarray(self.system.charges)
        coords = np.ascontiguousarray(positions[nuclei])
        q = np.ascontiguousarray(charges[nuclei, None])
        potential = np.zeros((len(nuclei), 1))
        self.pme.compute_P_rec(
            0, helpme_py.MatrixD(q), helpme_py.MatrixD(coords),
            helpme_py.MatrixD(coords), 0, helpme_py.MatrixD(potential),
        )
        self.pme.compute_P_adj(
            0, helpme_py.MatrixD(q), helpme_py.MatrixD(coords),
            helpme_py.MatrixD(coords), helpme_py.MatrixD(potential),
            pme_minimum_image(),
        )
        # The inherited -q_I phi(all) counts I-I twice. OpenMM contains
        # one half q_I phi(I), with direct intramolecular terms excluded.
        energy += float(0.5 * q[:, 0] @ potential[:, 0])
        return energy + self._real_space_remainder()[0]

    def _real_space_remainder(self) -> tuple[float, NDArray[np.float64]]:
        r"""Energy and forces of the I-(II, III) real-space remainder.

        Base OpenMM retains erfc inside its periodic cutoff. The I-II
        Coulomb subtraction plus erf exclusion removes direct erfc for
        all II pairs. This term cancels the difference, including III
        real-space tails.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) and the forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) on
            every atom in the system.
        """
        nuclei = sorted(self.system.select("subsystem I"))
        mm = sorted(self.system.select("subsystem II or subsystem III"))
        near = self.system.select("subsystem II")
        positions = np.asarray(self.system.positions)
        charges = np.asarray(self.system.charges)
        alpha = self.pme_alpha
        box = np.asarray(self.system.box).T

        def value_and_slope(r: float) -> tuple[float, float]:
            """erfc(alpha r)/r and its derivative with respect to r."""
            value = erfc(alpha * r) / r
            slope = -(value + 2 * alpha / np.sqrt(np.pi)
                      * np.exp(-(alpha * r) ** 2)) / r
            return value, slope

        energy = 0.0
        forces = np.zeros(positions.shape)
        for i in nuclei:
            for j in mm:
                delta = positions[i] - positions[j]
                direct = np.linalg.norm(delta)
                nearest = delta.copy()
                for axis in (2, 1, 0):
                    nearest -= box[axis] * np.floor(nearest[axis] / box[axis, axis] + 0.5)
                distance = np.linalg.norm(nearest)
                prefactor = 1389.3545764438198 * charges[i] * charges[j]
                # dE/dR_i; the image shift is constant, so dE/dR_j = -dE/dR_i.
                gradient = np.zeros(3)
                if distance < self.real_space_cutoff:
                    value, slope = value_and_slope(distance)
                    energy -= prefactor * value
                    gradient -= prefactor * slope * nearest / distance
                if j in near:
                    value, slope = value_and_slope(direct)
                    energy += prefactor * value
                    gradient += prefactor * slope * delta / direct
                forces[i] -= gradient
                forces[j] += gradient
        return energy, forces


    def compute_forces(self) -> NDArray[np.float64]:
        """Differentiate the correction only when its partner forces remain.

        Periodic-engine force mixing removes the classical forces on
        subsystem I only. The correction energy also depends on the
        subsystem II and III positions, and OpenMM keeps the matching
        classical forces there, so those atoms get the full derivative.
        """
        if self.include_forces:
            return super().compute_forces()
        # A plain array: in-place updates to an observed copy of the
        # positions would notify the interfaces.
        forces = np.zeros(np.shape(self.system.positions))
        if self.real_space_cutoff is None:
            return forces
        # The I-I self term depends on subsystem I alone and is dropped
        # with the other subsystem I forces below.
        forces += super().compute_forces()
        forces += self._real_space_remainder()[1]
        forces[sorted(self.system.select("subsystem I"))] = 0.0
        return forces

    def source_charges(self) -> NDArray[np.float64]:
        r"""Get the excluded force-field charges of Subsystem I.

        Negating these charges removes OpenMM's reciprocal interaction.

        Returns:
            The negated force-field charges (:math:`e`) of the
            Subsystem I atoms.
        """
        nuclei = sorted(self.system.select("subsystem I"))
        return -np.asarray(self.system.charges[nuclei], dtype=float)
