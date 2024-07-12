"""The basic Psi4 software interface.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import psi4.core
from numpy.typing import NDArray

from pydft_qmmm.common import Subsystem
from pydft_qmmm.common.units import BOHR_PER_ANGSTROM
from pydft_qmmm.common.units import KJMOL_PER_EH
from pydft_qmmm.interfaces import QMInterface

if TYPE_CHECKING:
    from typing import Any
    from pydft_qmmm.interfaces import QMSettings
    from .psi4_utils import Psi4Context
    from .psi4_utils import Psi4Options

psi4.core.be_quiet()


class Psi4Interface(QMInterface):
    """A software interface wrapping Psi4 functionality.

    Args:
        settings: The settings used to build the Psi4 interface.
        options: The Psi4 global options derived from the settings.
        functional: The name of the functional to use for
            exchange-correlation calculations.
        context: An object which holds system information to feed into
            Psi4.
    """

    def __init__(
            self,
            settings: QMSettings,
            options: Psi4Options,
            functional: str,
            context: Psi4Context,
    ) -> None:
        self._settings = settings
        self._options = options
        self._functional = functional
        self._context = context

    def compute_energy(self) -> float:
        r"""Compute the energy of the system using Psi4.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the system.
        """
        wfn = self._generate_wavefunction()
        energy = wfn.energy()
        energy = energy * KJMOL_PER_EH
        return energy

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system using Psi4.

        Returns:
            The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) acting
            on atoms in the system.
        """
        wfn = self._generate_wavefunction()
        psi4.set_options(asdict(self._options))
        forces = psi4.gradient(
            self._functional,
            ref_wfn=wfn,
        )
        forces = forces.to_array() * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
        forces_temp = np.zeros(self._context.positions.shape)
        qm_indices = list(self._context.atoms)
        qm_indices.sort()
        forces_temp[qm_indices, :] = forces
        if self._context.generate_external_potential():
            embed_indices = list(self._context.embedding)
            embed_indices.sort()
            forces = (
                wfn.external_pot().gradient_on_charges().to_array()
                * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
            )
            forces_temp[embed_indices, :] = forces
        forces = forces_temp
        return forces

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy using OpenMM.

        Returns:
            The components of the energy (:math:`\mathrm{kJ\;mol^{-1}}`)
            of the system.
        """
        components: dict[str, float] = {}
        return components

    @lru_cache
    def _generate_wavefunction(self) -> psi4.core.Wavefunction:
        """Generate the Psi4 Wavefunction object for use by Psi4.

        Returns:
            The Psi4 Wavefunction object, which contains the energy
            and coefficients determined through SCF.
        """
        molecule = self._context.generate_molecule()
        psi4.set_options(asdict(self._options))
        _, wfn = psi4.energy(
            self._functional,
            return_wfn=True,
            molecule=molecule,
            external_potentials=self._context.generate_external_potential(),
        )
        wfn.to_file(
            wfn.get_scratch_filename(180),
        )
        return wfn

    def disable_embedding(self) -> None:
        """Disable electrostatic embedding.
        """
        self._context.do_embedding = False

    def update_positions(self, positions: NDArray[np.float64]) -> None:
        r"""Set the atomic positions used by Psi4.

        Args:
            positions: The positions (:math:`\mathrm{\mathring{A}}`) of the
                atoms within the system.
        """
        self._context.update_positions(positions)
        self._generate_wavefunction.cache_clear()

    def update_charges(self, charges: NDArray[np.float64]) -> None:
        """Set the atomic partial charges used by Psi4 for embedding.

        Args:
            charges: The partial charges (:math:`e`) of the atoms.
        """
        self._context.update_charges(charges)
        self._generate_wavefunction.cache_clear()

    def update_subsystems(
            self,
            subsystems: np.ndarray[Any, np.dtype[np.object_]],
    ) -> None:
        """Adjust which atoms are embedded by subsystem membership.

        Args:
            subsystems: The subsystems of which the atoms are a part.
        """
        embedding = {i for i, s in enumerate(subsystems) if s == Subsystem.II}
        self._context.update_embedding(embedding)
        self._generate_wavefunction.cache_clear()

    def update_threads(self, threads: int) -> None:
        """Set the number of threads used by Psi4.

        Args:
            threads: The number of threads to utilize.
        """
        psi4.set_num_threads(threads)

    def update_memory(self, memory: str) -> None:
        """Set the amount of memory used by OpenMM.

        Args:
            memory: The amount of memory to utilize.
        """
        psi4.set_memory(memory)
