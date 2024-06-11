#! /usr/bin/env python3
"""A module to define the :class:`Psi4Interface` class.
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
psi4.set_num_threads(1)


class Psi4Interface(QMInterface):
    """A :class:`SoftwareInterface` class which wraps the functional
    components of Psi4.

    :param options: The :class:`Psi4Options` object for the interface.
    :param functional: |functional|
    :param context: The :class:`Psi4Context` object for the interface.
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
        wfn = self._generate_wavefunction()
        energy = wfn.energy()
        energy = energy * KJMOL_PER_EH
        return energy

    def compute_forces(self) -> NDArray[np.float64]:
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
        components: dict[str, float] = {}
        return components

    @lru_cache
    def _generate_wavefunction(self) -> psi4.core.Wavefunction:
        """Generate the Psi4 Wavefunction object for use in Psi4
        calculations.

        :return: The Psi4 Wavefunction object.
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
        self._context.do_embedding = False

    def update_positions(self, positions: NDArray[np.float64]) -> None:
        """Update the atom positions for Psi4.

        :param positions: |positions|
        """
        self._context.update_positions(positions)
        self._generate_wavefunction.cache_clear()

    def update_charges(self, charges: NDArray[np.float64]) -> None:
        """Update the atom positions for Psi4.

        :param positions: |positions|
        """
        self._context.update_charges(charges)
        self._generate_wavefunction.cache_clear()

    def update_subsystems(
            self,
            subsystems: np.ndarray[Any, np.dtype[np.object_]],
    ) -> None:
        """Update the analytic embedding atoms for Psi4.

        :param embedding: |ae_atoms|
        """
        embedding = {i for i, s in enumerate(subsystems) if s == Subsystem.II}
        self._context.update_embedding(embedding)
        self._generate_wavefunction.cache_clear()

    def update_num_threads(self, num_threads: int) -> None:
        """Update the number of threads for Psi4 to use.

        :param num_threads: The number of threads for Psi4 to use.
        """
        psi4.set_num_threads(num_threads)

    def update_memory(self, memory: str) -> None:
        """Update the amount of memory for Psi4 to use.

        :param memory: The amount of memory for Psi4 to use.
        """
        psi4.set_memory(memory)
