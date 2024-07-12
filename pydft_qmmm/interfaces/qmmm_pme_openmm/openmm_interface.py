"""The QM/MM/PME-hacked OpenMM software interface.
"""
from __future__ import annotations

from .openmm_utils import _generate_state
from pydft_qmmm.common import KJMOL_PER_EH
from pydft_qmmm.interfaces.openmm.openmm_interface import OpenMMInterface


class PMEOpenMMInterface(OpenMMInterface):
    """A software interface wrapping OpenMM functionality.
    """

    def compute_recip_potential(self) -> list[float]:
        r"""Calculate the PME potential grid on the system.

        Returns:
            A list of electrical potential values
            (:math:`\mathrm{a.u.}\;e^{-1}`) corresponding to each PME
            grid point.
        """
        state = _generate_state(self._base_context)
        potential_grid = [pot / KJMOL_PER_EH for pot in state.getVext_grid()]
        return potential_grid
