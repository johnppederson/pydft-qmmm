"""Region III reciprocal embedding for a periodic QM cell."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pydft_qmmm.potentials.pme_potential import helpme_py
from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential
from pydft_qmmm.utils import KJMOL_PER_EH


@dataclass(frozen=True)
class PeriodicPMEElectronicPotential(PMEElectronicPotential):
    """Use region III-only reciprocal sources, including periodic images.

    I has its own periodic QM Hamiltonian; II has a periodic Gaussian field.
    Local erf exclusions would leave unwanted I/II image contributions.
    """

    excluded_indices: tuple[int, ...] = ()

    def _source_indices(self):
        region_iii = set(self.system.select("subsystem III"))
        return sorted(region_iii - set(self.excluded_indices))

    def compute_potential(self, coordinates):
        """Return the region III electron potential in Hartree."""
        indices = self._source_indices()
        potential = np.zeros((len(coordinates), 1))
        if indices:
            self.pme.compute_P_rec(
                0,
                helpme_py.MatrixD(
                    np.ascontiguousarray(
                        self.system.charges[indices].reshape(-1, 1),
                    ),
                ),
                helpme_py.MatrixD(
                    np.ascontiguousarray(
                        self.system.positions[indices],
                    ),
                ),
                helpme_py.MatrixD(np.ascontiguousarray(coordinates)),
                0,
                helpme_py.MatrixD(potential),
            )
        return -potential / KJMOL_PER_EH

    def _source_potential_and_derivs(self, coordinates, weights):
        """Return the reaction field at III sites; other system rows are zero.

        Columns: potential (kJ/mol/e), then x/y/z derivatives (kJ/mol/e/Å).
        """
        indices = self._source_indices()
        result = np.zeros((len(self.system), 4))
        if indices:
            potential = np.zeros((len(indices), 4))
            self.pme.compute_P_rec(
                0,
                helpme_py.MatrixD(
                    np.ascontiguousarray(
                        np.asarray(weights, dtype=float).reshape(-1, 1),
                    ),
                ),
                helpme_py.MatrixD(np.ascontiguousarray(coordinates)),
                helpme_py.MatrixD(
                    np.ascontiguousarray(
                        self.system.positions[indices],
                    ),
                ),
                1,
                helpme_py.MatrixD(potential),
            )
            result[indices] = potential
        return result
