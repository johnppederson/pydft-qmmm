"""A plugin for performing residue-wise system partitioning by centroid.
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.common import Subsystem
from pydft_qmmm.plugins.plugin import PartitionPlugin

if TYPE_CHECKING:
    from pydft_qmmm.calculators import CompositeCalculator
    from pydft_qmmm.common import Results
    import mypy_extensions
    CalculateMethod = Callable[
        [
            mypy_extensions.DefaultArg(
                bool | None,
                "return_forces",  # noqa: F821
            ),
            mypy_extensions.DefaultArg(
                bool | None,
                "return_components",  # noqa: F821
            ),
        ],
        Results,
    ]


class CentroidPartition(PartitionPlugin):
    """Partition subsystems residue-wise according to centroid.

    Args:
        query: The VMD-like query representing the group of atoms whose
            subsystem membership will be determined on an residue-wise
            basis.
    """

    def __init__(
            self,
            query: str,
    ) -> None:
        self._query = query

    def modify(
            self,
            calculator: CompositeCalculator,
    ) -> None:
        """Modify the functionality of a calculator.

        Args:
            calculator: The calculator whose functionality will be
                modified by the plugin.
        """
        self._modifieds.append(type(calculator).__name__)
        self.system = calculator.system
        self.cutoff = calculator.cutoff
        calculator.calculate = self._modify_calculate(
            calculator.calculate,
        )

    def _modify_calculate(
            self,
            calculate: CalculateMethod,
    ) -> CalculateMethod:
        """Modify the calculate routine to perform residue-wise partitioning.

        Args:
            calculate: The calculation routine to modify.

        Returns:
            The modified calculation routine which implements
            residue-wise partitioning according to centroid before
            calculation.
        """
        def inner(
                return_forces: bool | None = True,
                return_components: bool | None = True,
        ) -> Results:
            self.generate_partition()
            results = calculate(return_forces, return_components)
            return results
        return inner

    def generate_partition(self) -> None:
        """Perform the residue-wise system partitioning.
        """
        qm_region = sorted(self.system.select("subsystem I"))
        qm_centroid = np.average(
            self.system.positions[qm_region, :],
            axis=0,
        )
        region_ii: list[int] = []
        selection = self.system.select(self._query)
        for residue in self.system.residue_map.values():
            atoms = sorted(residue & selection)
            not_qm = set(qm_region).isdisjoint(set(atoms))
            if not_qm and atoms:
                nth_centroid = np.average(
                    self.system.positions[atoms, :],
                    axis=0,
                )
                r_vector = nth_centroid - qm_centroid
                distance = np.sum(r_vector**2)**0.5
                if distance < self.cutoff:
                    region_ii.extend(atoms)
        # Update the topology with the current embedding atoms.
        temp = [Subsystem.III]*len(self.system)
        temp = [
            Subsystem.II if i in region_ii else x
            for i, x in enumerate(temp)
        ]
        temp = [
            Subsystem.I if i in qm_region else x
            for i, x in enumerate(temp)
        ]
        self.system.subsystems = np.array(temp)
