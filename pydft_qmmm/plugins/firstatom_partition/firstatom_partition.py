"""A plugin for performing residue-wise system partitioning by first atom.
"""
from __future__ import annotations

__all__ = ["FirstAtomPartition"]

from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.utils import Subsystem
from pydft_qmmm.calculators.composite_calculator import PartitionPlugin

if TYPE_CHECKING:
    from collections.abc import Callable
    from pydft_qmmm.calculators import Results


class FirstAtomPartition(PartitionPlugin):
    r"""Partition subsystems residue-wise according to first atom coordinate.

    Args:
        query: The VMD-like query representing the group of atoms whose
            subsystem membership will be determined on an residue-wise
            basis.
        cutoff: The cutoff distance (:math:`\mathrm{\mathring{A}}`) to
            apply in the partition.
    """

    def __init__(
            self,
            query: str,
            cutoff: float | int,
    ) -> None:
        self.query = query
        self.cutoff = cutoff

    def _modify_calculate(
            self,
            calculate: Callable[[bool, bool], Results],
    ) -> Callable[[bool, bool], Results]:
        """Modify the calculate routine to perform residue-wise partitioning.

        Args:
            calculate: The calculation routine to modify.

        Returns:
            The modified calculation routine which implements
            residue-wise partitioning according to first atom
            coordinate before calculation.
        """
        def inner(
                return_forces: bool = True,
                return_components: bool = True,
        ) -> Results:
            self.generate_partition()
            results = calculate(return_forces, return_components)
            return results
        return inner

    def generate_partition(self) -> None:
        """Perform the residue-wise system partitioning.
        """
        qm_region = self.calculator.system.select("subsystem I")
        qm_centroid = np.average(
            self.calculator.system.positions[sorted(qm_region), :],
            axis=0,
        )
        region_ii: set[int] = set()
        selection = self.calculator.system.select(self.query)
        for residue in self.calculator.system.residue_map.values():
            atoms = residue & selection - qm_region
            if atoms:
                index = sorted(atoms)[0]
                nth_centroid = self.calculator.system.positions[index, :]
                r_vector = nth_centroid - qm_centroid
                distance = np.sum(r_vector**2)**0.5
                if distance < self.cutoff:
                    region_ii |= atoms
        # Update the topology with the current embedding atoms.
        region_iii = selection - qm_region - region_ii
        self.calculator.system.subsystems[sorted(region_iii)] = Subsystem.III
        self.calculator.system.subsystems[sorted(region_ii)] = Subsystem.II
