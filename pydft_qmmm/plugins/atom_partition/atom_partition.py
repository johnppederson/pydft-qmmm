"""A plugin for performing atom-by-atom system partitioning.
"""
from __future__ import annotations

__all__ = ["AtomPartition"]

from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.utils import Subsystem
from pydft_qmmm.calculators.composite_calculator import PartitionPlugin

if TYPE_CHECKING:
    from collections.abc import Callable
    from pydft_qmmm.calculators import Results


class AtomPartition(PartitionPlugin):
    r"""Partition subsystems atom-by-atom.

    Args:
        query: The VMD-like query representing the group of atoms whose
            subsystem membership will be determined on an atom-by-atom
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
        """Modify the calculate routine to perform atom-wise partitioning.

        Args:
            calculate: The calculation routine to modify.

        Returns:
            The modified calculation routine which implements atom-wise
            partitioning before calculation.
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
        """Perform the atom-wise system partitioning.
        """
        qm_region = self.calculator.system.select("subsystem I")
        qm_centroid = np.average(
            self.calculator.system.positions[sorted(qm_region), :],
            axis=0,
        )
        region_ii: set[int] = set()
        selection = self.calculator.system.select(self.query)
        for atom in range(len(self.calculator.system)):
            atoms = {atom} & selection - qm_region
            if atoms:
                nth_centroid = np.average(
                    self.calculator.system.positions[sorted(atoms), :],
                    axis=0,
                )
                r_vector = nth_centroid - qm_centroid
                distance = np.sum(r_vector**2)**0.5
                if distance < self.cutoff:
                    region_ii |= atoms
        # Update the topology with the current embedding atoms.
        region_iii = selection - qm_region - region_ii
        self.calculator.system.subsystems[sorted(region_iii)] = Subsystem.III
        self.calculator.system.subsystems[sorted(region_ii)] = Subsystem.II
