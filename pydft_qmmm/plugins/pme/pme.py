#! /usr/bin/env python3
"""A module defining the pluggable implementation of the SETTLE
algorithm for the QM/MM/PME repository.
"""
from __future__ import annotations

from dataclasses import astuple
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

from openmm import NonbondedForce
from simtk.unit import nanometer

from .pme_utils import pme_components
from pydft_qmmm.calculators.calculator import CalculatorType
from pydft_qmmm.calculators.calculator import Results
from pydft_qmmm.common import KJMOL_PER_EH
from pydft_qmmm.plugins.plugin import QMMMCalculatorPlugin

if TYPE_CHECKING:
    from pydft_qmmm.calculators import QMMMCalculator


class PME(QMMMCalculatorPlugin):
    """A :class:`Plugin` which implements the QM/MM/PME algorithm for
    energy and force calculations.
    """

    def modify(
            self,
            calculator: QMMMCalculator,
    ) -> None:
        """Perform necessary modifications to the :class:`QMMMCalculator`
        object.

        :param calculator: The calculator to modify with the QM/MM/PME
            functionality.
        """
        self._modifieds.append(type(calculator).__name__)
        self.system = calculator.system
        self.calculators = calculator.calculators
        interface = calculator.calculators[
            CalculatorType.MM
        ].interface
        nonbonded_forces = [
            force for force in interface.system.getForces()
            if isinstance(force, NonbondedForce)
        ]
        (
            self.pme_alpha, self.pme_gridnumber, _, _,
        ) = nonbonded_forces[0].getPMEParameters()
        self.pme_alpha *= nanometer
        calculator.calculate = self._modify_calculate(calculator.calculate)

    def _modify_calculate(
            self,
            calculate: Callable[..., tuple[Any, ...]],
    ) -> Callable[..., tuple[Any, ...]]:
        """
        """
        def inner(**kwargs: bool) -> tuple[Any, ...]:
            pme_potential = self.calculators[
                CalculatorType.MM
            ].interface.compute_pme_potential() / KJMOL_PER_EH
            quadrature = self.calculators[
                CalculatorType.QM
            ].interface.compute_quadrature()
            (
                reciprocal_energy, quadrature_pme_potential,
                nuclei_pme_potential, nuclei_pme_gradient,
            ) = pme_components(
                self.system,
                quadrature,
                pme_potential,
                self.pme_gridnumber,
                self.pme_alpha,
            )
            self.calculators[CalculatorType.QM].options.update(
                {
                    "quad_extd_pot": tuple(quadrature_pme_potential),
                    "nuc_extd_pot": tuple(nuclei_pme_potential),
                    "nuc_extd_grad": tuple(
                        [tuple(x) for x in nuclei_pme_gradient],
                    ),
                },
            )
            qmmm_energy, qmmm_forces, qmmm_components = calculate(**kwargs)
            qmmm_energy += reciprocal_energy
            results = Results(qmmm_energy)
            results.forces = qmmm_forces
            qmmm_components.update(
                {"Reciprocal-Space Correction Energy": reciprocal_energy},
            )
            results.components = qmmm_components
            return astuple(results)
        return inner
