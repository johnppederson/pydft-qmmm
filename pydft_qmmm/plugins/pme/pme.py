"""A plugin organizing the QM/MM/PME algorithm for calculations.
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

import numpy as np
from openmm import NonbondedForce
from simtk.unit import nanometer

from .pme_utils import pme_components
from pydft_qmmm.calculators import InterfaceCalculator
from pydft_qmmm.interfaces.qmmm_pme_openmm.openmm_interface import (
    PMEOpenMMInterface,
)
from pydft_qmmm.interfaces.qmmm_pme_psi4.psi4_interface import (
    PMEPsi4Interface,
)
from pydft_qmmm.plugins.plugin import CompositeCalculatorPlugin
# This is bad practice and should be removed when the hacked versions
# of Psi4 and OpenMM are deprecated.

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


class PME(CompositeCalculatorPlugin):
    """Perform the QM/MM/PME algorithm during calculations.
    """

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
        for calc in calculator.calculators:
            if isinstance(calc, InterfaceCalculator):
                if isinstance(calc.interface, PMEOpenMMInterface):
                    nonbonded_forces = [
                        force for force in
                        calc.interface._base_context.getSystem().getForces()
                        if isinstance(force, NonbondedForce)
                    ]
                    (
                        self.pme_alpha, self.pme_gridnumber, _, _,
                    ) = nonbonded_forces[0].getPMEParameters()
                    self.pme_alpha *= nanometer
                    self.mm_interface = calc.interface
                if isinstance(calc.interface, PMEPsi4Interface):
                    self.qm_interface = calc.interface
        calculator.calculate = self._modify_calculate(calculator.calculate)

    def _modify_calculate(
            self,
            calculate: CalculateMethod,
    ) -> CalculateMethod:
        """Modify the calculate routine to perform QM/MM/PME.

        Args:
            calculate: The calculation routine to modify.

        Returns:
            The modified calculation routine which implements
            QM/MM/PME and all requisite operations.
        """
        def inner(
                return_forces: bool | None = True,
                return_components: bool | None = True,
        ) -> Results:
            pme_potential = self.mm_interface.compute_recip_potential()
            quadrature = self.qm_interface.compute_quadrature()
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
            self.qm_interface.update_quad_extd_pot(
                np.array(tuple(quadrature_pme_potential)),
            )
            self.qm_interface.update_nuc_extd_pot(
                np.array(tuple(nuclei_pme_potential)),
            )
            self.qm_interface.update_nuc_extd_grad(
                np.array(
                    tuple(
                        [tuple(x) for x in nuclei_pme_gradient],
                    ),
                ),
            )
            results = calculate(return_forces, return_components)
            results.energy += reciprocal_energy
            results.components.update(
                {"Correction Reciprocal-Space Energy": reciprocal_energy},
            )
            return results
        return inner
