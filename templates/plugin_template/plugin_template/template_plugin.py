"""A module containing the template plugin classes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydft_qmmm.calculators import CalculatorPlugin
from pydft_qmmm.integrators import IntegratorPlugin

if TYPE_CHECKING:
    from collections.abc import Callable
    from pydft_qmmm.calculators import Results
    from pydft_qmmm.integrators import Returns
    from pydft_qmmm.system import System


class TemplateCalculatorPlugin(CalculatorPlugin):
    """A class applying a templated modification to calculators.
    """

    # Perform initialization as needed.
    def __init__(self) -> None:
        ...

    def _modify_calculate(
            self,
            calculate: Callable[[bool, bool], Results],
    ) -> Callable[[bool, bool], Results]:
        """Modify the calculate routine with template.

        Args:
            calculate: The calculation routine to modify.

        Returns:
            The modified calculation routine which implements the
            templated modification.
        """
        # All calculator plugins must modify a Calculator object.  This
        # is achieved by "wrapping" the calculate routine with added
        # functionality.  The actual wrapping is applied internally,
        # so the plugin only needs to have a method called
        # "_modify_calculate" that takes the calculate method as an
        # argument and returns a wrapped method.
        def wrapped(
                return_forces: bool = True,
                return_components: bool = True,
        ) -> Results:
            # Perform modifications before the calculation is performed.
            ...

            # Perform the calculation.
            results = calculate(return_forces, return_components)

            # Perform modifications to the results after the calculation
            # is performed.
            ...

            return results
        return wrapped


class TemplateIntegratorPlugin(IntegratorPlugin):
    """A class applying a templated modification to integrators.
    """

    # Perform initialization as needed.
    def __init__(self) -> None:
        ...

    def _modify_integrate(
            self,
            integrate: Callable[[System], Returns],
    ) -> Callable[[System], Returns]:
        """Modify the integrate routine with template.

        Args:
            integrate: The integration routine to modify.

        Returns:
            The modified integration routine which implements the
            templated modification.
        """
        # All integrator plugins must modify an Integrator object.  This
        # is achieved by "wrapping" the integrate and/or
        # compute_kinetic_energy routines with added functionality.  The
        # actual wrapping is applied internally, so the plugin only
        # needs to have a method called "_modify_calculate" and/or
        # "_modify_compute_kinetic_energy" that takes the method as an
        # argument and returns a wrapped method.
        def wrapped(system: System) -> Returns:
            # Perform modifications before the integration is performed.
            ...

            # Perform the integration.
            positions, velocities = integrate(system)

            # Perform modifications to the positions and velocities
            # after the integration is performed.
            ...

            return positions, velocities
        return wrapped

    def _modify_compute_kinetic_energy(
            self,
            compute_kinetic_energy: Callable[[System], float],
    ) -> Callable[[System], float]:
        """Modify the kinetic energy computation with template.

        Args:
            compute_kinetic_energy: The kinetic energy routine to
                modify.

        Returns:
            The modified kinetic energy routine which implements the
            templated modification.
        """
        def wrapped(system: System) -> Returns:
            # Perform modifications before the computation is performed.
            ...

            # Perform the calculation.
            energy = compute_kinetic_energy(system)

            # Perform modifications to the kinetic energy after the
            # computation is performed.
            ...

            return energy
        return wrapped
