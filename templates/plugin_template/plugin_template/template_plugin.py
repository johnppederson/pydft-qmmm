#! /usr/bin/env python3
"""A module defining the TemplatePlugin class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydft_qmmm.plugins.plugin import CalculatorPlugin

if TYPE_CHECKING:
    from pydft_qmmm.calculator import Calculator


class TemplatePlugin(CalculatorPlugin):
    """The TemplateInterface class.
    """

    # Perform initialization as needed.
    def __init__(
            self,
    ) -> None:
        ...

    def modify(
            self,
            calculator: Calculator,
    ) -> None:
        """A function which modifies a Calculator object.

        :param calculator: The Calculator whose behavior will be
            modified.
        """
        # All calculator plugins must modify a Calculator object, and
        # all integrator plugins must modify an Integrator object.  This
        # is usually achieved by "wrapping" the calculate or integrate
        # routines with added functionality and then reassigning the
        # routines in the Calculator or Integrator to the wrapped method.
        # First, the "_modifieds" list[str] of the plugin should be
        # updated with the name of the modified object.
        self._modifieds.append(type(calculator).__name__)
        ...
