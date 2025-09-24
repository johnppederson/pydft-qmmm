"""Functionality for building the Template interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .template_interface import TemplatePotential

if TYPE_CHECKING:
    from typing import Any


def template_interface_factory(
        system: System,
        *args: Any,
        **kwargs: Any,
) -> TemplatePotential:
    r"""Build the interface to Template.

    Args:
        system: The system which will be tied to the Template interface.
        args: Any positional arguments required to make the Template
            interface.
        kwargs: Any keyword arguments required to make the Template
            interface.

    Returns:
        The Template interface.
    """
    # Perform any setup required for the Template interface.
    ...

    # Instantiate a TemplatePotential with any requisite arguments.
    wrapper = TemplatePotential(system)

    # Register observer functions.  Important geometry (position), property
    # (charge), and topology (residue, subsystem membership) are
    # communicated to the interfaces through an Observer design pattern,
    # so the Interface class should have functions which update the state
    # of the external software upon receiving updated coordinates, charges,
    # etc.  These "update" functions are registered here, in the factory
    # method.  Update functions can be defined for more system datatypes
    # than are presented here.
    ...
    return wrapper
