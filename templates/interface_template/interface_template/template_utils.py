#! /usr/bin/env python3
"""A module to define operations for building the TemplateInterface class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .template_interface import TemplateInterface

if TYPE_CHECKING:
    from pydft_qmmm.interfaces.interface import SoftwareSettings


def template_interface_factory(settings: SoftwareSettings) -> TemplateInterface:
    """A function which constructs the TemplateInterface.

    :param settings: The SoftwareSettings object to build the
        template interface from.
    :return: The TemplateInterface.
    """
    # Perform any setup required for arguments to the TemplateInterface
    # __init__() call based on the provided settings.
    ...

    # Instantiate a TemplateInterface.
    wrapper = TemplateInterface(
        ...,
    )

    # Register observer functions.  Important geometry (position), property
    # (charge), and topology (residue, subsystem membership) are
    # communicated to the interfaces through an Observer design pattern,
    # so the Interface class should have functions which update the state
    # of the external software upon receiving updated coordinates, charges,
    # etc.  These "update" functions are registered here, in the factory
    # method.  Update functions can be defined for more system datatypes
    # than are presented here.
    settings.system.charges.register_notifier(wrapper.update_charges)
    settings.system.positions.register_notifier(wrapper.update_positions)
    settings.system.subsystems.register_notifier(wrapper.update_subsystems)
    return wrapper
