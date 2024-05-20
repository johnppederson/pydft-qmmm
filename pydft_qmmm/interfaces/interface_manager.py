#! /usr/bin/env python3
"""A module for handling software interface imports.

.. warning:: MyPy is not currently happy with this module.
"""
from __future__ import annotations

from configparser import ConfigParser
from importlib import import_module
from importlib.resources import files
from os import listdir
from typing import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import SoftwareSettings, SoftwareInterface
    from types import ModuleType
    Factory = Callable[[SoftwareSettings], SoftwareInterface]

MODULE_PATH = files("pydft_qmmm") / "interfaces"


def _import(module_name: str) -> ModuleType:
    """Import an module from the pydft_qmmm.interfaces subpackage.

    :param module_name: The name of the interfaces module to import.
    :return: The imported module.
    """
    module = import_module(
        ".interfaces." + module_name.split(".")[0], package="pydft_qmmm",
    )
    return module


def _get_factory(module_name: str) -> Factory:
    """Get the FACTORY dictionary from a module in the
    pydft_qmmm.interfaces subpackage.

    :param module_name: The name of the interfaces module to extract the
        FACTORY dictionary from.
    :return: The FACTORY dictionary from the specified module.
    """
    return getattr(_import(module_name), "FACTORY")


def get_software_factory(field: str) -> Factory:
    """Get the FACTORY dictionary for the specified field of the
    interfaces configuration file.  Fields include 'MMSoftware' or
    'QMSoftware'.

    :param field: The field of the interfaces configuration file to
        extract a FACTORY dictionary for.
    :return: The FACTORY dictionary for the specified field of the
        interfaces configuration file.
    """
    config = ConfigParser()
    config.read(str(MODULE_PATH / "interfaces.conf"))
    software_name = config["DEFAULT"][field].lower()
    file_names = listdir(str(MODULE_PATH))
    for name in file_names:
        if software_name in name:
            factory = _get_factory(name)
    return factory
