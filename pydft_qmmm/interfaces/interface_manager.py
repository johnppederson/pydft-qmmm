"""Functionality for importing package and external interfaces.

Attributes:
    MODULE_PATH: The directory where the PyDFT-QMMM interfaces
        sub-package is installed.
    DISCOVERED_INTERFACES: A list of entry points into the interface
        architecture of PyDFT-QMMM within installed package metadata.
"""
from __future__ import annotations

from configparser import ConfigParser
from importlib import import_module
from importlib.metadata import entry_points
from importlib.resources import files
from os import listdir
from typing import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import SoftwareSettings, SoftwareInterface
    Factory = Callable[[SoftwareSettings], SoftwareInterface]

MODULE_PATH = files("pydft_qmmm") / "interfaces"

try:
    DISCOVERED_INTERFACES: set[str] = {
        point.name for point
        in entry_points().get("pydft_qmmm.interfaces", [])
    }
except AttributeError:
    DISCOVERED_INTERFACES = entry_points(
        group="pydft_qmmm.interfaces",
    ).names


class _Checked:
    """Whether or not default settings have been checked."""
    CHECKED = False


def _get_factory(module_name: str, package_name: str) -> Factory:
    """Get a software interface factory method from a package/module.

    Args:
        module_name: The name of the interface module to load the
            factory method from.
        package_name: The name of the package containing an interface
            module.

    Returns:
        A factory method which builds a software interface from the
        specified package and module.
    """
    module = import_module(
        module_name, package=package_name,
    )
    return getattr(module, "FACTORY")


def _check_settings() -> None:
    """Check the default interfaces and set them to active.
    """
    config = ConfigParser()
    config.read(str(MODULE_PATH / "interfaces.conf"))
    qm_interface = config["DEFAULT"]["QMSoftware"].lower()
    mm_interface = config["DEFAULT"]["MMSoftware"].lower()
    config.set("ACTIVE", "QMSoftware", qm_interface)
    config.set("ACTIVE", "MMSoftware", mm_interface)
    with open(str(MODULE_PATH / "interfaces.conf"), "w") as fh:
        config.write(fh)


def get_software_factory(field: str) -> Factory:
    """Get a factory method according to the interfaces in ``interfaces.conf``.

    Fields include 'MMSoftware' or 'QMSoftware'.

    Args:
        field: The field of the interfaces configuration file to
            extract a factory method for.

    Returns:
        A factory method which builds a software interface for the
        specified field.
    """
    if not _Checked.CHECKED:
        _check_settings()
        _Checked.CHECKED = True
    config = ConfigParser()
    config.read(str(MODULE_PATH / "interfaces.conf"))
    software_name = config["ACTIVE"][field].lower()
    local_names = listdir(str(MODULE_PATH))
    package_names = [name for name in DISCOVERED_INTERFACES]
    found = False
    for name in local_names:
        if found:
            break
        if software_name == name:
            factory = _get_factory(
                ".interfaces." + name.split(".")[0], "pydft_qmmm",
            )
            found = True
    for name in package_names:
        if found:
            break
        if software_name == name:
            factory = _get_factory(name, name)
            found = True
    return factory


def set_interfaces(
        qm_interface: str | None = "Psi4",
        mm_interface: str | None = "OpenMM",
) -> None:
    """Set the active QM and MM interfaces in ``interfaces.conf``.

    Args:
        qm_interface: The name of the QM interface to use.
        mm_interface: The name of the MM interface to use.
    """
    config = ConfigParser()
    config.read(str(MODULE_PATH / "interfaces.conf"))
    config.set("ACTIVE", "QMSoftware", qm_interface)
    config.set("ACTIVE", "MMSoftware", mm_interface)
    with open(str(MODULE_PATH / "interfaces.conf"), "w") as fh:
        config.write(fh)


def set_default_interfaces(
        qm_interface: str | None = "Psi4",
        mm_interface: str | None = "OpenMM",
) -> None:
    """Set the default QM and MM interfaces in ``interfaces.conf``.

    Args:
        qm_interface: The name of the new default QM interface.
        mm_interface: The name of the new default MM interface.
    """
    config = ConfigParser()
    config.read(str(MODULE_PATH / "interfaces.conf"))
    config.set("DEFAULT", "QMSoftware", qm_interface)
    config.set("DEFAULT", "MMSoftware", mm_interface)
    with open(str(MODULE_PATH / "interfaces.conf"), "w") as fh:
        config.write(fh)
    _check_settings()
