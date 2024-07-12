"""A sub-package containing interfaces to external software.
"""
from __future__ import annotations

from .interface import MMInterface
from .interface import MMSettings
from .interface import QMInterface
from .interface import QMSettings
from .interface import SoftwareInterface
from .interface_manager import get_software_factory
from .interface_manager import set_default_interfaces
from .interface_manager import set_interfaces

mm_factory = get_software_factory("MMSoftware")
qm_factory = get_software_factory("QMSoftware")

del get_software_factory

__author__ = "John Pederson"
