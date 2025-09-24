"""A sub-package containing interfaces to external software.
"""
from __future__ import annotations

__author__ = "John Pederson"

from .interface import *
from .interface_manager import get_interfaces

interfaces = get_interfaces()

del get_interfaces
