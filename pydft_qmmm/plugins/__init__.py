"""A sub-package for defining and dynamically loading plugins.
"""
from __future__ import annotations

from .atom_partition import AtomPartition
from .center import CalculatorCenter
from .center import IntegratorCenter
from .centroid_partition import CentroidPartition
from .firstatom_partition import FirstAtomPartition
from .plugin import Plugin
from .plugin_manager import get_external_plugins
from .plumed import Plumed
from .rigid import RigidBody
from .rigid import Stationary
from .settle import SETTLE
from .wrap import CalculatorWrap
from .wrap import IntegratorWrap

globals().update(get_external_plugins())

del get_external_plugins

__author__ = "John Pederson"
