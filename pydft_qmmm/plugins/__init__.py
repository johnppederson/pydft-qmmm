"""A sub-package for defining and dynamically loading plugins.
"""
from __future__ import annotations

__author__ = "John Pederson"


from .atom_partition import *
from .center import *
from .centroid_partition import *
from .firstatom_partition import *
from .plumed import *
from .rigid import *
from .settle import *
from .wrap import *

from .plugin_manager import get_plugins
from .plugin_manager import Plugin

globals().update(get_plugins())

del get_plugins
