"""Grid-based electrostatic embedding physics shared by QM backends.

Pure numpy.  ``grid_potential`` must stay importable by the Python
interpreter embedded inside VASP, so nothing here may import OpenMM,
ASE, scipy, or the rest of PyDFT-QMMM.
"""
from __future__ import annotations
