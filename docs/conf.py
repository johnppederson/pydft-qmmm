# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# General settings.
project = 'PyDFT-QMMM'
copyright = '2024-2025, John Pederson, Jesse McDaniel'
author = 'John Pederson, Jesse McDaniel'
try:
    version = version("pydft-qmmm")
except PackageNotFoundError:
    version = ''
release = ''
language = 'en'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Abbreviations.
rst_epilog = r'''
.. |alpha| replace:: :math:`\mathrm{\alpha}`
.. |beta| replace:: :math:`\mathrm{\beta}`
.. |gamma| replace:: :math:`\mathrm{\gamma}`
.. _VMD selection language:  https://www.ks.uiuc.edu/Research/vmd/vmd-1.3/ug/node132.html
.. _Psi4 options: https://psicode.org/psi4manual/master/autodir_options_c/module__scf.html
'''

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_static_path = ['_static']

source_suffix = '.rst'
master_doc = 'index'

nitpicky = True
nitpick_ignore = [
    ("py:class", "ST"),
    ("py:class", "DT"),
    ("py:class", "pydft_qmmm.utils.descriptor._Descriptor"),
    ("py:class", "pydft_qmmm.utils.descriptor._PluggableMethod"),
    ("py:class", "pydft_qmmm.system.atom._SystemAtom"),
    ("py:obj", "pydft_qmmm.system.variable.T"),
    ("py:obj", "pydft_qmmm.system.variable.ST"),
    ("py:obj", "pydft_qmmm.system.variable.DT"),
]

# Extensions to use in this project.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.extlinks',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
]

# External links in the documentation.
extlinks = {
    'testcase': ('https://github.com/johnppederson/pydft-qmmm/tree/master/tests/%s', 'test %s'),
    'example': ('https://github.com/johnppederson/pydft-qmmm/tree/master/examples/case_%s', 'example case %s'),
    'openmm': ('https://github.com/openmm/openmm/tree/master/platforms/reference/src/%s', '%s'),
}

# Configure autodoc settings for type-hints.
autodoc_typehints = 'description'
autodoc_default_options = {
    "show-inheritance": True,
}

# Configure Napoleon parsing settings.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Configure sphinx-autodoc-typehints
typehints_defaults = 'comma'

# Configure autosummary.
autosummary_generate = True

# Add custom references to the inventory.
from sphinx.ext.intersphinx import InventoryAdapter
from sphinx.util.inventory import _InventoryItem


def add_ndarray_alias(app):
    inv = InventoryAdapter(app.builder.env)
    numpy_inv = inv.named_inventory["np"]
    np_kwargs = {
        "project_name": numpy_inv["py:class"]["numpy.ndarray"].project_name,
        "project_version": numpy_inv["py:class"]["numpy.ndarray"].project_version,
        "display_name": numpy_inv["py:class"]["numpy.ndarray"].display_name,
    }
    inv.main_inventory["py:class"]["numpy.float64"] = _InventoryItem(
        uri='https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.float64',
        **np_kwargs,
    )
    inv.main_inventory["py:class"]["numpy.int64"] = _InventoryItem(
        uri='https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.int64',
        **np_kwargs,
    )


def setup(app):
    app.connect("builder-inited", add_ndarray_alias)


# Provide intersphinx with links to third-party documentation.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'np': ('https://numpy.org/doc/stable/', None),
    'openmm': ('http://docs.openmm.org/latest/api-python/', None),
    'psi4': ('http://psicode.org/psi4manual/master/', None),
}

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False
