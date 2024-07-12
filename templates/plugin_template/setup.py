from __future__ import annotations

from setuptools import find_packages
from setuptools import setup


setup(
    name="plugin_template",
    packages=find_packages(where="plugin_template"),
    # This is how PyDFT-QMMM can find the templated plugin, so every
    # local or third-party plugin must have entry_points defined for
    # the key "pydft_qmmm.plugins" with a list[str] value that has
    # the name of the plugin assigned to a value (it does not matter
    # what value it is assigned to).
    entry_points={
        "pydft_qmmm.plugins": ["plugin_template = None"],
    },
)
