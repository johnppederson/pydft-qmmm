from __future__ import annotations

from setuptools import find_packages
from setuptools import setup


setup(
    name="interface_template",
    packages=find_packages(where="interface_template"),
    # This is how PyDFT-QMMM can find the templated interface, so every
    # local or third-party interface must have entry_points defined for
    # the key "pydft_qmmm.interfaces" with a list[str] value that has
    # the name of the interface assigned to a value (it does not matter
    # what value it is assigned to).
    entry_points={
        "pydft_qmmm.interfaces": ["interface_template = None"],
    },
)
