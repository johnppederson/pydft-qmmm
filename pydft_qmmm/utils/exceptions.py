r"""A module containing PyDFT-QMMM exception classes.
"""
from __future__ import annotations

__all__ = [
    "DependencyImportError",
    "PluginImportError",
    "InterfaceImportError",
    "TheoryLevelError",
]

from abc import ABC, abstractmethod
import textwrap

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydft_qmmm.utils import TheoryLevel


class PyDFTQMMMException(Exception):
    """Base exception class for PyDFT-QMMM."""
    pass


class DependencyImportError(PyDFTQMMMException):
    """An exception for failing to import optional dependencies.

    Args:
        dependency: The optional dependency attempting to be imported.
        functionality: A description of the functionality associated
            with the optional dependency.
        website: The project page for the optional dependency.
    """

    def __init__(
            self,
            dependency: str,
            functionality: str,
            website: str,
    ) -> None:
        super().__init__(
            ("Unable to import optional dependency\n  "
             + textwrap.fill(f"'{dependency}' for {functionality}.")
             + "\n\n"
             + textwrap.fill(f"Please check the '{dependency}' project "
                             "page for installation instructions:")
             + f"\n\n{website}\n\n"
             + textwrap.fill(f"If '{dependency}' is no longer "
                             "maintained, please contact the "
                             "maintainers of PyDFT-QMMM:")
             + "\n\nhttps://github.com/johnppederson/pydft-qmmm/issues"),
        )


class AssetImportError(PyDFTQMMMException, ABC):
    """An exception for failing to import a managed asset.

    Args:
        name: The name of the asset that the user attempted to import.
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            (f"Unable to import {self.type_} '{name}'.\n\n"
             + textwrap.fill(f"If the '{name}' {self.type_} is a "
                             "third-party project, please find the "
                             "project page on github for "
                             "installation instructions.  This will "
                             "most commonly involve cloning the "
                             "repository, navigating to the repository "
                             "root directory, and calling `pip install "
                             ".`, as in the pip documentation:")
             + "\n\nhttps://packaging.python.org/en/latest/tutorials/"
             + "installing-packages/#installing-from-a-local-src-tree"
             + "\n\n"
             + textwrap.fill(f"If the '{name}' {self.type_} is "
                             "included as part of PyDFT-QMMM, please "
                             "contact the maintainers of PyDFT-QMMM:")
             + "\n\nhttps://github.com/johnppederson/pydft-qmmm/issues"),
        )

    @property
    @abstractmethod
    def type_(self) -> str:
        """The type of asset that the user attempted to import."""
        pass


class PluginImportError(AssetImportError):
    """An exception for failing to import a plugin.

    Args:
        name: The name of the asset that the user attempted to import.
    """

    @property
    def type_(self) -> str:
        """The type of asset that the user attempted to import."""
        return "plugin"


class InterfaceImportError(AssetImportError):
    """An exception for failing to import an interface.

    Args:
        name: The name of the asset that the user attempted to import.
    """

    @property
    def type_(self) -> str:
        """The type of asset that the user attempted to import."""
        return "interface"


class TheoryLevelError(PyDFTQMMMException):
    """An exception for mismatches in TheoryLevel.

    Args:
        correct_theory_level: The expected TheoryLevel for the given
            functionality.
        actual_theory_level: The received TheoryLevel for the given
            functionality.
        functionality: A description of the functionality impacted by
            an erroneous TheoryLevel.
        solution: A description of a proposed solution to avoid the
            exception.
    """

    def __init__(
            self,
            correct_theory_level: TheoryLevel,
            actual_theory_level: TheoryLevel,
            functionality: str,
            solution: str = "",
    ) -> None:
        super().__init__(
            ("Mismatch in expected and received TheoryLevel.\n\n"
             + textwrap.fill(f"Expected '{correct_theory_level}', but "
                             f"got '{actual_theory_level}' for "
                             f"{functionality}.")
             + "\n\n"
             + textwrap.fill(f"{solution}")),
        )
