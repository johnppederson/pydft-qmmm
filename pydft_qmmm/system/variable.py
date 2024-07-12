"""Data container classes for implementing the observer pattern.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import Generic
from typing import TYPE_CHECKING
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from types import EllipsisType
    from typing import Callable
    from typing import TypeAlias

    ArrayLikeInt: TypeAlias = (
        int | np.integer[Any] | Sequence[int | np.integer[Any]]
        | Sequence[Sequence[Any]] | NDArray[Any]
    )

    Index: TypeAlias = (
        ArrayLikeInt | slice | EllipsisType
        | tuple[ArrayLikeInt | slice | EllipsisType]
    )

T = TypeVar("T")

DT = TypeVar("DT", covariant=True, bound=np.dtype[Any])
ST = TypeVar("ST", bound=Any)

DT2 = TypeVar("DT2", bound=np.dtype[Any])
ST2 = TypeVar("ST2", bound=Any)

array_float: TypeAlias = np.dtype[np.float64]
array_int: TypeAlias = np.dtype[np.int32]
array_str: TypeAlias = np.dtype[np.str_]
array_obj: TypeAlias = np.dtype[np.object_]


class ObservedArray(np.ndarray[ST, DT]):
    """A data container for arrays implementing the observer pattern.

    Attributes:
        base: The actual array data
        _notifiers: Functions that are called when a value of the
            array is changed.
    """
    base: np.ndarray[ST, DT]
    _notifiers: list[Callable[[np.ndarray[ST, DT]], None]]

    def __new__(cls, array: np.ndarray[ST, DT]) -> ObservedArray[ST, DT]:
        """Generate a new observed array object.

        Args:
            array: The Numpy array data from which to make an observed
                array.

        Returns:
            An observed array containing the same data as was input.
        """
        obj = np.asarray(array).view(cls)
        obj._notifiers = []
        return obj

    def __setitem__(
            self,
            key: Index,
            value: Any,
    ) -> None:
        """Change a value of the observed array.

        Args:
            key: The index of the array corresponding to the value
                that will be updated.
            value: The new value for the array element corresponding
                to the index.
        """
        super().__setitem__(key, value)
        for notify in self._notifiers:
            notify(self.base)

    def __array_wrap__(
            self,
            array: np.ndarray[ST2, DT2],
            context: tuple[np.ufunc, tuple[Any, ...], int] | None = None,
    ) -> np.ndarray[ST2, DT2]:
        """Notify observers when operations are performed on the array.

        Args:
            array: The array which is being updated by an operation.
            context: The operation being performed on an array.

        Returns:
            A regular Numpy array resulting from the operation.
        """
        if array is self:
            for notify in self._notifiers:
                notify(self.base)
            return array
        return array.view(np.ndarray)

    def __array_finalize__(self, array: None | np.ndarray[Any, Any]) -> None:
        """Finalize the instantiation of an array.

        Args:
            array: The array being instantiated.
        """
        if array is None:
            return
        self._notifiers = getattr(array, "_notifiers", [])

    def register_notifier(
            self,
            notifier: Callable[[np.ndarray[ST, DT]], None],
    ) -> None:
        """Register observers to be notified when array data is edited.

        Args:
            notifier: A function which will be called when array data
                is edited.
        """
        self._notifiers.append(notifier)


@dataclass
class ArrayValue(Generic[T]):
    """A wrapper for single values from observed arrays.

    Args:
        _array: The observed array containing the value.
        _key: The index for the value in the observed array.
    """
    _array: ObservedArray[Any, array_float | array_int | array_str | array_obj]
    _key: int

    def update(self, value: T) -> None:
        """Update the value in the observed array.

        Args:
            value: The new value for the array element corresponding
                to the index.
        """
        self._array[self._key] = value

    def value(self) -> T:
        """Return the observed array value.

        Returns:
            The value of the array element corresponding to the index.
        """
        return self._array[self._key]
