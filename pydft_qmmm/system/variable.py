#! /usr/bin/env python3
"""A module defining the :class:`Record` and :class:`Variable` classes.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import EllipsisType
from typing import Any
from typing import Generic
from typing import TYPE_CHECKING
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing import Callable
    from typing import TypeAlias

T = TypeVar("T")

DT = TypeVar("DT", covariant=True, bound=np.dtype[Any])
ST = TypeVar("ST", bound=Any)

DT2 = TypeVar("DT2", bound=np.dtype[Any])
ST2 = TypeVar("ST2", bound=Any)

ArrayLikeInt: TypeAlias = (
    int | np.integer[Any] | Sequence[int | np.integer[Any]]
    | Sequence[Sequence[Any]] | NDArray[Any]
)

Index: TypeAlias = (
    ArrayLikeInt | slice | EllipsisType
    | tuple[ArrayLikeInt | slice | EllipsisType]
)

array_float: TypeAlias = np.dtype[np.float64]
array_int: TypeAlias = np.dtype[np.int32]
array_str: TypeAlias = np.dtype[np.str_]
array_obj: TypeAlias = np.dtype[np.object_]


class ObservedArray(np.ndarray[ST, DT]):
    base: np.ndarray[ST, DT]
    _notifiers: list[Callable[[np.ndarray[ST, DT]], None]]

    def __new__(cls, array: np.ndarray[ST, DT]) -> ObservedArray[ST, DT]:
        obj = np.asarray(array).view(cls)
        obj._notifiers = []
        return obj

    def __setitem__(
            self,
            key: Index,
            value: Any,
    ) -> None:
        super().__setitem__(key, value)
        for notify in self._notifiers:
            notify(self.base)

    def __array_wrap__(
            self,
            array: np.ndarray[ST2, DT2],
            context: tuple[np.ufunc, tuple[Any, ...], int] | None = None,
    ) -> np.ndarray[ST2, DT2]:
        if array is self:
            for notify in self._notifiers:
                notify(self.base)
            return array
        return array.view(np.ndarray)

    def __array_finalize__(self, array: None | np.ndarray[Any, Any]) -> None:
        if array is None:
            return
        self._notifiers = getattr(array, "_notifiers", [])

    def register_notifier(
            self,
            notifier: Callable[[np.ndarray[ST, DT]], None],
    ) -> None:
        self._notifiers.append(notifier)


@dataclass
class ArrayValue(Generic[T]):
    _array: ObservedArray[Any, array_float | array_int | array_str | array_obj]
    _key: int

    def update(self, value: T) -> None:
        self._array[self._key] = value

    def value(self) -> T:
        return self._array[self._key]
