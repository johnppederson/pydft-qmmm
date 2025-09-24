"""Data container classes for implementing the observer pattern.
"""
from __future__ import annotations

__all__ = [
    "ObservedArray",
    "ArrayValue",
    "observed_class",
]

from dataclasses import dataclass
from typing import Any
from typing import Generic
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast
from typing import overload

import numpy

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

T = TypeVar("T")
Class = TypeVar("Class")

DT = TypeVar("DT", covariant=True, bound=numpy.dtype)
ST = TypeVar("ST", bound=tuple[int, ...])

DT2 = TypeVar("DT2", bound=numpy.dtype)
ST2 = TypeVar("ST2", bound=tuple[int, ...])

array_float: TypeAlias = numpy.dtype[numpy.float64]
array_int: TypeAlias = numpy.dtype[numpy.int64]
array_str: TypeAlias = numpy.dtype[numpy.str_]
array_obj: TypeAlias = numpy.dtype[numpy.object_]


class ObservedArray(numpy.ndarray[ST, DT]):
    """A data container for arrays implementing the observer pattern.

    Attributes:
        base: The actual array data
        _notifiers: Functions that are called when a value of the
            array is changed.
    """
    base: numpy.ndarray[ST, DT]
    _notifiers: list[Callable[[numpy.ndarray[ST, DT]], None]]

    def __new__(cls, array: numpy.ndarray[ST, DT]) -> ObservedArray[ST, DT]:
        """Generate a new observed array object.

        Args:
            array: The Numpy array data from which to make an observed
                array.

        Returns:
            An observed array containing the same data as was input.
        """
        obj = numpy.asarray(array).view(cls)
        obj._notifiers = []
        return obj

    def __setitem__(self, key: Any, value: Any) -> None:
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
            array: numpy.ndarray[ST2, DT2],
            context: tuple[numpy.ufunc, tuple[Any, ...], int] | None = None,
            return_scalar: bool = False,
    ) -> numpy.ndarray[ST2, DT2]:
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
        return cast(numpy.ndarray[ST2, DT2], array.view(numpy.ndarray))

    def __array_finalize__(
            self,
            array: None | numpy.ndarray[Any, Any],
    ) -> None:
        """Finalize the instantiation of an array.

        Args:
            array: The array being instantiated.
        """
        if array is None:
            return
        self._notifiers = getattr(array, "_notifiers", [])

    def register_notifier(
            self,
            notifier: Callable[[numpy.ndarray[ST, DT]], None],
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
    _array: ObservedArray[
        Any,
        array_float | array_int | array_str | array_obj,
    ]
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


@overload
def observed_class(*args: type[Class], **kwargs: Any) -> type[Class]:
    ...


@overload
def observed_class(
        *args: None,
        **kwargs: Any,
) -> Callable[[type[Class]], type[Class]]:
    ...


def observed_class(*args, **kwargs):
    """Modify a class to have protected observed properties.

    Args:
        args: An optional class to modify.
        kwargs: Optional dataclass settings.

    Returns:
        Either the modified dataclass with added property getters and
        setters, or the function to generate the modified dataclass.
    """
    def modify_class(cls: type[Class]) -> type[Class]:
        """Modify a class to have protected observed properties.

        Args:
            cls: The class to modify.

        Returns:
            The modified dataclass with added property getters and
            setters.
        """
        cls = dataclass(cls, **kwargs)

        def value_getter(name: str) -> Callable[[Class], Any]:
            return lambda obj: getattr(obj, name).value()

        def value_setter(name: str) -> Callable[[Class, Any], None]:
            return lambda obj, val: getattr(obj, name).update(val)

        def array_getter(name: str) -> Callable[[Class], Any]:
            return lambda obj: getattr(obj, name)

        def array_setter(name: str) -> Callable[[Class, Any], None]:
            return lambda obj, arr: getattr(obj, name).__setitem__(
                slice(None),
                arr,
            )

        for name, field in getattr(cls, "__dataclass_fields__").items():
            _name = "_" + name
            if "ObservedArray" in field.type:
                property_ = property(array_getter(_name), array_setter(_name))
            elif "ArrayValue" in field.type:
                property_ = property(value_getter(_name), value_setter(_name))
            else:
                continue
            setattr(cls, name, property_)
        return cls
    if len(args) == 1:
        return modify_class(args[0])
    if len(args) > 1:
        raise TypeError
    return modify_class
