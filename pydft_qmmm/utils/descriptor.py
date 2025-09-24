"""Helpful descriptors for caching and plugin loading.
"""
from __future__ import annotations

__all__ = [
    "system_cache",
    "pluggable_method",
]

from abc import ABC
from abc import abstractmethod
from functools import lru_cache
from functools import reduce
from typing import Generic
from typing import overload
from typing import TypeVar
from typing import ParamSpec
from typing import Concatenate
from typing import Callable

Class = TypeVar("Class")

P = ParamSpec("P")
R = TypeVar("R")

Bound = Callable[P, R]
Unbound = Callable[Concatenate[Class, P], R]


class _Descriptor(Generic[Class, P, R], ABC):
    """The base class for a Python descriptor that wraps a method.

    Args:
        method: The unbound method wrapped by the descriptor.
    """

    def __init__(self, method: Unbound[Class, P, R]) -> None:
        self.method = method
        self.wrapped_name = f"__wrapped_{method.__name__}"

    @overload
    def __get__(self, obj: None, owner: type[Class]) -> Unbound[Class, P, R]:
        pass

    @overload
    def __get__(self, obj: Class, owner: type[Class] | None) -> Bound[P, R]:
        pass

    @abstractmethod
    def __get__(self, obj, owner=None):
        """Get the method.

        Args:
            obj: An instance of the class containing the method.
            owner: The class containing the method.

        Returns:
            Either the bound or unbound method.
        """
        pass


def system_cache(
        *attributes: str,
        obj_is_system: bool = False
) -> Callable[[Unbound[Class, P, R]], _Descriptor]:
    """Create a cached method that resets when the system changes.

    Args:
        attributes: The name of the system attributes for which the
            cache will reset.
        obj_is_system: Whether the object is the system or merely
            contains a system object.

    Returns:
        A function that takes a bound method and returns a descriptor
        object that implements the caching.
    """
    class _SystemCache(_Descriptor):
        """A descriptor class that creates a cached version of a method.
        """

        def __get__(self, obj, owner=None):
            """Get the method.

            Args:
                obj: An instance of the class containing the method.
                owner: The class containing the method.

            Returns:
                Either the bound or unbound method.
            """
            if obj is None:
                return self.method
            if not hasattr(obj, self.wrapped_name):
                bound = self.method.__get__(obj, owner)
                wrapped = lru_cache(bound)
                object.__setattr__(obj, self.wrapped_name, wrapped)
                if obj_is_system:
                    system = obj
                else:
                    system = getattr(obj, "system", None)
                for attr in attributes:
                    observed = getattr(system, attr, None)
                    if observed is not None:
                        observed.register_notifier(
                            lambda _: wrapped.cache_clear(),
                        )
            return getattr(obj, self.wrapped_name)
    return lambda func: _SystemCache(func)


class _PluggableMethod(_Descriptor):
    """A descriptor class that applies plugins to a method.
    """
    plugin_no = 0

    def __get__(self, obj, owner=None):
        """Get the method.

        Args:
            obj: An instance of the class containing the method.
            owner: The class containing the method.

        Returns:
            Either the bound or unbound method.
        """
        if obj is None:
            return self.method
        if (not hasattr(obj, self.wrapped_name)
                or self.plugin_no != len(getattr(obj, "_plugins"))):
            wrapped = self.wrap(obj, owner)
            self.plugin_no = len(getattr(obj, "_plugins"))
            object.__setattr__(obj, self.wrapped_name, wrapped)
        return getattr(obj, self.wrapped_name)

    def wrap(self, obj: Class, owner: type[Class] | None) -> Bound[P, R]:
        """Wrap the bound method in plugins in reverse order.

        Args:
            obj: An instance of the class containing the method.
            owner: The class containing the method.

        Returns:
            The wrapped bound method.
        """
        bound = self.method.__get__(obj, owner)
        initial_wrappers = map(
            lambda x: getattr(x, "_modify_" + bound.__name__, None),
            getattr(obj, "_plugins"),
        )
        wrappers = tuple(filter(None, initial_wrappers))
        wrapped = reduce(lambda y, z: z(y), wrappers[-1::-1], bound)
        return wrapped


def pluggable_method(func: Unbound[Class, P, R]) -> _PluggableMethod:
    """Create a pluggable method that applies plugins in reverse order.

    Args:
        func: The method to wrap.

    Returns:
        A descriptor class that can be updated with additional wrapper
        functions, which are applied in reverse order to ensure that
        the first wrapper may apply the outermost changes.
    """
    return _PluggableMethod(func)
