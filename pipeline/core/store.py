"""
Data Storage and Configuration Utilities.

This module provides the Store class, a recursive dictionary wrapper that
supports attribute-style access and nested path resolution (e.g., 'a.b.c').
"""

from collections.abc import MutableMapping
from typing import Any, Iterator


class Store(MutableMapping):
    """
    A recursive dot-accessible dictionary.

    Provides a convenient way to manage nested configurations or results
    where keys can be accessed as attributes. Intermediate levels are
    automatically wrapped in Store instances.
    """

    def __init__(self, initial: dict[str, Any] | None = None, strict: bool = True):
        """
        Initializes the Store.

        Args:
            initial: Optional dictionary to populate the store.
            strict: If True, accessing non-existent keys via attributes raises AttributeError.
        """
        self._data: dict[str, Any] = {}
        self._is_strict = strict
        if initial:
            self.update(initial)

    # -------------------------------------------------------------------------
    # MutableMapping Implementation
    # -------------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Retrieves an item from the store."""
        if key not in self._data:
            if self._is_strict:
                raise KeyError(f"Key '{key}' not found in Store.")
            return None
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets an item, converting nested dictionaries to Store instances."""
        if isinstance(value, dict):
            # If the target is already a Store, merge the data; otherwise, create a new Store
            if key in self._data and isinstance(self._data[key], Store):
                self._data[key].update(value)
            else:
                self._data[key] = Store(value, strict=self._is_strict)
        else:
            self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Removes an item from the store."""
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Returns an iterator over the store keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Returns the number of top-level items in the store."""
        return len(self._data)

    # -------------------------------------------------------------------------
    # Attribute-Style Access
    # -------------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Enables access via dot-notation: store.key."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'Store' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Enables assignment via dot-notation: store.key = value."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    # -------------------------------------------------------------------------
    # Deep Path Utilities
    # -------------------------------------------------------------------------

    def deep_set(self, path: str, value: Any) -> None:
        """
        Sets a value using a dot-separated path (e.g., 'database.config.port').
        Intermediate levels are created as Store instances if they do not exist.
        """
        parts = path.split(".")
        current_node: Store = self

        for part in parts[:-1]:
            if part not in current_node or not isinstance(current_node[part], Store):
                current_node[part] = Store(strict=self._is_strict)
            current_node = current_node[part]

        current_node[parts[-1]] = value

    def get_path(self, path: str, default: Any = None) -> Any:
        """
        Retrieves a value using a dot-separated path.
        Returns the default value if any part of the path is missing.
        """
        parts = path.split(".")
        current: Any = self

        for part in parts:
            if not isinstance(current, Store) or part not in current:
                return default
            current = current[part]

        return current

    def as_dict(self) -> dict[str, Any]:
        """Recursively converts the Store and its children back into standard dictionaries."""
        return {
            key: (val.as_dict() if isinstance(val, Store) else val)
            for key, val in self._data.items()
        }

    def __repr__(self) -> str:
        """Return a string representation of the store's data."""
        return f"Store({repr(self.as_dict())})"

    def __bool__(self) -> bool:
        """Returns True if the store contains data."""
        return bool(self._data)
