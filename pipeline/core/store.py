from collections.abc import MutableMapping
from typing import Any, Iterator, Mapping, Optional


class Store(MutableMapping):
    """Dictionnaire récursif avec accès par attribut et chemins de type 'x.y.z'.
    Reprend (mal) l'idée de `Box` mais sans dépendance externe.

    Paramètres:
    - strict: si True, les accès à des clés inexistantes lèvent KeyError.

    """

    def __init__(self, initial: Optional[dict[str, Any]] = None, strict: bool = True):
        self._data: dict[str, Any] = {}
        self._strict = strict
        if initial:
            for k, v in initial.items():
                self[k] = v

    # ---------------------
    # MutableMapping
    # ---------------------
    def __getitem__(self, key: str) -> Any:
        if key not in self._data:
            if self._strict:
                raise KeyError(key)
            return None
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict):
            if key in self._data and isinstance(self._data[key], Store):
                self._data[key].update(value)
            else:
                self._data[key] = Store(value, strict=self._strict)
        else:
            self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # ---------------------
    # Accès par attribut
    # ---------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return self[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    # ---------------------
    # Méthodes utilitaires
    # ---------------------
    def deep_set(self, path: str, value: Any) -> None:
        """Assigne une valeur via un chemin 'x.y.z', crée les niveaux intermédiaires si nécessaire."""
        parts = path.split(".")
        node: Store = self
        for part in parts[:-1]:
            if part not in node._data or not isinstance(node._data[part], Store):
                node._data[part] = Store(strict=self._strict)
            node = node._data[part]
        node[parts[-1]] = value

    def get_path(self, path: str, default: Any = None) -> Any:
        """Récupère une valeur via un chemin 'x.y.z', retourne default si une étape n'existe pas."""
        parts = path.split(".")
        node: Any = self
        for part in parts:
            if not isinstance(node, Store):
                return default
            if part not in node._data:
                return default
            node = node._data[part]
        return node

    def update(self, values: Mapping[str, Any]) -> None:
        for k, v in values.items():
            self[k] = v

    def as_dict(self) -> dict[str, Any]:
        return {
            k: v.as_dict() if isinstance(v, Store) else v for k, v in self._data.items()
        }

    def __repr__(self) -> str:
        return repr(self.as_dict())

    def __bool__(self) -> bool:
        return bool(self._data)
