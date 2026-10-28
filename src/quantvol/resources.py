"""Files shipped inside the package: example configs and the bundled SPY sample."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

BUILTIN_PREFIX = "builtin:"
_DIR = Path(str(files("quantvol"))) / "assets"


def resource_path(name: str) -> Path:
    """Absolute path of a bundled file, e.g. ``resource_path("spy.yaml")``."""
    path = _DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"no bundled resource named {name!r}; available: {available()}")
    return path


def available() -> list[str]:
    """Names of all bundled files."""
    return sorted(p.name for p in _DIR.iterdir() if p.is_file())


def resolve_data_path(value: str) -> Path:
    """Map ``builtin:<file>`` to the bundled file; any other value is a normal path."""
    if value.startswith(BUILTIN_PREFIX):
        return resource_path(value[len(BUILTIN_PREFIX) :])
    return Path(value)


def resolve_config_path(value: str) -> Path:
    """Return an existing config file, or the bundled config of that name (spy, default)."""
    path = Path(value)
    if path.is_file():
        return path
    bundled = _DIR / f"{value}.yaml"
    if bundled.is_file():
        return bundled
    names = ", ".join(p.stem for p in _DIR.glob("*.yaml"))
    raise FileNotFoundError(f"config {value!r} not found as a file or bundled name ({names})")
