"""Representation autoencoder components for OneWorld."""

__all__ = [
    "BOUNDARY_BLOCK",
    "RAE",
    "RAEEncoder",
    "RAELatentEncoder",
    "canonicalize_cameras",
    "canonicalize_pi3_cameras",
]

_EXPORTS = {
    "BOUNDARY_BLOCK": ("encoder", "BOUNDARY_BLOCK"),
    "RAEEncoder": ("encoder", "RAEEncoder"),
    "RAE": ("model", "RAE"),
    "RAELatentEncoder": ("model", "RAELatentEncoder"),
    "canonicalize_cameras": ("camera", "canonicalize_cameras"),
    "canonicalize_pi3_cameras": ("camera", "canonicalize_pi3_cameras"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
