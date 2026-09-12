"""3D Gaussian rendering utilities used by OneWorld."""

__all__ = ["render_diagonal_views", "render_views"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from oneworld.geometry import render

    value = getattr(render, name)
    globals()[name] = value
    return value
