"""Wan modules with lazy exports to keep CPU-side tooling lightweight."""

__all__ = ["WanModel", "T5EncoderModel"]


def __getattr__(name):
    if name == "WanModel":
        from .model import WanModel

        return WanModel
    if name == "T5EncoderModel":
        from .t5 import T5EncoderModel

        return T5EncoderModel
    raise AttributeError(name)
