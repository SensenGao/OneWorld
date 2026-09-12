"""Attach the RGB and 3D Gaussian decoder heads to the base RAE."""

from __future__ import annotations

from oneworld.rae.heads import RAEHeads
from oneworld.rae.model_base import BaseRAE


class ReconstructionRAE(BaseRAE):
    def __init__(self, pi3x, *args, detail_dim=32, detail_hidden_dim=0,
                 detail_merge="feature", **kwargs):
        kwargs["allow_variable_appearance_dim"] = True
        super().__init__(pi3x, *args, **kwargs)
        self.heads = RAEHeads(
            geometry_dim=self.geometry_dim,
            appearance_dim=self.appearance_dim,
            features=int(kwargs.get("features", 256)),
            patch_size=pi3x.patch_size,
            level_drop_prob=float(kwargs.get("level_drop_prob", 0.0)),
            sh_degree=self.sh_degree,
            detail_dim=detail_dim,
            detail_hidden_dim=detail_hidden_dim,
            detail_merge=detail_merge,
        )
