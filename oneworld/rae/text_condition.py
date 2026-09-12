"""Precomputed Wan UMT5 scene-conditioning utilities.

The expensive frozen UMT5-XXL encoder is run once offline.  Training reads ragged
4096-D token features from memory-mappable safetensor shards, then applies Wan's
small native text projection online so that projection remains trainable.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import torch
from safetensors import safe_open


TEXT_STORE_FORMAT = "re10k-wan-umt5-raw-dual-v1"


class WanTextEmbeddingStore:
    """Random-access reader for ragged, scene-level UMT5 embeddings."""

    def __init__(self, manifest_path: str):
        self.manifest_path = os.path.abspath(manifest_path)
        with open(self.manifest_path) as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("format") != TEXT_STORE_FORMAT:
            raise ValueError(
                f"unsupported text store: {self.manifest.get('format')}")
        if int(self.manifest.get("embedding_dim", -1)) != 4096:
            raise ValueError("Wan UMT5 store must contain 4096-D embeddings")
        if self.manifest.get("caption_fields") != [
                "short_caption", "long_caption"]:
            raise ValueError("text store must contain both short and long captions")
        if self.manifest.get("training_mix") != {
                "short_caption": 0.5, "long_caption": 0.5}:
            raise ValueError("text store must declare a 1:1 long/short training mix")
        entries = self.manifest.get("entries")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("text manifest has no scene entries")
        if int(self.manifest.get("total_scenes", -1)) != len(entries):
            raise ValueError("text manifest scene count does not match entries")
        self.entries = entries
        self.root = os.path.dirname(self.manifest_path)
        self._null = None

    def require_keys(self, keys: Iterable[str]):
        missing = [key for key in keys if key not in self.entries]
        if missing:
            raise KeyError(
                f"mandatory text embeddings missing for {len(missing)} scenes: "
                f"{missing[:4]}")

    @staticmethod
    def _read_slices(path: str, requests):
        values = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor = handle.get_slice("embeddings")
            for output_index, start, end in requests:
                value = tensor[int(start):int(end)]
                if value.ndim != 2 or value.shape[1] != 4096 or value.shape[0] < 1:
                    raise ValueError(
                        f"invalid text slice {tuple(value.shape)} in {path}")
                if not bool(torch.isfinite(value.float()).all()):
                    raise FloatingPointError(f"non-finite text embedding in {path}")
                values[output_index] = value
        return values

    def null_embedding(self):
        if self._null is None:
            null_file = self.manifest.get("null_embedding")
            if not null_file:
                raise ValueError("text manifest has no empty-prompt embedding")
            path = os.path.join(self.root, null_file)
            with safe_open(path, framework="pt", device="cpu") as handle:
                self._null = handle.get_tensor("embeddings")
            if (self._null.ndim != 2 or self._null.shape[1] != 4096
                    or self._null.shape[0] < 1
                    or not bool(torch.isfinite(self._null.float()).all())):
                raise ValueError("invalid empty-prompt UMT5 embedding")
        return self._null

    def load_batch(self, keys, variants, keep=None):
        """Return padded CPU BF16 features and true token lengths.

        ``keep`` selects real captions per sample.  Dropped samples use the official
        UMT5 embedding of an empty prompt, rather than an arbitrary all-zero vector.
        """
        keys = list(keys)
        self.require_keys(keys)
        variants = list(variants)
        if len(variants) != len(keys) or any(
                value not in {"short_caption", "long_caption"}
                for value in variants):
            raise ValueError("each scene requires a short_caption/long_caption choice")
        if keep is None:
            keep = [True] * len(keys)
        else:
            keep = [bool(value) for value in keep]
        if len(keep) != len(keys):
            raise ValueError("text keep mask and scene keys have different lengths")

        values = [None] * len(keys)
        grouped = {}
        null = None
        for output_index, (key, variant, retained) in enumerate(
                zip(keys, variants, keep)):
            if not retained:
                null = self.null_embedding() if null is None else null
                values[output_index] = null
                continue
            entry = self.entries[key]
            if not isinstance(entry, dict) or variant not in entry:
                raise KeyError(f"{key} has no mandatory {variant} embedding")
            shard, start, end = entry[variant]
            grouped.setdefault(shard, []).append((output_index, start, end))
        for shard, requests in grouped.items():
            values_by_index = self._read_slices(
                os.path.join(self.root, shard), requests)
            for output_index, value in values_by_index.items():
                values[output_index] = value

        lengths = torch.tensor([value.shape[0] for value in values], dtype=torch.long)
        padded = torch.zeros(
            len(values), int(lengths.max()), 4096, dtype=torch.bfloat16)
        for index, value in enumerate(values):
            padded[index, :value.shape[0]].copy_(value.to(torch.bfloat16))
        return padded, lengths
