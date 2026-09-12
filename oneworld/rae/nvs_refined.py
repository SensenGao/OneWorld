"""NVS-Refined readers and exact mixed-data/multi-resolution sampling.

The public dataset stores DL3DV cameras as Nerfstudio JSON and ACID/Re10K cameras as
packed float32 OpenCV w2c matrices.  This module exposes one OpenCV c2w contract for
all three static subsets and deliberately never indexes SpatialVID.
"""

from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
STATIC_SUBSETS = ("ACID", "DL3DV", "Re10K")
DL3DV_CONTEXT_WARMUP_STEPS = 10_000
DL3DV_INITIAL_MIN_SPAN = 15
DL3DV_INITIAL_MAX_SPAN = 30
DL3DV_FINAL_MIN_SPAN = 20
DL3DV_FINAL_MAX_SPAN = 50
RAE_SOURCE_IDS = {
    "re10k": 0,
    "nvs_refined_acid": 1,
    "nvs_refined_dl3dv": 2,
    "nvs_refined_re10k": 3,
}


@dataclass(frozen=True)
class SceneLocation:
    parquet: str
    row_group: int
    subset: str


def _frame_number(name: str, fallback: int) -> int:
    digits = "".join(character for character in Path(name).stem if character.isdigit())
    return int(digits) if digits else int(fallback)


def _center_crop_resize(
    image: Image.Image,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> tuple[Image.Image, torch.Tensor]:
    """Principal-point-centred crop followed by resize, with exact K adjustment."""
    source_width, source_height = image.size
    if min(source_width, source_height, height, width) <= 0:
        raise ValueError("image dimensions must be positive")
    K = intrinsics.float().clone()
    cx, cy = float(K[0, 2]), float(K[1, 2])
    margin_x = min(cx, source_width - cx)
    margin_y = min(cy, source_height - cy)
    if margin_x <= 1 or margin_y <= 1:
        raise RuntimeError(
            f"principal point {(cx, cy)} lies outside usable image margins "
            f"{(source_width, source_height)}")

    aspect = width / height
    crop_width = min(2.0 * margin_x, 2.0 * margin_y * aspect)
    crop_height = crop_width / aspect
    crop_width = max(2, min(source_width, int(math.floor(crop_width))))
    crop_height = max(2, min(source_height, int(math.floor(crop_height))))
    # Rounding can move the principal point by at most half a source pixel.  Clamp the
    # box instead of padding, because padding would create artificial NVS supervision.
    left = int(round(cx - crop_width / 2.0))
    top = int(round(cy - crop_height / 2.0))
    left = min(max(left, 0), source_width - crop_width)
    top = min(max(top, 0), source_height - crop_height)
    right, bottom = left + crop_width, top + crop_height
    image = image.crop((left, top, right, bottom))

    scale_x, scale_y = width / crop_width, height / crop_height
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] = (K[0, 2] - left) * scale_x
    K[1, 2] = (K[1, 2] - top) * scale_y
    if image.size != (width, height):
        image = image.resize((width, height), resample=Image.Resampling.BICUBIC)
    return image, K


class NVSRefined:
    """Indexed reader pooling ACID, DL3DV and refined-Re10K scenes.

    One Parquet row group contains one scene.  Only the selected JPEGs are decoded.
    Scene sampling is uniform over the pooled non-SpatialVID scenes rather than first
    choosing a subset, which makes every scene have equal probability.
    """

    def __init__(
        self,
        root: str,
        height: int = 252,
        width: int = 448,
        subsets: tuple[str, ...] = STATIC_SUBSETS,
        excluded_scene_ids: tuple[str, ...] = (),
    ):
        self.root = os.path.abspath(root)
        self.height = int(height)
        self.width = int(width)
        self.excluded_scene_ids = frozenset(
            str(scene_id).strip()
            for scene_id in excluded_scene_ids
            if str(scene_id).strip())
        requested = tuple(subsets)
        if not requested or any(subset not in STATIC_SUBSETS for subset in requested):
            raise ValueError(
                f"subsets must be drawn from {STATIC_SUBSETS}, got {requested}")

        root_path = Path(self.root)
        # Backwards compatibility: a direct .../DL3DV root still indexes only DL3DV.
        direct_files = sorted(root_path.glob("*.parquet"))
        locations: list[tuple[str, list[Path]]] = []
        if direct_files:
            inferred = root_path.name
            if inferred not in requested:
                raise ValueError(
                    f"direct subset root {inferred} is not enabled by {requested}")
            locations.append((inferred, direct_files))
        else:
            for subset in requested:
                files = sorted((root_path / subset).glob(f"{subset}_*.parquet"))
                if not files:
                    raise FileNotFoundError(
                        f"no {subset} parquet shards under {root_path / subset}")
                locations.append((subset, files))

        import pyarrow as pa
        import pyarrow.parquet as pq
        # DataLoader workers must not each create a large Arrow thread pool.
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        self.scenes: list[SceneLocation] = []
        self.subset_scene_counts: dict[str, int] = {}
        for subset, files in locations:
            before = len(self.scenes)
            for path in files:
                parquet = pq.ParquetFile(path)
                self.scenes.extend(
                    SceneLocation(str(path), row_group, subset)
                    for row_group in range(parquet.metadata.num_row_groups)
                )
            self.subset_scene_counts[subset] = len(self.scenes) - before
        self._parquet_cache = {}

    def __len__(self):
        return len(self.scenes)

    def _parquet(self, path: str):
        parquet = self._parquet_cache.get(path)
        if parquet is None:
            import pyarrow.parquet as pq
            parquet = pq.ParquetFile(path)
            self._parquet_cache[path] = parquet
        return parquet

    @staticmethod
    def _opencv_c2w_from_nerfstudio(matrix) -> torch.Tensor:
        # Nerfstudio uses OpenGL camera axes. Right multiplication changes only the
        # camera basis: +x right, +y down, +z forward for Pi3/gsplat.
        c2w = torch.tensor(matrix, dtype=torch.float32)
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
        return c2w @ flip

    @staticmethod
    def _unpack_float32(value: bytes, shape, name: str) -> np.ndarray:
        array = np.frombuffer(value, dtype=np.float32)
        expected = int(np.prod(shape))
        if array.size != expected:
            raise RuntimeError(
                f"packed {name} has {array.size} values, expected {expected}")
        return array.reshape(shape)

    @staticmethod
    def _sample_indices(frame_count: int, rng, views: int,
                        min_span: int, max_span: int) -> np.ndarray:
        capped_max = min(int(max_span), frame_count - 1)
        capped_min = max(views - 1, min(int(min_span), capped_max))
        if capped_max < views - 1:
            raise RuntimeError(
                f"scene has only {frame_count} frames for {views} unique views")
        span = int(rng.integers(capped_min, capped_max + 1))
        # ``span`` is the inclusive index difference, so valid starts number
        # ``frame_count - span`` (e.g. span=N-1 admits only start zero).
        start = int(rng.integers(0, frame_count - span))
        indices = np.rint(np.linspace(start, start + span, views)).astype(np.int64)
        if len(np.unique(indices)) != views:
            raise RuntimeError("temporal view sampler produced duplicate frames")
        return indices

    @staticmethod
    def _dl3dv_span_bounds(global_step: int) -> tuple[int, int]:
        """DepthSplat's DL3DV 15--30 -> 20--50 span warm-up."""
        fraction = min(max(int(global_step), 0) / DL3DV_CONTEXT_WARMUP_STEPS, 1.0)
        min_span = DL3DV_INITIAL_MIN_SPAN + int(
            (DL3DV_FINAL_MIN_SPAN - DL3DV_INITIAL_MIN_SPAN) * fraction)
        max_span = DL3DV_INITIAL_MAX_SPAN + int(
            (DL3DV_FINAL_MAX_SPAN - DL3DV_INITIAL_MAX_SPAN) * fraction)
        return min_span, max_span

    @classmethod
    def _sample_dl3dv_context_target_indices(
        cls,
        camera_centers: np.ndarray,
        rng: np.random.Generator,
        context_views: int,
        target_views: int,
        global_step: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Adapt DepthSplat's short-window/FPS sampling to fixed 8->8 RAE.

        DepthSplat samples a short temporal window and uses farthest-point sampling
        over camera centres for its extra context views.  Our RAE additionally keeps
        both temporal endpoints as context (the context must bracket every NVS target)
        and excludes context indices from the eight target indices.
        """
        centers = np.asarray(camera_centers, dtype=np.float32)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError(f"camera_centers must be [frames,3], got {centers.shape}")
        if not np.isfinite(centers).all():
            raise RuntimeError("DL3DV camera centres contain NaN/Inf")
        frame_count = len(centers)
        required = int(context_views) + int(target_views)
        min_span, max_span = cls._dl3dv_span_bounds(global_step)
        min_span = max(min_span, required - 1)
        max_span = min(max_span, frame_count - 1)
        if max_span < min_span:
            raise RuntimeError(
                f"DL3DV scene has only {frame_count} frames for a {min_span}-frame span")

        span = int(rng.integers(min_span, max_span + 1))
        left = int(rng.integers(0, frame_count - span))
        right = left + span
        candidates = np.arange(left, right + 1, dtype=np.int64)

        # The two endpoints define the canonical interpolation window. Select the
        # remaining context cameras by greedy farthest-point sampling in world space.
        selected_local = [0, len(candidates) - 1]
        candidate_centers = centers[candidates]
        min_sq_distance = np.minimum(
            ((candidate_centers - candidate_centers[0]) ** 2).sum(axis=1),
            ((candidate_centers - candidate_centers[-1]) ** 2).sum(axis=1),
        )
        min_sq_distance[selected_local] = -1.0
        while len(selected_local) < int(context_views):
            next_local = int(np.argmax(min_sq_distance))
            if min_sq_distance[next_local] < 0:
                raise RuntimeError("not enough unique DL3DV context candidates")
            selected_local.append(next_local)
            distance = ((candidate_centers - candidate_centers[next_local]) ** 2).sum(
                axis=1)
            min_sq_distance = np.minimum(min_sq_distance, distance)
            min_sq_distance[selected_local] = -1.0

        context_indices = np.sort(candidates[np.asarray(selected_local)])
        target_candidates = candidates[~np.isin(candidates, context_indices)]
        if len(target_candidates) < int(target_views):
            raise RuntimeError("not enough distinct DL3DV target candidates")
        target_indices = np.sort(
            rng.choice(target_candidates, size=int(target_views), replace=False))
        return context_indices, target_indices

    def load(
        self,
        scene_index: int,
        frame_indices: np.ndarray | None = None,
        *,
        rng: np.random.Generator | None = None,
        views: int = 16,
        min_span: int = 48,
        max_span: int = 160,
        dl3dv_depthsplat_sampling: bool = False,
        global_step: int = 0,
        context_views: int = 8,
        target_views: int = 8,
        return_metadata: bool = False,
        height: int | None = None,
        width: int | None = None,
    ):
        location = self.scenes[int(scene_index)]
        parquet = self._parquet(location.parquet)
        available = set(parquet.schema_arrow.names)
        wanted = (
            "id", "frames", "frame_names", "resolution", "pose",
            "pose_convention", "caption", "pose_shape", "intrinsics", "intr_shape",
        )
        columns = [name for name in wanted if name in available]
        row = parquet.read_row_group(
            location.row_group, columns=columns, use_threads=False).to_pylist()[0]
        if str(row["id"]) in self.excluded_scene_ids:
            raise RuntimeError(f"excluded NVS-Refined scene {row['id']}")
        names, frames = row["frame_names"], row["frames"]
        if len(names) != len(frames):
            raise RuntimeError(f"frame metadata mismatch in scene {row['id']}")
        if location.subset == "DL3DV":
            temporal_ids = [_frame_number(name, index) for index, name in enumerate(names)]
            if any(right <= left for left, right in zip(temporal_ids, temporal_ids[1:])):
                raise RuntimeError(
                    f"DL3DV frame_names are not strictly time ordered in {row['id']}")
        output_height = self.height if height is None else int(height)
        output_width = self.width if width is None else int(width)
        convention = row["pose_convention"]
        nerfstudio_pose = None
        packed_pose = packed_intrinsics = None
        if convention == "c2w_nerfstudio_json":
            nerfstudio_pose = json.loads(row["pose"])
            pose_frames = {
                Path(item["file_path"]).stem: item
                for item in nerfstudio_pose["frames"]
            }
        elif convention == "w2c_opencv_mat34":
            packed_pose = self._unpack_float32(
                row["pose"], row["pose_shape"], "pose")
            packed_intrinsics = self._unpack_float32(
                row["intrinsics"], row["intr_shape"], "intrinsics")
            if packed_pose.shape != (len(frames), 12):
                raise RuntimeError(f"unexpected pose shape {packed_pose.shape}")
            if packed_intrinsics.shape != (len(frames), 4):
                raise RuntimeError(
                    f"unexpected intrinsics shape {packed_intrinsics.shape}")
        else:
            raise ValueError(
                f"unsupported pose convention {convention!r} in {row['id']}")

        context_target_ordered = False
        if frame_indices is None:
            if rng is None:
                raise ValueError("rng is required when frame_indices are not supplied")
            if dl3dv_depthsplat_sampling and location.subset == "DL3DV":
                if nerfstudio_pose is None:
                    raise RuntimeError("DL3DV DepthSplat sampling requires c2w JSON poses")
                camera_centers = []
                for name in names:
                    pose_item = pose_frames.get(Path(name).stem)
                    if pose_item is None:
                        raise KeyError(f"pose for {name} missing in scene {row['id']}")
                    matrix = np.asarray(pose_item["transform_matrix"], dtype=np.float32)
                    camera_centers.append(matrix[:3, 3])
                context_indices, target_indices = (
                    self._sample_dl3dv_context_target_indices(
                        np.stack(camera_centers), rng, context_views, target_views,
                        global_step))
                frame_indices = np.concatenate((context_indices, target_indices))
                context_target_ordered = True
            else:
                frame_indices = self._sample_indices(
                    len(frames), rng, views, min_span, max_span)
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if bool(((frame_indices < 0) | (frame_indices >= len(frames))).any()):
            raise IndexError(f"frame index outside scene {row['id']}")

        result_images, result_c2w, result_K, result_ids = [], [], [], []
        for index in frame_indices.tolist():
            name = names[index]
            with Image.open(io.BytesIO(frames[index])) as decoded:
                image = decoded.convert("RGB")
                source_width, source_height = image.size

                if nerfstudio_pose is not None:
                    pose_item = pose_frames.get(Path(name).stem)
                    if pose_item is None:
                        raise KeyError(f"pose for {name} missing in scene {row['id']}")
                    c2w = self._opencv_c2w_from_nerfstudio(
                        pose_item["transform_matrix"])
                    original_width = float(nerfstudio_pose["w"])
                    original_height = float(nerfstudio_pose["h"])
                    fx = float(pose_item.get("fl_x", nerfstudio_pose["fl_x"]))
                    fy = float(pose_item.get("fl_y", nerfstudio_pose["fl_y"]))
                    cx = float(pose_item.get("cx", nerfstudio_pose["cx"]))
                    cy = float(pose_item.get("cy", nerfstudio_pose["cy"]))
                    sx, sy = source_width / original_width, source_height / original_height
                    K = torch.tensor([
                        [fx * sx, 0.0, cx * sx],
                        [0.0, fy * sy, cy * sy],
                        [0.0, 0.0, 1.0],
                    ], dtype=torch.float32)
                else:
                    w2c = np.eye(4, dtype=np.float32)
                    w2c[:3, :4] = packed_pose[index].reshape(3, 4)
                    c2w = torch.from_numpy(np.linalg.inv(w2c).astype(np.float32))
                    fx, fy, cx, cy = packed_intrinsics[index].tolist()
                    K = torch.tensor([
                        [fx * source_width, 0.0, cx * source_width],
                        [0.0, fy * source_height, cy * source_height],
                        [0.0, 0.0, 1.0],
                    ], dtype=torch.float32)

                image, K = _center_crop_resize(
                    image, K, output_height, output_width)
                array = np.asarray(image, dtype=np.uint8).copy()
            result_images.append(
                torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0))
            result_c2w.append(c2w)
            result_K.append(K)
            result_ids.append(_frame_number(name, index))

        images_tensor = torch.stack(result_images)
        c2w_tensor = torch.stack(result_c2w)
        K_tensor = torch.stack(result_K)
        if location.subset == "DL3DV":
            if not bool(torch.isfinite(c2w_tensor).all()):
                raise RuntimeError(f"non-finite DL3DV pose in {row['id']}")
            rotation_determinants = torch.linalg.det(c2w_tensor[:, :3, :3])
            if not bool(torch.allclose(
                    rotation_determinants, torch.ones_like(rotation_determinants),
                    atol=1e-3, rtol=1e-3)):
                raise RuntimeError(f"invalid DL3DV rotation in {row['id']}")
            if bool((c2w_tensor[:, :3, 3].abs() > 1e3).any()):
                raise RuntimeError(f"extreme DL3DV camera translation in {row['id']}")
            horizontal_fov = torch.rad2deg(2 * torch.atan(
                torch.tensor(output_width, dtype=torch.float32)
                / (2 * K_tensor[:, 0, 0])))
            vertical_fov = torch.rad2deg(2 * torch.atan(
                torch.tensor(output_height, dtype=torch.float32)
                / (2 * K_tensor[:, 1, 1])))
            if bool((torch.maximum(horizontal_fov, vertical_fov) > 100).any()):
                raise RuntimeError(f"DL3DV FOV exceeds 100 degrees in {row['id']}")

        return {
            "images": images_tensor,
            "c2w": c2w_tensor,
            "K": K_tensor,
            "frame_ids": torch.tensor(result_ids, dtype=torch.long),
            "scene_id": row["id"],
            "caption": row.get("caption") or "",
            "pose_convention": convention,
            "subset": location.subset,
            "resolution_hw": (output_height, output_width),
            "context_target_ordered": context_target_ordered,
        }


def _endpoint_split(sample: dict, context_views: int, target_views: int):
    if sample.get("context_target_ordered", False):
        views = context_views + target_views
        if len(sample["images"]) != views:
            raise RuntimeError(
                f"pre-split sample has {len(sample['images'])} views, expected {views}")
        return (
            sample["images"], sample["c2w"], sample["K"], sample["frame_ids"])
    views = context_views + target_views
    context_idx = torch.linspace(
        0, views - 1, context_views).round().long().unique()
    all_idx = torch.arange(views)
    target_idx = all_idx[~torch.isin(all_idx, context_idx)]
    if (len(context_idx) != context_views or len(target_idx) != target_views
            or int(context_idx[0]) != 0 or int(context_idx[-1]) != views - 1):
        raise RuntimeError("endpoint-bracketing view split is malformed")
    order = torch.cat((context_idx, target_idx))
    return (
        sample["images"][order],
        sample["c2w"][order],
        sample["K"][order],
        sample["frame_ids"][order],
    )


class OrderedNVSRefinedSampler(Dataset):
    """Deterministically sample endpoint-bracketed context/target clips."""

    def __init__(
        self,
        dataset: NVSRefined,
        context_views: int,
        target_views: int,
        length: int,
        start_step: int = 0,
        seed: int = 0,
        min_span: int = 48,
        max_span: int = 160,
    ):
        self.dataset = dataset
        self.context_views = int(context_views)
        self.target_views = int(target_views)
        self.views = self.context_views + self.target_views
        self.length = int(length)
        self.start_step = int(start_step)
        self.seed = int(seed)
        self.min_span = int(min_span)
        self.max_span = int(max_span)
        if self.context_views != 8 or self.target_views != 8:
            raise ValueError("NVS-Refined contract requires 8 context + 8 target")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample_index = self.start_step + int(index)
        for retry in range(16):
            rng = np.random.default_rng(self.seed + sample_index + retry * 104729)
            scene_index = int(rng.integers(0, len(self.dataset)))
            try:
                sample = self.dataset.load(
                    scene_index, rng=rng, views=self.views,
                    min_span=self.min_span, max_span=self.max_span,
                    dl3dv_depthsplat_sampling=True,
                    global_step=sample_index,
                    context_views=self.context_views,
                    target_views=self.target_views)
                return _endpoint_split(
                    sample, self.context_views, self.target_views)
            except (KeyError, RuntimeError, ValueError, OSError):
                if retry == 15:
                    raise


class MixedMultiResolutionRAESampler(Dataset):
    """Exact 1:1 RE10K/NVS-Refined sampler with synchronized resolution.

    Resolution is selected once per optimizer step, so every sample and every DDP rank
    has the same H/W. Dataset source is selected by global-sample parity, yielding exact
    50/50 mixing whenever the global batch is even.
    """

    def __init__(
        self,
        re10k_dataset,
        refined_dataset: NVSRefined,
        resolutions: tuple[tuple[int, int], ...],
        context_views: int,
        target_views: int,
        batch_size: int,
        world_size: int,
        rank: int,
        length: int,
        start_step: int = 0,
        seed: int = 0,
        min_span: int = 48,
        max_span: int = 160,
        return_metadata: bool = False,
        return_source_id: bool = False,
    ):
        self.re10k_dataset = re10k_dataset
        self.refined_dataset = refined_dataset
        self.resolutions = tuple((int(h), int(w)) for h, w in resolutions)
        self.context_views = int(context_views)
        self.target_views = int(target_views)
        self.views = self.context_views + self.target_views
        self.batch_size = int(batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.length = int(length)
        self.start_step = int(start_step)
        self.seed = int(seed)
        self.min_span = int(min_span)
        self.max_span = int(max_span)
        self.return_metadata = bool(return_metadata)
        self.return_source_id = bool(return_source_id)
        if self.context_views != 8 or self.target_views != 8:
            raise ValueError("mixed RAE contract requires 8 context + 8 target")
        if not self.resolutions or any(h % 14 or w % 14 for h, w in self.resolutions):
            raise ValueError("every resolution must be non-empty and divisible by 14")
        if self.world_size * self.batch_size % 2:
            raise ValueError("exact 50/50 data mixing requires an even global batch")

    def __len__(self):
        return self.length

    def _resolution_index(self, global_step: int) -> int:
        count = len(self.resolutions)
        cycle, position = divmod(global_step, count)
        permutation = np.random.default_rng(
            self.seed + 7000003 + cycle).permutation(count)
        return int(permutation[position])

    def _re10k(self, rng, resolution_index):
        scene = int(rng.integers(0, len(self.re10k_dataset)))
        views = self.re10k_dataset[(scene, resolution_index, self.views)]
        images = torch.stack([view["img"] for view in views])
        images = (images * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
        sample = {
            "images": images,
            "c2w": torch.stack([
                torch.from_numpy(view["camera_pose"]) for view in views]),
            "K": torch.stack([
                torch.from_numpy(view["camera_intrinsics"]) for view in views]),
            "frame_ids": torch.tensor([
                int(view["frame_id"]) for view in views]),
            "scene_id": f"re10k_{self.re10k_dataset.start_img_ids[scene][0]}",
            "caption": "",
        }
        if not bool((sample["frame_ids"][1:] >= sample["frame_ids"][:-1]).all()):
            raise RuntimeError("RE10K ordered-view contract is broken")
        return sample

    def _refined(self, rng, resolution, global_step):
        for _ in range(16):
            scene = int(rng.integers(0, len(self.refined_dataset)))
            try:
                return self.refined_dataset.load(
                    scene, rng=rng, views=self.views,
                    min_span=self.min_span, max_span=self.max_span,
                    dl3dv_depthsplat_sampling=True,
                    global_step=global_step,
                    context_views=self.context_views,
                    target_views=self.target_views,
                    height=resolution[0], width=resolution[1])
            except (KeyError, RuntimeError, ValueError, OSError):
                continue
        raise RuntimeError("failed to draw a valid NVS-Refined scene after 16 retries")

    def __getitem__(self, index):
        local_step, within_batch = divmod(int(index), self.batch_size)
        global_step = self.start_step + local_step
        resolution_index = self._resolution_index(global_step)
        resolution = self.resolutions[resolution_index]
        global_sample = (
            global_step * self.world_size * self.batch_size
            + self.rank * self.batch_size + within_batch)
        refined = global_sample % 2 == 0
        rng = np.random.default_rng(self.seed + global_sample)
        sample = (
            self._refined(rng, resolution, global_step)
            if refined else self._re10k(rng, resolution_index))
        images, c2w, K, frame_ids = _endpoint_split(
            sample, self.context_views, self.target_views)
        if images.shape[-2:] != resolution:
            raise RuntimeError(
                f"resolution sampler requested {resolution}, got {images.shape[-2:]}")
        result = (
            images, c2w, K, frame_ids,
            torch.tensor(int(refined), dtype=torch.int64),
            torch.tensor(resolution_index, dtype=torch.int64),
        )
        if self.return_source_id:
            result += (torch.tensor(
                RAE_SOURCE_IDS[
                    f"nvs_refined_{sample['subset'].lower()}"
                    if refined else "re10k"],
                dtype=torch.int64),)
        if self.return_metadata:
            result += (sample["scene_id"], sample.get("caption", ""))
        return result
