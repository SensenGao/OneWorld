"""CPU regressions for the released v19 weights and public Pi3X API."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from oneworld.rae.checkpoint import (
    _checkpoint_state, _compatible_v19_state, read_rae_checkpoint, validate_rae_parameters,
)
from oneworld.rae.encoder import RAEEncoder
from oneworld.rae.model import RAELatentEncoder
from oneworld.rae.model_base import BaseRAE
from oneworld.rae.release import check_flat_layout, fingerprints_close, file_sha256, load_named_parameters, verify_manifest
from oneworld.rae.wan_flow import WanPi3Flow


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("oneworld_infer", ROOT / "infer/infer.py")
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)
merge_spec = importlib.util.spec_from_file_location("merge_weights", ROOT / "infer/merge_weights.py")
merge_weights = importlib.util.module_from_spec(merge_spec)
merge_spec.loader.exec_module(merge_weights)
LEGACY_CONTRACT = {"dit_latent": (
    "concat(layernorm_stock_pi3_z18,layernorm_rgb32);"
    "reversible_per_resolution_channel_standardization_deferred_to_dit")}


class ReleaseCompatibilityTests(unittest.TestCase):
    def test_merge_restores_named_parameters_and_discards_only_padding(self):
        payloads = [{"ema": [torch.tensor([1., 2., 3.]), torch.tensor([7.])]},
                    {"ema": [torch.tensor([4., 5., 0.]), torch.tensor([8.])]}]
        groups = [{"module": "", "parameters": [
            {"name": "bias", "numel": 2, "shape": [2]},
            {"name": "weight", "numel": 3, "shape": [1, 3]}]},
            {"module": "blocks.0", "parameters": [
                {"name": "weight", "numel": 2, "shape": [1, 2]}]}]
        state = dict(merge_weights.named_shard_parameters(payloads, "ema", groups))
        self.assertTrue(torch.equal(state["bias"], torch.tensor([1., 2.])))
        self.assertTrue(torch.equal(state["weight"], torch.tensor([[3., 4., 5.]])))
        self.assertTrue(torch.equal(state["blocks.0.weight"], torch.tensor([[7., 8.]])))
        payloads[1]["ema"][0] = torch.zeros(2)
        with self.assertRaisesRegex(RuntimeError, "shape/dtype"):
            dict(merge_weights.named_shard_parameters(payloads, "ema", groups))

    def test_named_parameters_load_strictly_without_fsdp(self):
        from safetensors.torch import save_file
        module = nn.Linear(3, 2)
        state = {"generator."+name: value.detach().clone() for name, value in module.named_parameters()}
        state["text_projector.unrelated"] = torch.ones(1)
        metadata = {"format": "oneworld-inference-named-v1"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.safetensors"
            save_file(state, str(path), metadata=metadata)
            target = nn.Linear(3, 2).bfloat16()
            with self.assertRaisesRegex(RuntimeError, "shape/dtype"):
                load_named_parameters(path, target, "generator.")
            target.float()  # Restore the full-precision EMA before inference casting.
            load_named_parameters(path, target, "generator.")
            for name, value in target.named_parameters():
                self.assertTrue(torch.equal(value, state["generator."+name]))
            damaged = {name: value for name, value in state.items() if name != "generator.bias"}
            save_file(damaged, str(path), metadata=metadata)
            with self.assertRaisesRegex(RuntimeError, "parameters differ"):
                load_named_parameters(path, target, "generator.")
            damaged = {**state, "generator.weight": torch.zeros(2, 4)}
            save_file(damaged, str(path), metadata=metadata)
            with self.assertRaisesRegex(RuntimeError, "shape/dtype"):
                load_named_parameters(path, target, "generator.")
            damaged = {**state, "generator.weight": state["generator.weight"].double()}
            save_file(damaged, str(path), metadata=metadata)
            with self.assertRaisesRegex(RuntimeError, "shape/dtype"):
                load_named_parameters(path, target, "generator.")

    def test_merged_rae_json_loads_explicit_identity_parameters(self):
        from safetensors.torch import save_file
        model = self.posterior_model()
        state = _compatible_v19_state(model, {}, LEGACY_CONTRACT)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_file({"rae."+name: value for name, value in state.items()},
                      str(root/"model.safetensors"), metadata={"format":"oneworld-inference-named-v1"})
            metadata = {"format":"oneworld-rae-inference-v2",
                        "contract":{**LEGACY_CONTRACT,"appearance_posterior":"identity"},
                        "tensor_source":{"format":"safetensors","file":"model.safetensors","prefix":"rae."}}
            (root/"rae.json").write_text(json.dumps(metadata))
            loaded = _checkpoint_state(model, read_rae_checkpoint(root/"rae.json"))
            self.assertTrue(model.appearance_identity)
            for name, value in state.items():
                self.assertTrue(torch.equal(loaded[name],value))

    def test_merged_package_requires_one_weight_file_and_no_shard_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"format":"oneworld-4step-v3","checkpoint":"checkpoint","rae":"rae.json",
                      "stats":"latent_stats.pt","weights":"model.safetensors","manifest":"manifest.json",
                      "aligned_input":True,"weight_layout":"named_parameters_v1"}
            (root/"config.json").write_text(json.dumps(config))
            (root/"checkpoint").mkdir()
            for name in ("rae.json","latent_stats.pt","model.safetensors","manifest.json"):
                (root/name).touch()
            _, paths = inference.resolve_model_files(root)
            self.assertNotIn("layout",paths)
            self.assertNotIn("input_alignment",paths)
            (root/"model.safetensors").unlink()
            with self.assertRaises(FileNotFoundError):
                inference.resolve_model_files(root)

    def posterior_model(self):
        model = nn.Module()
        model.appearance_dim = 32
        model.posterior_mean = nn.Conv2d(32, 32, 1)
        model.posterior_logvar = nn.Conv2d(32, 32, 1)
        return model

    def test_repaired_rae_never_fills_missing_or_incorrect_parameters(self):
        model = self.posterior_model()
        state = _compatible_v19_state(model, {}, LEGACY_CONTRACT)
        payload = {"format": "oneworld-rae-inference-v2", "rae": state,
                   "contract": {**LEGACY_CONTRACT, "appearance_posterior": "identity"}}
        self.assertIs(_checkpoint_state(model, payload), state)
        self.assertTrue(model.appearance_identity)
        for missing in state:
            damaged = {**payload, "rae": {k: v for k, v in state.items() if k != missing}}
            with self.assertRaisesRegex(RuntimeError, "saved posterior"):
                _checkpoint_state(model, damaged)
        damaged = {**payload, "rae": {**state, "posterior_mean.weight": state["posterior_mean.weight"] + 0.1}}
        with self.assertRaisesRegex(RuntimeError, "identity posterior"):
            _checkpoint_state(model, damaged)
        damaged = {**payload, "rae": {**state, "posterior_mean.bias": torch.zeros(1)}}
        with self.assertRaisesRegex(RuntimeError, "shape differs"):
            _checkpoint_state(model, damaged)
        damaged = {**payload, "rae": {**state, "wan_alignment_projection.bias": torch.zeros(16)}}
        with self.assertRaisesRegex(RuntimeError, "obsolete"):
            _checkpoint_state(model, damaged)

    def test_identity_posterior_preserves_fp32_under_autocast(self):
        model = self.posterior_model().eval()
        state = _checkpoint_state(model, {"rae": {}, "contract": LEGACY_CONTRACT})
        model.load_state_dict(state)
        appearance = torch.randn(1, 2, 32, 3, 4)
        model.encoder = SimpleNamespace(encode_z18=lambda *a, **kw: {"z18": appearance})
        model.appearance_encoder = lambda *a: {"rgb_residual_raw": appearance}
        model.geometry_norm = nn.LayerNorm(32, elementwise_affine=False)
        model.appearance_norm = nn.LayerNorm(32, elementwise_affine=False)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            latent, mean, logvar, variance = BaseRAE._appearance_posterior(model, appearance)
            encoded = RAELatentEncoder.encode_latent(model, None, None, None)
            self.assertIs(latent, appearance)
            self.assertIs(mean, appearance)
            expected = model.appearance_norm(appearance.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            self.assertTrue(torch.equal(encoded["z_rgb"], expected))
            self.assertEqual(encoded["z_rgb"].dtype, torch.float32)
            self.assertEqual(mean.dtype, torch.float32)
            self.assertEqual(logvar.count_nonzero(), 0)
            self.assertTrue(torch.equal(variance, torch.ones_like(appearance)))
            # A complete learned posterior must retain its ordinary computation.
            _checkpoint_state(model, {"rae": state, "contract": LEGACY_CONTRACT})
            self.assertFalse(model.appearance_identity)
            self.assertEqual(BaseRAE._appearance_posterior(model, appearance)[1].dtype,
                             torch.bfloat16)

    def test_missing_trained_pi3_decoder_cannot_fall_back_to_stock_weights(self):
        model = nn.Module()
        model.encoder = nn.Module()
        model.encoder.pi3x = nn.Module()
        model.encoder.pi3x.decoder = nn.ModuleList([nn.Linear(2, 2) for _ in range(36)])
        state = {k: p.detach().clone() for k, p in model.named_parameters()
                 if int(k.split('.')[3]) >= 18}
        validate_rae_parameters(model, state)
        del state["encoder.pi3x.decoder.18.weight"]
        with self.assertRaisesRegex(RuntimeError, "missing parameters"):
            validate_rae_parameters(model, state)

    def test_full_file_manifest_detects_changes_and_missing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "weights.bin"
            path.write_bytes(b'authentic parameters')
            manifest = {"format": "oneworld-release-manifest-v1", "files": {
                "weights.bin": {"size": path.stat().st_size, "sha256": file_sha256(path)}}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            # A full Hugging Face snapshot also includes repository metadata.
            (root / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n")
            verify_manifest(root)
            path.write_bytes(b'corrupted parameters')
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                verify_manifest(root)
            path.unlink()
            with self.assertRaisesRegex(RuntimeError, "file list"):
                verify_manifest(root)

    def test_flat_parameter_layout_rejects_order_and_shape_mismatches(self):
        parameter = SimpleNamespace(_fqns=["weight", "bias"], _shapes=[(3, 4), (3,)])
        groups = [{"parameters": [{"name": "weight", "shape": [3, 4]},
                                  {"name": "bias", "shape": [3]}]}]
        check_flat_layout([parameter], groups)
        parameter._fqns.reverse()
        with self.assertRaisesRegex(RuntimeError, "names differ"):
            check_flat_layout([parameter], groups)
        parameter._fqns.reverse(); parameter._shapes = [(4, 3), (3,)]
        with self.assertRaisesRegex(RuntimeError, "shapes differ"):
            check_flat_layout([parameter], groups)

    def test_fingerprints_accept_reduction_roundoff_and_reject_wrong_counts(self):
        expected = torch.tensor([123, 0.9, 8.4], dtype=torch.float64)
        actual = expected.clone(); actual[1] += 1e-7
        self.assertTrue(fingerprints_close(expected, actual))
        actual[0] += 1
        self.assertFalse(fingerprints_close(expected, actual))
        actual[0] -= 1; actual[2] = float('nan')
        self.assertFalse(fingerprints_close(expected, actual))

    def test_final_package_without_alignment_initializer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"checkpoint": "checkpoint", "rae": "rae.pt", "stats": "stats.pt"}
            (root / "config.json").write_text(json.dumps(config))
            for name in config.values():
                (root / name).touch()
            _, paths = inference.resolve_model_files(root)
            self.assertNotIn("input_alignment", paths)
            config["input_alignment"] = "missing.pt"
            (root / "config.json").write_text(json.dumps(config))
            with self.assertRaises(FileNotFoundError):
                inference.resolve_model_files(root)

    def test_aligned_architecture_without_initializer(self):
        with torch.device("meta"):
            backbone = SimpleNamespace(
                model_type="t2v", dim=1536, num_heads=12, freq_dim=256,
                time_embedding=nn.Linear(256, 1536),
                time_projection=nn.Linear(1536, 9216),
                blocks=nn.ModuleList(), freqs=torch.zeros(1), eps=1e-6)
            model = WanPi3Flow(backbone, latent_dim=1056, aligned_input=True)
        self.assertEqual(model.main_latent_input.in_features, 1056)
        self.assertTrue(hasattr(model, "main_condition_input"))
        self.assertFalse(hasattr(model, "main_ref_input"))

    def test_text_fingerprint_roundoff_and_corruption(self):
        projector = nn.Linear(4, 3)
        expected = inference.module_fingerprint(projector)
        expected[1] += 1e-7
        payload = {"format": "wan-native-text-projector-v1",
                   "ema": projector.state_dict(), "ema_fingerprint": expected}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "text_projector.pt"
            torch.save(payload, path)
            inference.load_text_projector(Path(folder), projector)
            payload["ema"]["weight"] = projector.weight.detach().clone() + 0.1
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                inference.load_text_projector(Path(folder), projector)

    def test_legacy_posterior_preserves_appearance(self):
        model = nn.Module()
        model.appearance_dim = 32
        model.posterior_mean = nn.Conv2d(32, 32, 1)
        model.posterior_logvar = nn.Conv2d(32, 32, 1)
        legacy = {"wan_alignment_projection.weight": torch.empty(16, 32, 1, 1),
                  "wan_alignment_projection.bias": torch.empty(16)}
        state = _compatible_v19_state(model, legacy, LEGACY_CONTRACT)
        model.load_state_dict(state, strict=True)
        appearance = torch.randn(2, 32, 3, 4)
        torch.testing.assert_close(model.posterior_mean(appearance), appearance, rtol=0, atol=0)
        self.assertEqual(model.posterior_logvar(appearance).count_nonzero(), 0)
        with self.assertRaisesRegex(RuntimeError, "legacy v19 latent contract"):
            _compatible_v19_state(model, {}, {})
        with self.assertRaisesRegex(RuntimeError, "incomplete RAE posterior"):
            _compatible_v19_state(model, {"posterior_mean.weight": state["posterior_mean.weight"]},
                                  LEGACY_CONTRACT)
        self.assertIn("wan_alignment_projection.weight", legacy)

    def test_public_pi3_pose_metadata_is_not_semantic_tokens(self):
        class Dino(nn.Module):
            def forward(self, images, is_training=True):
                return {"x_norm_patchtokens": images.flatten(2).transpose(1, 2)}

        class Pi3(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Dino()
                self.image_mean, self.image_std = 0, 1

            def encode(self, images, **kwargs):
                hidden = self.encoder(images.reshape(-1, *images.shape[2:]))[
                    "x_norm_patchtokens"]
                hidden.add_(1)  # A backbone may inject the ray prior in place.
                return hidden, torch.eye(4).reshape(1, 1, 4, 4), None, None, None

        pi3 = Pi3()
        images = torch.randn(1, 1, 3, 2, 2)
        hidden, semantic = RAEEncoder._base_tokens(SimpleNamespace(pi3x=pi3), images,
                                                 torch.eye(3).reshape(1, 1, 3, 3))
        torch.testing.assert_close(semantic, images.flatten(3).reshape(1, 3, 4).transpose(1, 2))
        torch.testing.assert_close(hidden, semantic + 1)
        self.assertEqual(len(pi3.encoder._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
