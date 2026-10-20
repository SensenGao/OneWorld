#!/usr/bin/env python3
"""Losslessly consolidate a repaired v2 release into one named safetensors file.

CPU only. Final generator EMA and decoder EMA are restored before serialization;
neither averaging nor reduced-precision conversion is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oneworld.rae.release import file_sha256, verify_manifest


def named_shard_parameters(payloads, field, groups):
    world = len(payloads)
    if len(groups) != len(payloads[0][field]):
        raise RuntimeError(f"source group count differs: {field}")
    for index, group in enumerate(groups):
        full_numel = sum(parameter["numel"] for parameter in group["parameters"])
        size = (full_numel + world - 1) // world
        parts = [payload[field][index] for payload in payloads]
        if any(value.ndim != 1 or value.numel() != size or value.dtype != parts[0].dtype
               for value in parts):
            raise RuntimeError(f"source shard shape/dtype differs: {field}, {index}")
        full = torch.cat(parts)[:full_numel]
        offset = 0
        for parameter in group["parameters"]:
            count = parameter["numel"]
            name = ".".join(value for value in (group["module"], parameter["name"]) if value)
            value = full[offset:offset+count].reshape(parameter["shape"]).clone()
            offset += count
            if not torch.isfinite(value).all():
                raise RuntimeError(f"nonfinite source parameter: {name}")
            yield name, value
        assert offset == full_numel


def export(source, destination):
    source_manifest = verify_manifest(source)
    config = json.loads((source/"config.json").read_text())
    if config.get("format") != "oneworld-4step-v2":
        raise RuntimeError("merge expects the verified repaired v2 package")
    checkpoint = source/config["checkpoint"]
    layout = json.loads((checkpoint/"layout.json").read_text())
    first = torch.load(checkpoint/"rank00000.pt", map_location="cpu", mmap=True, weights_only=False)
    world = int(first["world_size"])
    payloads = [torch.load(checkpoint/f"rank{rank:05d}.pt", map_location="cpu", mmap=True,
                           weights_only=False) for rank in range(world)]
    for rank, payload in enumerate(payloads):
        if (payload["rank"],payload["world_size"],payload["step"]) != (rank,world,first["step"]):
            raise RuntimeError("source shard metadata differs")
    parameters = {}
    for name, value in named_shard_parameters(payloads,"ema",layout["ema"]):
        key = "generator."+name
        if key in parameters:
            raise RuntimeError(f"duplicate generator parameter: {name}")
        parameters[key] = value
    generator_count = len(parameters)
    rae = torch.load(source/config["rae"], map_location="cpu", mmap=True, weights_only=False)
    if rae.get("format") != "oneworld-rae-inference-v2":
        raise RuntimeError("source RAE must already be strictly repaired")
    state = dict(rae["rae"])
    decoder_count = 0
    for name, value in named_shard_parameters(payloads,"decoder_ema",layout["decoder_ema"]):
        if name not in state or state[name].shape != value.shape or state[name].dtype != value.dtype:
            raise RuntimeError(f"decoder EMA parameter differs from RAE: {name}")
        state[name] = value
        decoder_count += 1
    for name,value in state.items():
        parameters["rae."+name] = value.contiguous().clone()
    text = torch.load(checkpoint/"text_projector.pt",map_location="cpu",weights_only=False)
    for name,value in text["ema"].items():
        parameters["text_projector."+name] = value.contiguous().clone()

    destination.mkdir(parents=True)
    (destination/"checkpoint").mkdir()
    model = destination/"model.safetensors"
    save_file(parameters,str(model),metadata={"format":"oneworld-inference-named-v1",
              "step":str(first["step"]),"generator":"ema","decoder":"ema"})
    with safe_open(str(model),framework="pt",device="cpu") as handle:
        if set(handle.keys()) != set(parameters):
            raise RuntimeError("merged parameter names differ after serialization")
        for name,value in parameters.items():
            if not torch.equal(handle.get_tensor(name),value):
                raise RuntimeError(f"merge changed a parameter: {name}")
    metadata = {key:value for key,value in rae.items() if key != "rae"}
    metadata["contract"]["decoder_denorm"]["statistics"] = "embedded_in_model.safetensors"
    metadata.update(tensor_source={"format":"safetensors","file":"model.safetensors","prefix":"rae."},
                    decoder_ema_merged=True)
    (destination/"rae.json").write_text(json.dumps(metadata,indent=2)+"\n")
    merged_config = {key:value for key,value in config.items()
                     if key not in ("input_alignment","decoder_layout")}
    merged_config.update(format="oneworld-4step-v3",weights="model.safetensors",rae="rae.json",
                         weight_layout="named_parameters_v1",step=int(first["step"]),
                         source_world_size=world,text_projector_fingerprint=text["ema_fingerprint"].tolist())
    (destination/"config.json").write_text(json.dumps(merged_config,indent=2)+"\n")
    contract=json.loads((checkpoint/"contract.json").read_text())
    contract.pop("parameter_layout",None)
    contract.update(format="oneworld-4step-v3",weight_layout="named_parameters_v1",decoder_ema_merged=True)
    (destination/"checkpoint/contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    shutil.copy2(source/config["stats"],destination/"latent_stats.pt")
    (destination/"README.md").write_text(
        "# OneWorld merged inference weights\n\n"
        "One complete `model.safetensors` contains the generator EMA, RAE with final decoder EMA, "
        "and text-projector EMA, stored under their original parameter names. "
        "All values retain their original precision. There are no rank-local model shards.\n\n"
        "Use the matching repaired inference code supporting `oneworld-4step-v3`, with the "
        "public Wan2.1-T2V-1.3B and Pi3X backbones. GitHub code publication is pending. "
        "No input-alignment initializer or decoder shard-layout file is needed. "
        "Latent statistics are stored separately in `latent_stats.pt`.\n\n"
        "Verify with `python infer/verify_weights.py --model /path/to/this/package`, then use "
        "the same path with `python infer/infer.py --model ...`.\n")
    files={str(path.relative_to(destination)):{"size":path.stat().st_size,"sha256":file_sha256(path)}
           for path in sorted(destination.rglob("*")) if path.is_file()}
    manifest={"format":"oneworld-release-manifest-v1","files":files,
              "source_files_sha256":{name:value["sha256"] for name,value in source_manifest["files"].items()},
              "weight_layout":"named_parameters_v1","trained_parameters_changed":False,
              "precision_conversion":False,"generator_ema_parameters":generator_count,
              "decoder_ema_parameters_merged_into_rae":decoder_count,"rae_parameters":len(state),
              "total_named_parameters":len(parameters),"source_world_size":world,"source_step":int(first["step"])}
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    verify_manifest(destination)
    print(json.dumps(manifest,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();destination=args.out.resolve()
    if destination.exists():raise FileExistsError(destination)
    staging=destination.with_name(destination.name+".exporting")
    if staging.exists():raise FileExistsError(staging)
    export(args.source.resolve(),staging)
    staging.rename(destination)
    print("Merged checkpoint exported atomically:",destination,flush=True)


if __name__=="__main__":
    main()
