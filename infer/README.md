# Inference

One reference image and an eight-view camera path produce multi-view images and a video rendered from the generated 3D Gaussian scene. Text is optional.

## Setup

For the tested Python 3.10 / Linux x86_64 / CUDA 12.1 runtime, install
`pip install -r infer/requirements-tested.txt`. The general requirements below
allow other versions, which should be validated separately.

```bash
pip install -r requirements.txt
pip install -r infer/requirements.txt

huggingface-cli download Sensen02/OneWorld \
    --local-dir weights/OneWorld
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir weights/Wan2.1-T2V-1.3B
huggingface-cli download yyfz233/Pi3X \
    --local-dir weights/Pi3X

export PYTHONPATH=/path/to/Pi3:$PYTHONPATH
```

The default Hugging Face release is already the merged v3 package, with all
OneWorld parameters in one `model.safetensors` file. Use the downloaded
`weights/OneWorld` directory directly with the inference commands below;
the repair and merge commands are only for older packages.

The final line points to a checkout of the [Pi3](https://github.com/yyfz/Pi3) Python source. `--pi3` points to the downloaded Pi3X weights.
Public Pi3 revision `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b` is used for
compatibility testing. VGGT's decoder utilities are installed by the root requirements.

The final OneWorld checkpoint includes the trained input projection. No separate
input-alignment checkpoint or placeholder file is needed for inference. The
released v19 RAE uses the normalized RGB32 appearance features directly; the
loader restores this deterministic mapping when loading its legacy weights.
The identity path passes these features through directly, preserving their FP32
values under mixed precision instead of rounding them through a convolution.

## Repair and verify a downloaded v1 package

The repaired v2 package saves the deterministic posterior explicitly, removes the
unused training alignment head, and includes the real final-EMA input projection.
It also records the exact generator/decoder parameter order and a SHA256 manifest.
All trained parameter values and the sampling schedule are preserved.

Run this CPU-only export once; the source directory is preserved and the output
directory must not already exist:

```bash
python infer/repair_weights.py \
    --source weights/OneWorld \
    --out weights/OneWorld-fixed \
    --wan weights/Wan2.1-T2V-1.3B \
    --pi3 weights/Pi3X
python infer/verify_weights.py --model weights/OneWorld-fixed
```

Use `--model weights/OneWorld-fixed` in the inference commands below. A v2 RAE
loads strictly: missing posterior/decoder tensors, changed shapes, and unexpected
parameters raise an error. The original v1 package remains supported by the
legacy compatibility loader. Parameter layouts and fingerprint checks are
validated for the final generator and decoder shards during inference.

For deployment, merge a repaired v2 package into one complete named checkpoint:

```bash
python infer/merge_weights.py --source weights/OneWorld-fixed --out weights/OneWorld-merged
python infer/verify_weights.py --model weights/OneWorld-merged
```

The merged v3 package stores all OneWorld parameters in `model.safetensors`:
generator EMA, RAE with the final decoder EMA already applied, and text-projector
EMA. Parameter names, values and precision are preserved. It needs neither
rank-local shard files, an input-alignment initializer, nor a saved FSDP group
layout. Use `--model weights/OneWorld-merged` with the commands below. The public
Wan/Pi3 backbones are still required. Legacy v1/v2 releases remain supported.

## Image + Camera

Use an empty prompt by omitting `--prompt`:

```bash
python infer/infer.py \
    --model weights/OneWorld \
    --wan weights/Wan2.1-T2V-1.3B \
    --pi3 weights/Pi3X \
    --image infer/examples/shared/reference.jpg \
    --cameras infer/examples/shared/cameras.json \
    --out outputs/image_camera
```

## Image + Camera + Text

```bash
python infer/infer.py \
    --model weights/OneWorld \
    --wan weights/Wan2.1-T2V-1.3B \
    --pi3 weights/Pi3X \
    --image infer/examples/shared/reference.jpg \
    --cameras infer/examples/shared/cameras.json \
    --prompt "A compact bedroom with a purple bed, white sink, and soft daylight." \
    --out outputs/image_camera_text
```

The four-step distilled checkpoint does not use CFG.

## Bundled Examples

Run both included cases with one command:

```bash
python infer/make_examples.py \
    --model weights/OneWorld \
    --wan weights/Wan2.1-T2V-1.3B \
    --pi3 weights/Pi3X \
    --out outputs/examples
```

## Output

```text
views/v00.png ... v07.png   eight requested 3DGS renders
grid.png                    source and rendered views as one contact sheet
sweep.mp4                   interpolated camera-path render
source.png                  processed reference image
meta.json                   prompt, seed, and output metadata
```

Set `--video-frames 0` to skip video export, or change `--video-frames` and `--video-fps` to control its duration.

The original v1/v2 checkpoints contain 16 rank-local shards and require 1, 2, 4, 8, or 16 GPUs. The merged v3 checkpoint loads parameters by name before FSDP wrapping and has no requirement that the runtime GPU count divide 16:

```bash
torchrun --standalone --nproc_per_node=4 infer/infer.py ...
```
