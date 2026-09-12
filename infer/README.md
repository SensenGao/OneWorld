# Inference

One reference image and an eight-view camera path produce multi-view images and a video rendered from the generated 3D Gaussian scene. Text is optional.

## Setup

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

The final line points to a checkout of the [Pi3](https://github.com/yyfz/Pi3) Python source. `--pi3` points to the downloaded Pi3X weights.

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

The released checkpoint contains 16 rank-local shards. Inference supports 1, 2, 4, 8, or 16 GPUs, and the runtime GPU count must divide 16:

```bash
torchrun --standalone --nproc_per_node=4 infer/infer.py ...
```
