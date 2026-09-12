# Training

OneWorld uses four release stages, run in order.

| stage | launcher | default steps | output |
|---|---|---:|---|
| RAE | `train_rae.sh` | 20,000 | representation encoder and 3DGS decoder |
| DiT + CVC | `train_dit.sh` | 100,000 | camera- and text-conditioned diffusion transformer |
| MDF | `train_mdf.sh` | 20,000 additional | joint DiT and 3D decoder training |
| Distillation | `train_distill.sh` | 20,000 | four-step generator with 3DGS render-and-encode feedback |

## Setup

Install the repository requirements and make the Pi3X and dataset packages importable:

```bash
pip install -r requirements.txt
export PYTHONPATH=/path/to/Pi3:/path/to/cut3r:$PYTHONPATH
```

Set the external checkpoints and datasets:

```bash
export PI3_CHECKPOINT=/path/to/Pi3X
export WAN_CHECKPOINT=/path/to/Wan2.1-T2V-1.3B
export RE10K_TORCH_ROOT=/path/to/re10k_torch
export NVS_REFINED_ROOT=/path/to/NVS-Refined
```

## Stage 1: RAE

```bash
export ONEWORLD_DENORM_STATS=/path/to/decoder_denorm_stats.pt
bash train/train_rae.sh
```

The default output is `outputs/rae`. Override it with `ONEWORLD_RAE_OUTPUT`.

## Stage 2: DiT

```bash
export ONEWORLD_RAE=/path/to/rae/final.pt
export ONEWORLD_STATS=/path/to/latent_stats.pt
export ONEWORLD_INPUT_ALIGN=/path/to/input_alignment.pt
export ONEWORLD_TEXT_STORE=/path/to/text_embeddings.pt
bash train/train_dit.sh
```

The default output is `outputs/dit`. Override it with `ONEWORLD_DIT_OUTPUT` or change the schedule with `ONEWORLD_DIT_STEPS`. The diffusion objective includes token-level cross-view correspondence preservation.

## Stage 3: MDF

```bash
export ONEWORLD_DIT_CHECKPOINT=/path/to/dit/checkpoint_directory
bash train/train_mdf.sh
```

MDF continues the DiT objective while jointly optimizing the 3D decoder on convex mixtures of predicted and ground-truth latents. The default schedule continues from step 100,000 to step 120,000.

## Stage 4: Distillation

```bash
export ONEWORLD_SOURCE_CHECKPOINT=/path/to/mdf/checkpoint_directory
bash train/train_distill.sh
```

The default output is `outputs/distill`. Override it with `ONEWORLD_DISTILL_OUTPUT`.

All three launchers use `torchrun`. Set `TORCHRUN_ARGS` for a different distributed configuration and `EXTRA_ARGS` for additional script arguments.
