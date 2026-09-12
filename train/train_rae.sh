#!/usr/bin/env bash
set -euo pipefail

: "${PI3_CHECKPOINT:?Set PI3_CHECKPOINT}"
: "${ONEWORLD_DENORM_STATS:?Set ONEWORLD_DENORM_STATS}"
: "${NVS_REFINED_ROOT:?Set NVS_REFINED_ROOT}"
: "${RE10K_TORCH_ROOT:?Set RE10K_TORCH_ROOT}"

read -r -a launcher_args <<< "${TORCHRUN_ARGS:---standalone --nproc_per_node=8}"

torchrun "${launcher_args[@]}" train/train_rae.py \
  --steps "${ONEWORLD_RAE_STEPS:-20000}" \
  --data-root "$NVS_REFINED_ROOT" \
  --re10k-root "$RE10K_TORCH_ROOT" \
  --pi3 "$PI3_CHECKPOINT" \
  --decoder-denorm-stats "$ONEWORLD_DENORM_STATS" \
  --out "${ONEWORLD_RAE_OUTPUT:-outputs/rae}" \
  ${EXTRA_ARGS:-}
