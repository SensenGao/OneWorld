#!/usr/bin/env bash
set -euo pipefail

: "${ONEWORLD_SOURCE_CHECKPOINT:?Set ONEWORLD_SOURCE_CHECKPOINT}"
: "${ONEWORLD_RAE:?Set ONEWORLD_RAE}"
: "${ONEWORLD_STATS:?Set ONEWORLD_STATS}"
: "${ONEWORLD_INPUT_ALIGN:?Set ONEWORLD_INPUT_ALIGN}"
: "${ONEWORLD_TEXT_STORE:?Set ONEWORLD_TEXT_STORE}"
: "${PI3_CHECKPOINT:?Set PI3_CHECKPOINT}"
: "${WAN_CHECKPOINT:?Set WAN_CHECKPOINT}"
: "${NVS_REFINED_ROOT:?Set NVS_REFINED_ROOT}"
: "${RE10K_TORCH_ROOT:?Set RE10K_TORCH_ROOT}"

read -r -a launcher_args <<< "${TORCHRUN_ARGS:---standalone --nproc_per_node=16}"

torchrun "${launcher_args[@]}" train/train_distill.py \
  --data-root "$NVS_REFINED_ROOT" \
  --re10k-root "$RE10K_TORCH_ROOT" \
  --pi3 "$PI3_CHECKPOINT" \
  --wan "$WAN_CHECKPOINT" \
  --rae "$ONEWORLD_RAE" \
  --stats "$ONEWORLD_STATS" \
  --input-align "$ONEWORLD_INPUT_ALIGN" \
  --text-store "$ONEWORLD_TEXT_STORE" \
  --source-checkpoint "$ONEWORLD_SOURCE_CHECKPOINT" \
  --out "${ONEWORLD_DISTILL_OUTPUT:-outputs/distill}" \
  ${EXTRA_ARGS:-}
