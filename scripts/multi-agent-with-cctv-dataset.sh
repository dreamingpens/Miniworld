#!/bin/sh
set -eu

# Override these at launch time if needed, for example:
#   NUM_WORKERS=8 OUTPUT_ROOT=/data/miniworld ./scripts/multi-agent-with-cctv-dataset.sh
NUM_WORKERS="${NUM_WORKERS:-12}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./out/multi_512_dataset}"
BLOCK_SIZE="${BLOCK_SIZE:-100}"

# 120 frames at the generator's 15 FPS output rate = 8 seconds.
generate_split() (
    split_name="$1"
    num_videos="$2"
    base_seed="$3"

    .venv/bin/python -m scripts.generate_videos_batch \
        --env-name MiniWorld-MovingBlockWorld-v0 \
        --dataset-root "${OUTPUT_ROOT}/${split_name}" \
        --num-videos "${num_videos}" \
        --block-size "${BLOCK_SIZE}" \
        --num-processes "${NUM_WORKERS}" \
        --seed "${base_seed}" \
        -- \
        --num-agents 4 \
        --agent-mesh agent_robot \
        --turn-step-deg 90 \
        --forward-step 1.0 \
        --heading-zero \
        --grid-mode \
        --grid-vel-min -1 \
        --grid-vel-max 1 \
        --grid-cardinal-only \
        --no-time-limit \
        --render-width 512 \
        --render-height 512 \
        --obs-width 64 \
        --obs-height 64 \
        --steps 120 \
        --room-size 16 \
        --block-size-xy 0.7 \
        --block-height 1.5 \
        --agent-box-allow-overlap \
        --box-allow-overlap \
        --policy biased_walk_v2 \
        --forward-prob 0.90 \
        --cam-fov-y 60 \
        --num-blocks-min 6 \
        --num-blocks-max 10 \
        --ensure-base-palette \
        --randomize-wall-tex \
        --randomize-floor-tex \
        --randomize-box-tex \
        --box-and-ball \
        --num-static-objects 12 \
        --static-object-spacing 3.0 \
        --four-corner-cameras \
        --camera-streams-only \
        --output-2d-map-frame
)

# Seeds do not overlap: train uses 123..5122, validation uses 5123..5622.
generate_split train 5000 123
generate_split validation 500 5123
