#!/usr/bin/env bash
# Run the GRID VSR demo web UI in an isolated container.
#
# Isolation: the container sees the conda env and the icefall repo read-only,
# can write only to demo_saved/ (consented data) and its private tmpfs /tmp,
# runs as the invoking (non-root) user with all capabilities dropped, and the
# UI is published on 127.0.0.1 only (put cloudflared/a proxy in front for
# public access).
set -euo pipefail
cd "$(dirname "$0")/.."  # egs/grid/VSR

docker build -q -t vsr-demo demo_docker

mkdir -p demo_saved
exec docker run --rm --name vsr-demo \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    --read-only --tmpfs /tmp:size=2g \
    --cap-drop ALL --security-opt no-new-privileges \
    --memory 16g \
    -v /data/miniconda3/envs/icefall-vsr:/data/miniconda3/envs/icefall-vsr:ro \
    -v /data/icefall:/data/icefall:ro \
    -v "$PWD/demo_saved:/data/icefall/egs/grid/VSR/demo_saved" \
    -p 127.0.0.1:7860:7860 \
    vsr-demo \
    python conformer_ctc2/demo.py --ui \
        --checkpoint conformer_ctc2/exp32/pretrained.pt \
        --tokens data/lang_bpe_58/tokens.txt \
        --method 1best \
        --HLG data/lang_bpe_58/HLG.pt \
        --words-file data/lang_bpe_58/words.txt \
        --avhubert-ckpt download/avhubert-ckpts/base_vox_iter5.pt \
        "$@"
