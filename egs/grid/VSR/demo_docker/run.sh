#!/usr/bin/env bash
# Run the GRID VSR demo web UI in an isolated container.
#
# Isolation: the container sees the conda env and the icefall repo read-only,
# can write only to demo_saved/ (consented data) and its private tmpfs /tmp,
# and runs as the invoking (non-root) user with all capabilities dropped. It
# lives on the internal-only 'vsr-net' network: no internet egress, no LAN
# access, no published host port -- the only way in is the cloudflared
# container (run_tunnel.sh), which joins the same network and proxies
# https://demo.ibrahimalmajai.com to http://vsr-demo:7860.
#
# Runs detached with a restart policy, so the demo survives crashes and
# reboots (the docker daemon is boot-enabled). Stop for good with:
#   docker rm -f vsr-demo
set -euo pipefail
cd "$(dirname "$0")/.."  # egs/grid/VSR

# Serving the mean-face / phone-lexicon / layer-8 system since 2026-08-16
# (was: centroid ROIs, BPE-58, layer 9, conformer_ctc2/exp32). It is ~0.7 pts
# worse on GRID and ~5 pts better cross-corpus (8.40% vs 13.38% on Lombard
# GRID), which is the trade we want for arbitrary webcam input.
#
# Three settings must move together or accuracy degrades silently rather than
# erroring: --roi-mode meanface, --layer 8, and the phone lexicon/HLG. The
# word-confidence calibration is fitted per model too -- the default map
# belongs to the BPE system. Previous launch line kept in run.sh.bpe-backup.

docker build -q -t vsr-demo demo_docker

docker network inspect vsr-net >/dev/null 2>&1 || \
    docker network create --internal vsr-net
mkdir -p demo_saved
docker rm -f vsr-demo >/dev/null 2>&1 || true
docker run -d --restart unless-stopped --name vsr-demo \
    --network vsr-net \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    --read-only --tmpfs /tmp:size=4g \
    --cap-drop ALL --security-opt no-new-privileges \
    --memory 16g \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data/miniconda3/envs/icefall-vsr:/data/miniconda3/envs/icefall-vsr:ro \
    -v /data/icefall:/data/icefall:ro \
    -v "$PWD/demo_saved:/data/icefall/egs/grid/VSR/demo_saved" \
    vsr-demo \
    python conformer_ctc2/demo.py --ui \
        --checkpoint conformer_ctc2/exp-phone-l8/pretrained.pt \
        --roi-mode meanface \
        --layer 8 \
        --tokens data/lang_phone/tokens_model.txt \
        --method 1best \
        --HLG data/lang_phone/HLG.pt \
        --words-file data/lang_phone/words.txt \
        --word-conf-calibration conformer_ctc2/word_conf_isotonic_calibration_phone_l8.npz \
        --avhubert-ckpt download/avhubert-ckpts/base_vox_iter5.pt \
        "$@"

# Operator access from this machine, quota-exempt (private-IP requests
# bypass the per-visitor limits; public traffic arrives via Cloudflare
# with a public Cf-Connecting-IP and is limited as usual).
sleep 2
IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' vsr-demo)
echo "Local (quota-exempt) URL: http://${IP}:7860  (public: https://demo.ibrahimalmajai.com)"
