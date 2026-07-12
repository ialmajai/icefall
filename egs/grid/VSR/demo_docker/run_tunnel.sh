#!/usr/bin/env bash
# Run the cloudflared tunnel container that fronts the vsr-demo container.
#
# The demo lives on the internal-only 'vsr-net' network (no internet egress,
# no published host port). cloudflared straddles two networks: the default
# bridge for outbound egress to Cloudflare's edge, and vsr-net to reach the
# demo at http://vsr-demo:7860 (the ingress service in
# ~/.cloudflared/config.yml). Tunnel credentials stay outside the demo
# container's reach.
set -euo pipefail

docker network inspect vsr-net >/dev/null 2>&1 || \
    docker network create --internal vsr-net
docker rm -f cloudflared-vsr >/dev/null 2>&1 || true
docker run -d --restart unless-stopped --name cloudflared-vsr \
    --user "$(id -u):$(id -g)" \
    -v "$HOME/.cloudflared:$HOME/.cloudflared:ro" \
    cloudflare/cloudflared:latest \
    tunnel --config "$HOME/.cloudflared/config.yml" run vsr-demo
docker network connect vsr-net cloudflared-vsr
