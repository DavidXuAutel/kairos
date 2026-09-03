#!/usr/bin/env bash
# Direct WAM tunnel: H100:8005 -> yao@10.229.20.125:8005 (no Mac hop).
# Lab PC talks to FR3 FCI at 10.229.66.91; do not change Desk / 10.229.66.70.
set -euo pipefail

H100_HOST="${H100_HOST:-10.239.121.23}"
H100_PORT="${H100_PORT:-30987}"
H100_USER="${H100_USER:-a25689}"
LAB_HOST="${LAB_HOST:-10.229.20.125}"
LAB_USER="${LAB_USER:-yao}"
WAM_PORT="${WAM_PORT:-8005}"
MAC_KEY="${MAC_KEY:-$HOME/.ssh/franka_ros2_ed25519}"
# Path on the H100 pod (do not expand local $HOME here).
TUNNEL_KEY_ON_H100="${TUNNEL_KEY_ON_H100:-/home/a25689/.ssh/kairos_h100_to_125_ed25519}"

ssh_h100() {
  ssh -o BatchMode=yes -o IdentitiesOnly=yes -i "$MAC_KEY" -p "$H100_PORT" \
    "${H100_USER}@${H100_HOST}" "$@"
}

ssh_lab() {
  ssh -o BatchMode=yes -o IdentitiesOnly=yes -i "$MAC_KEY" \
    "${LAB_USER}@${LAB_HOST}" "$@"
}

cmd="${1:-status}"
case "$cmd" in
  start)
    # Drop legacy Mac double-hop if present
    pkill -f "ssh -fN .* -R ${WAM_PORT}:127.0.0.1:18005 ${LAB_USER}@${LAB_HOST}" 2>/dev/null || true
    pkill -f "ssh -fN .* -L 18005:127.0.0.1:${WAM_PORT} -p ${H100_PORT}" 2>/dev/null || true
    ssh_h100 "bash -s" <<EOF
set -euo pipefail
pkill -f 'ssh .* -R ${WAM_PORT}:127.0.0.1:${WAM_PORT} ${LAB_USER}@${LAB_HOST}' 2>/dev/null || true
sleep 1
test -f ${TUNNEL_KEY_ON_H100}
ssh -fN -o ExitOnForwardFailure=yes -o BatchMode=yes -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=accept-new \
  -i ${TUNNEL_KEY_ON_H100} \
  -R ${WAM_PORT}:127.0.0.1:${WAM_PORT} ${LAB_USER}@${LAB_HOST}
pgrep -af 'ssh .* -R ${WAM_PORT}:127.0.0.1:${WAM_PORT}' | head -3
EOF
    sleep 2
    ssh_lab "curl -4 -sS --noproxy '*' -m 30 http://127.0.0.1:${WAM_PORT}/health"; echo
    ;;
  stop)
    ssh_h100 "pkill -f 'ssh .* -R ${WAM_PORT}:127.0.0.1:${WAM_PORT} ${LAB_USER}@${LAB_HOST}' || true"
    echo "stopped"
    ;;
  status)
    echo "=== H100 WAM ==="
    ssh_h100 "curl -4 -sS --noproxy '*' -m 15 http://127.0.0.1:${WAM_PORT}/health; echo; pgrep -af 'ssh .* -R ${WAM_PORT}:127.0.0.1:${WAM_PORT}' || echo 'no reverse tunnel'"
    echo "=== LAB ${LAB_HOST} ==="
    ssh_lab "curl -4 -sS --noproxy '*' -m 30 http://127.0.0.1:${WAM_PORT}/health; echo; ping -c 1 -W 1 10.229.66.91 | tail -2"
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
