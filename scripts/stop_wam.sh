#!/usr/bin/env bash
# Stop WAM listening on WAM_PORT (default 8005).
set -euo pipefail
KAIROS_ROOT="${KAIROS_ROOT:-$HOME/kairos}"
WAM_PORT="${WAM_PORT:-8005}"
PIDFILE="${WAM_PIDFILE:-$KAIROS_ROOT/logs/wam_${WAM_PORT}.pid}"

if [ -f "$PIDFILE" ]; then
  pid="$(cat "$PIDFILE" || true)"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi

# Fallback: free the port
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${WAM_PORT}/tcp" 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
  pids="$(lsof -t -iTCP:"$WAM_PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null || true
    sleep 1
    kill -9 $pids 2>/dev/null || true
  fi
fi
echo "[stop_wam] port $WAM_PORT cleared"
