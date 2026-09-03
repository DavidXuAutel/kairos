#!/usr/bin/env bash
# Poll H100 AB50 (steps=5) and show a macOS dialog popup.
#
#   bash scripts/monitor_ab50_popup.sh              # once
#   bash scripts/monitor_ab50_popup.sh --loop 600    # every 10min until done
#   bash scripts/monitor_ab50_popup.sh --stop
#   bash scripts/monitor_ab50_popup.sh --status
set -euo pipefail

SSH_KEY="${SSH_KEY:-$HOME/.ssh/franka_ros2_ed25519}"
SSH_HOST="${SSH_HOST:-a25689@10.239.121.23}"
SSH_PORT="${SSH_PORT:-30987}"
STATE_DIR="${STATE_DIR:-$HOME/.cache/kairos_ab50_monitor}"
mkdir -p "$STATE_DIR"
PIDFILE="$STATE_DIR/loop.pid"
LASTFILE="$STATE_DIR/last_status.txt"
LOGFILE="$STATE_DIR/monitor.log"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

ssh_h100() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 -o IdentitiesOnly=yes \
    -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_HOST" "$@"
}

fetch_status() {
  ssh_h100 'bash -s' <<'EOF'
set +e
AB=$(cat ~/kairos/outputs/libero_plus/ab50_latest.txt 2>/dev/null || true)
if [ -z "$AB" ] || [ ! -d "$AB" ]; then
  echo "STATE=missing"
  exit 0
fi
python3 - <<PY
import json, time, subprocess
from pathlib import Path
ab = Path("""$AB""")
meta = {}
mp = ab / "ab_meta.json"
if mp.exists():
    meta = json.loads(mp.read_text())
files = sorted(ab.rglob("*results.json"))
n = len(files)
target = int(meta.get("baseline_tasks") or len(meta.get("task_ids") or []) or 50)
succ = eps = 0
durs = []
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    succ += int(d.get("successes") or 0)
    eps += int(d.get("total_episodes") or 0)
    if d.get("duration") is not None:
        durs.append(float(d["duration"]))
try:
    out = subprocess.check_output(
        ["bash", "-lc", "pgrep -af eval_libero_single | grep -v grep | wc -l"],
        text=True,
    )
    eval_n = int(out.strip() or 0)
except Exception:
    eval_n = 0
base_sr = meta.get("baseline_sr")
treat_sr = (succ / eps) if eps else None
done = n >= target and eval_n == 0
avg = (sum(durs) / len(durs)) if durs else None
print(f"STATE={'done' if done else 'running'}")
print(f"AB={ab}")
print(f"DONE={n}/{target}")
print(f"EVAL={eval_n}")
print(f"SUCC={succ}")
print(f"EPS={eps}")
print(f"SR={treat_sr if treat_sr is not None else 'na'}")
print(f"BASE_SR={base_sr if base_sr is not None else 'na'}")
print(f"AVG_DUR={avg if avg is not None else 'na'}")
print(f"TS={time.strftime('%Y-%m-%d %H:%M:%S')}")
PY
EOF
}

popup() {
  local title="$1"
  local body="$2"
  local timeout="${3:-20}"
  local title_q body_q
  title_q=$(printf '%s' "$title" | sed 's/\\/\\\\/g; s/"/\\"/g')
  body_q=$(printf '%s' "$body" | sed 's/\\/\\\\/g; s/"/\\"/g')
  osascript >/dev/null 2>&1 <<OSA || true
display notification "${body_q}" with title "${title_q}"
display dialog "${body_q}" with title "${title_q}" buttons {"OK"} default button 1 giving up after ${timeout}
OSA
}

run_once() {
  local raw
  if ! raw=$(fetch_status 2>&1); then
    popup "Kairos AB50" "监控失败：无法 SSH 到 H100"$'\n'"$raw" 25
    printf '%s\n' "$raw" | tee "$LASTFILE"
    return 1
  fi
  printf '%s\n' "$raw" | tee "$LASTFILE"
  {
    echo "---- $(date '+%F %T') ----"
    printf '%s\n' "$raw"
  } >>"$LOGFILE"

  local state done evaln sr base avg ts
  state=$(echo "$raw" | awk -F= '/^STATE=/{print $2}')
  done=$(echo "$raw" | awk -F= '/^DONE=/{print $2}')
  evaln=$(echo "$raw" | awk -F= '/^EVAL=/{print $2}')
  sr=$(echo "$raw" | awk -F= '/^SR=/{print $2}')
  base=$(echo "$raw" | awk -F= '/^BASE_SR=/{print $2}')
  avg=$(echo "$raw" | awk -F= '/^AVG_DUR=/{print $2}')
  ts=$(echo "$raw" | awk -F= '/^TS=/{print $2}')

  local msg
  msg="steps=5  进度 ${done}
worker=${evaln}  SR=${sr} (baseline10=${base})
avg_dur=${avg}s
${ts}"

  if [ "$state" = "done" ]; then
    popup "Kairos AB50 完成" "$msg"$'\n'"可恢复全量 sweep" 60
    return 10
  fi
  if [ "$state" = "missing" ]; then
    popup "Kairos AB50" "找不到 AB 输出目录" 20
    return 2
  fi
  popup "Kairos AB50 监控" "$msg" 20
  return 0
}

stop_loop() {
  if [ -f "$PIDFILE" ]; then
    local pid
    pid=$(cat "$PIDFILE" || true)
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 0.5
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
  fi
}

case "${1:-once}" in
  once|"")
    run_once
    ;;
  --loop|loop)
    INTERVAL_S="${2:-600}"
    stop_loop
    # First popup now (foreground).
    set +e
    run_once
    first_rc=$?
    set -e
    if [ "$first_rc" -eq 10 ]; then
      echo "AB50 already done; loop not started"
      exit 0
    fi
    # Fully detached ticker (survives terminal close).
    LOOP_WRAPPER="$STATE_DIR/loop_wrapper.sh"
    cat >"$LOOP_WRAPPER" <<WRAP
#!/usr/bin/env bash
echo \$\$ > '$PIDFILE'
while true; do
  sleep '$INTERVAL_S'
  date '+%F %T tick' >> '$LOGFILE'
  bash '$SELF' once >> '$LOGFILE' 2>&1
  rc=\$?
  if [ \$rc -eq 10 ]; then
    date '+%F %T done' >> '$LOGFILE'
    rm -f '$PIDFILE'
    exit 0
  fi
done
WRAP
    chmod +x "$LOOP_WRAPPER"
    if command -v setsid >/dev/null 2>&1; then
      setsid "$LOOP_WRAPPER" </dev/null >>"$LOGFILE" 2>&1 &
    else
      nohup "$LOOP_WRAPPER" </dev/null >>"$LOGFILE" 2>&1 &
    fi
    sleep 0.5
    echo "AB50 monitor loop pid=$(cat "$PIDFILE" 2>/dev/null || echo '?') interval=${INTERVAL_S}s"
    echo "Log: $LOGFILE"
    echo "Stop: bash $SELF --stop"
    ;;
  --status|status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "loop pid=$(cat "$PIDFILE")"
    else
      echo "loop: not running"
    fi
    if launchctl print "gui/$(id -u)/com.kairos.ab50.monitor" >/dev/null 2>&1; then
      echo "launchd: com.kairos.ab50.monitor loaded (every 600s + popup)"
    else
      echo "launchd: not loaded"
    fi
    [ -f "$LASTFILE" ] && { echo "---- last ----"; cat "$LASTFILE"; }
    ;;
  --stop|stop)
    stop_loop
    launchctl bootout "gui/$(id -u)/com.kairos.ab50.monitor" 2>/dev/null \
      || launchctl unload "$HOME/Library/LaunchAgents/com.kairos.ab50.monitor.plist" 2>/dev/null \
      || true
    echo "AB50 monitor stopped (loop + launchd)"
    ;;
  *)
    echo "Usage: $0 [once|--loop SECONDS|--stop|--status]" >&2
    exit 1
    ;;
esac
