#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PORT=9000
URL="http://127.0.0.1:${PORT}"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/lemur"
PID_FILE="$CONFIG_DIR/server.pid"
LOG_FILE="$CONFIG_DIR/server.log"
LAUNCH_LOG="$CONFIG_DIR/launch.log"
SYS_PYTHON="/usr/bin/python3"

mkdir -p "$CONFIG_DIR"
exec >>"$LAUNCH_LOG" 2>&1
echo "---- $(date -Is) launch ----"

if [[ ! -d "$VENV" ]]; then
  "$SYS_PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install -q -r "$ROOT/requirements.lock"
fi

server_up() {
  # Use a small health check during startup.
  curl -sf --max-time 1 "$URL/v1/health" >/dev/null 2>&1 \
    || curl -sf --max-time 1 "$URL/" >/dev/null 2>&1
}

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    if server_up; then
      echo "server already running ($old_pid)"
      exec "$SYS_PYTHON" "$ROOT/scripts/window.py" "$URL"
    fi
    echo "stopping an unhealthy Lemur server ($old_pid)"
    kill "$old_pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi

echo "starting uvicorn on :$PORT"
nohup "$VENV/bin/uvicorn" server.main:app \
  --host 127.0.0.1 \
  --port "$PORT" \
  --no-access-log \
  >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

ready=0
for _ in $(seq 1 60); do
  if server_up; then
    ready=1
    break
  fi
  sleep 0.25
done
if [[ "$ready" -ne 1 ]]; then
  echo "WARNING: server not ready after wait; see $LOG_FILE"
else
  echo "server ready"
fi

echo "opening GTK window → $URL"
exec "$SYS_PYTHON" "$ROOT/scripts/window.py" "$URL"
