#!/usr/bin/env bash
# Find largest DiffusionGemma MAXTOK that can generate with a near-full prompt.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python3}"
BENCH="$ROOT/scripts/bench_diffusion_ctx.py"
OUT=/tmp/dg_ctx_sweep.log
: > "$OUT"

# Prefer exact-name kill so we don't match this script's argv.
kill_visual() {
  local p
  p=$(pgrep -f '/bin/llama-diffusion-gemma-visual-server' || true)
  if [[ -n "${p}" ]]; then
    # shellcheck disable=SC2086
    kill $p 2>/dev/null || true
    sleep 2
  fi
}

for N in 16384 12288 8192 6144; do
  kill_visual
  echo "======== MAXTOK=$N ========" | tee -a "$OUT"
  fill=$((N - 512))
  if ! $PY "$BENCH" --maxtok "$N" --fill-tokens "$fill" --blocks 2 --fa 1 --load-timeout 300 \
      2>&1 | tee -a "$OUT" | tee /tmp/dg_last_run.log | tail -n 5; then
    echo "run failed for N=$N" | tee -a "$OUT"
    continue
  fi
  if rg -q 'output_tok_s=' /tmp/dg_last_run.log; then
    echo "SUCCESS at MAXTOK=$N" | tee -a "$OUT"
    rg 'READY|resolved_maxtok|prompt_n=|output_tok_s=|parallel_tok_s=|wall_ms=' /tmp/dg_last_run.log | tee -a "$OUT"
    break
  fi
  if rg -qi 'GGML_ASSERT|out of memory|server closed|ERR ' /tmp/dg_last_run.log; then
    echo "OOM/crash at MAXTOK=$N" | tee -a "$OUT"
  fi
done

kill_visual
echo "done" | tee -a "$OUT"
