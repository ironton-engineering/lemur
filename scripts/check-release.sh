#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
failed=0

require_file() {
  if [[ ! -s "$ROOT/$1" ]]; then
    printf 'ERROR: Required release file is missing or empty: %s\n' "$1" >&2
    failed=1
  fi
}

for file in LICENSE NOTICE THIRD_PARTY_NOTICES.md VERSION requirements.lock \
  release/versions.env release/vllm.lock docs/INSTALL_LAYOUT.md; do
  require_file "$file"
done

for script in install.sh scripts/install-release.sh scripts/install-vllm.sh \
  scripts/install-desktop.sh scripts/uninstall.sh scripts/lemur scripts/system_probe.py; do
  require_file "$script"
  if [[ ! -x "$ROOT/$script" ]]; then
    printf 'ERROR: Release script is not executable: %s\n' "$script" >&2
    failed=1
  fi
done

if rg -n -i 'dream|diffusiongemma|diffusion-gemma|DG_VISUAL' \
  --glob '!check-release.sh' \
  "$ROOT/server" "$ROOT/static" "$ROOT/scripts" "$ROOT/tests" "$ROOT/README.md"; then
  printf 'ERROR: An unsupported backend reference remains.\n' >&2
  failed=1
fi

if rg -n '/home/|RTX 5090|RTX 3060|miniconda|\.unsloth' \
  --glob '!check-release.sh' \
  "$ROOT/server" "$ROOT/static" "$ROOT/scripts" "$ROOT/README.md"; then
  printf 'ERROR: A machine-specific release value remains.\n' >&2
  failed=1
fi

python3 -m compileall -q "$ROOT/server" "$ROOT/scripts" "$ROOT/tests"
bash -n "$ROOT/install.sh" "$ROOT/scripts/"*.sh "$ROOT/scripts/lemur"
rg -q -- '--hash=sha256:' "$ROOT/requirements.lock" || {
  printf 'ERROR: The application lock file has no package hashes.\n' >&2
  failed=1
}
rg -q -- '--hash=sha256:' "$ROOT/release/vllm.lock" || {
  printf 'ERROR: The vLLM lock file has no package hashes.\n' >&2
  failed=1
}
python3 - "$ROOT/release/manifest.example.json" <<'PY' || failed=1
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for field in ("version", "archive", "archive_url", "size", "sha256"):
    if field not in manifest:
        raise SystemExit(f"ERROR: The release manifest has no {field} field")
PY

exit "$failed"
