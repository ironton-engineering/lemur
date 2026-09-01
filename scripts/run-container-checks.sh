#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_check() {
  local image="$1"
  printf '\n==> %s\n' "$image"
  docker run --rm -v "$ROOT:/src" -w /src "$image" bash -lc '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq python3 python3-venv python3-pip git ca-certificates sudo >/dev/null
    useradd -m -u 1000 lemurci
    chown -R lemurci:lemurci /src
    su lemurci -c "
      set -euo pipefail
      cd /src
      python3 -m venv /tmp/venv
      /tmp/venv/bin/pip install -q --require-hashes -r requirements.lock
      /tmp/venv/bin/pip install -q pytest==9.1.1 ruff==0.15.20
      /tmp/venv/bin/python -m pytest -q
      ./scripts/check-release.sh
      ./release/build-release.sh >/dev/null
    "
  '
}

for image in ubuntu:22.04 ubuntu:24.04 ubuntu:26.04; do
  run_check "$image"
done

printf '\nContainer checks passed on Ubuntu 22.04, 24.04, and 26.04.\n'
