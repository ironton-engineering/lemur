#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/release/versions.env"
INSTALL_ROOT="${LEMUR_INSTALL_ROOT:-$HOME/.local/share/llm-hub}"
backend="$INSTALL_ROOT/backends/vllm-$VLLM_VERSION-$VLLM_CUDA_VARIANT"
resolved_root="$(realpath -m "$INSTALL_ROOT")"
resolved_home="$(realpath -m "$HOME")"
[[ "$resolved_root" == "$resolved_home"/* && "$resolved_root" != "$resolved_home" ]] || {
  printf 'ERROR: The install directory must be inside the user home directory.\n' >&2
  exit 1
}

record_backend() {
  local state_file="${XDG_CONFIG_HOME:-$HOME/.config}/lemur/state.json"
  python3 - "$state_file" "$backend/bin/vllm" <<'PY'
import json
import os
import sys
from pathlib import Path

state_file = Path(sys.argv[1])
try:
    state = json.loads(state_file.read_text()) if state_file.is_file() else {}
except (json.JSONDecodeError, OSError):
    state = {}
if not isinstance(state, dict):
    state = {}
settings = state.get("settings")
if not isinstance(settings, dict):
    settings = {}
settings["vllm_bin"] = sys.argv[2]
state["settings"] = settings
state_file.parent.mkdir(parents=True, exist_ok=True)
temporary = state_file.with_name(f".{state_file.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(state, indent=2) + "\n")
os.replace(temporary, state_file)
PY
}

driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)"
python3 - "$driver" "$VLLM_MIN_DRIVER" <<'PY'
import re
import sys

def ver(value):
    return tuple(int(item) for item in re.findall(r"\d+", value))

if ver(sys.argv[1]) < ver(sys.argv[2]):
    raise SystemExit(
        f"ERROR: vLLM needs NVIDIA driver {sys.argv[2]} or newer; found {sys.argv[1]}"
    )
PY

if [[ -x "$backend/bin/vllm" ]] && "$backend/bin/vllm" --version >/dev/null 2>&1; then
  record_backend
  printf 'vLLM %s is already installed.\n' "$VLLM_VERSION"
  exit 0
fi

mkdir -p "$INSTALL_ROOT/backends"
python3 - "$INSTALL_ROOT" <<'PY'
import shutil
import sys

free_gib = shutil.disk_usage(sys.argv[1]).free / (1024**3)
if free_gib < 20:
    raise SystemExit(
        f"ERROR: vLLM installation needs at least 20 GiB free; found {free_gib:.1f} GiB"
    )
PY
stage="$INSTALL_ROOT/backends/.stage-vllm-$VLLM_VERSION-$$"
uv_stage="$INSTALL_ROOT/backends/.stage-uv-$UV_VERSION-$$"
wheel="$stage/vllm-${VLLM_VERSION}+${VLLM_CUDA_VARIANT}-cp38-abi3-manylinux_2_28_x86_64.whl"
uv_archive="$uv_stage/uv.tar.gz"
trap 'rm -rf -- "$stage" "$uv_stage"' EXIT
mkdir -p "$uv_stage"
uv_url="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz"
curl --fail --location --proto '=https' --tlsv1.2 "$uv_url" --output "$uv_archive"
printf '%s  %s\n' "$UV_X86_64_LINUX_SHA256" "$uv_archive" | \
  sha256sum --check --status || {
    printf 'ERROR: The uv archive checksum does not match.\n' >&2
    exit 1
  }
tar -xzf "$uv_archive" -C "$uv_stage"
uv_bin="$uv_stage/uv-x86_64-unknown-linux-gnu/uv"
[[ -x "$uv_bin" ]] || {
  printf 'ERROR: The uv command is missing from its verified archive.\n' >&2
  exit 1
}
UV_PYTHON_INSTALL_DIR="$INSTALL_ROOT/python" \
  "$uv_bin" venv --python "$VLLM_PYTHON_VERSION" --relocatable "$stage"
wheel_url="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}%2B${VLLM_CUDA_VARIANT}-cp38-abi3-manylinux_2_28_x86_64.whl"
curl --fail --location --proto '=https' --tlsv1.2 "$wheel_url" --output "$wheel"
printf '%s  %s\n' "$VLLM_WHEEL_SHA256" "$wheel" | sha256sum --check --status || {
  printf 'ERROR: The vLLM wheel checksum does not match.\n' >&2
  exit 1
}
python3 - "$ROOT/release/vllm.lock" "$stage/vllm-install.lock" "$wheel" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
wheel_uri = Path(sys.argv[3]).resolve().as_uri()
lines = [
    f"vllm @ {wheel_uri} \\" if line.startswith("vllm @ ") else line
    for line in source.splitlines()
]
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
PY
"$uv_bin" pip sync --python "$stage/bin/python" "$stage/vllm-install.lock" \
  --require-hashes \
  --index-strategy unsafe-best-match \
  --extra-index-url "https://download.pytorch.org/whl/$VLLM_CUDA_VARIANT"
cp "$ROOT/release/vllm.lock" "$stage/requirements.lock"
rm -f -- "$stage/vllm-install.lock"
rm -f -- "$wheel"
"$stage/bin/python" -c 'import torch, vllm; assert torch.cuda.is_available(); print("vLLM", vllm.__version__, "CUDA", torch.version.cuda)'
mkdir -p "$stage/licenses"
shopt -s nullglob
vllm_license=("$stage"/lib/python*/site-packages/vllm-*.dist-info/licenses/LICENSE)
shopt -u nullglob
[[ "${#vllm_license[@]}" -eq 1 ]] || {
  printf 'ERROR: The installed vLLM package has no unique license file.\n' >&2
  exit 1
}
cp "${vllm_license[0]}" "$stage/licenses/vLLM-LICENSE"
cp "$ROOT/THIRD_PARTY_NOTICES.md" "$stage/licenses/THIRD_PARTY_NOTICES.md"
"$uv_bin" pip freeze --python "$stage/bin/python" | sort > "$stage/installed-packages.txt"
"$stage/bin/vllm" --version
old_backend=""
if [[ -e "$backend" ]]; then
  old_backend="$INSTALL_ROOT/backends/.old-vllm-$VLLM_VERSION-$$"
  mv "$backend" "$old_backend"
fi
mv "$stage" "$backend"
"$backend/bin/vllm" --version
record_backend
if [[ -n "$old_backend" ]]; then
  rm -rf -- "$old_backend"
fi
rm -rf -- "$uv_stage"
trap - EXIT
printf 'vLLM %s is installed at %s.\n' "$VLLM_VERSION" "$backend"
