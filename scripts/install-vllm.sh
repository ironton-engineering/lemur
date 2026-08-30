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

if [[ -x "$backend/bin/vllm" ]]; then
  printf 'vLLM %s is already installed.\n' "$VLLM_VERSION"
  exit 0
fi

mkdir -p "$INSTALL_ROOT/backends"
stage="$INSTALL_ROOT/backends/.stage-vllm-$VLLM_VERSION-$$"
wheel="$stage/vllm.whl"
trap 'rm -rf -- "$stage"' EXIT
python3 -m venv "$stage"
"$stage/bin/pip" install --disable-pip-version-check "uv==$UV_VERSION"
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
    f"vllm @ {wheel_uri}" if line.startswith("vllm @ ") else line
    for line in source.splitlines()
]
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
PY
"$stage/bin/uv" pip sync --python "$stage/bin/python" "$stage/vllm-install.lock" \
  --require-hashes \
  --extra-index-url "https://download.pytorch.org/whl/$VLLM_CUDA_VARIANT"
cp "$ROOT/release/vllm.lock" "$stage/requirements.lock"
rm -f -- "$stage/vllm-install.lock"
rm -f -- "$wheel"
"$stage/bin/python" -c 'import torch, vllm; assert torch.cuda.is_available(); print("vLLM", vllm.__version__, "CUDA", torch.version.cuda)'
mkdir -p "$stage/licenses"
cp "$ROOT/LICENSE" "$stage/licenses/Apache-2.0-LICENSE"
cp "$ROOT/THIRD_PARTY_NOTICES.md" "$stage/licenses/THIRD_PARTY_NOTICES.md"
"$stage/bin/pip" freeze | sort > "$stage/installed-packages.txt"
mv "$stage" "$backend"
trap - EXIT
printf 'vLLM %s is installed at %s.\n' "$VLLM_VERSION" "$backend"
