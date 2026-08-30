#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SOURCE_ROOT/release/versions.env"

INSTALL_ROOT="${LEMUR_INSTALL_ROOT:-$HOME/.local/share/llm-hub}"
BIN_DIR="${LEMUR_BIN_DIR:-$HOME/.local/bin}"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/lemur"
VERSION="$(tr -d '[:space:]' < "$SOURCE_ROOT/VERSION")"
WITH_VLLM=0
WITH_DESKTOP=1
ASSUME_YES=0
NONINTERACTIVE=0
TEST_MODE="${LEMUR_TEST_MODE:-0}"

validate_user_directory() {
  local value resolved home_resolved
  value="$1"
  resolved="$(realpath -m "$value")"
  home_resolved="$(realpath -m "$HOME")"
  [[ "$resolved" == "$home_resolved"/* ]] || die "Install paths must be inside the user home directory"
  [[ "$resolved" != "$home_resolved" ]] || die "The user home directory cannot be an install target"
}

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: install.sh [options]

  --with-vllm      Install the optional vLLM backend.
  --no-desktop     Do not install the desktop launcher.
  --yes            Approve the shown Ubuntu package commands.
  --non-interactive  Stop if approval is required.
  --version        Show the Lemur version.
  --help           Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --with-vllm) WITH_VLLM=1 ;;
    --no-desktop) WITH_DESKTOP=0 ;;
    --yes) ASSUME_YES=1 ;;
    --non-interactive) NONINTERACTIVE=1 ;;
    --version) say "Lemur $VERSION"; exit 0 ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
  shift
done

[[ "$(id -u)" -ne 0 ]] || die "Run the installer as a normal user, not as root"
validate_user_directory "$INSTALL_ROOT"
validate_user_directory "$BIN_DIR"
validate_user_directory "$CONFIG_DIR"

approve() {
  local prompt="$1"
  if [[ "$ASSUME_YES" -eq 1 ]]; then
    return 0
  fi
  [[ "$NONINTERACTIVE" -eq 0 ]] || die "Approval is required: $prompt"
  read -r -p "$prompt [y/N] " answer
  [[ "$answer" == "y" || "$answer" == "Y" ]] || die "Installation was cancelled"
}

if [[ "$TEST_MODE" != "1" && "${LEMUR_SKIP_SYSTEM_CHECK:-0}" != "1" ]]; then
  python3 "$SOURCE_ROOT/scripts/system_probe.py" --install-root "$INSTALL_ROOT"
fi

# shellcheck disable=SC1091
os_version="$(. /etc/os-release && printf '%s' "$VERSION_ID")"
base_packages=(python3 python3-venv curl git cmake ninja-build build-essential pkg-config libcurl4-openssl-dev)
if [[ "$WITH_DESKTOP" -eq 1 ]]; then
  base_packages+=(gir1.2-gtk-3.0)
  if [[ "$os_version" == "22.04" ]]; then
    base_packages+=(gir1.2-webkit2-4.0)
  else
    base_packages+=(gir1.2-webkit2-4.1)
  fi
fi

missing_packages=()
if [[ "$TEST_MODE" != "1" ]]; then
  for package in "${base_packages[@]}"; do
    dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'ok installed' || \
      missing_packages+=("$package")
  done
fi
if ((${#missing_packages[@]})); then
  say "The installer must run this command:"
  printf '  sudo apt-get update && sudo apt-get install -y'
  printf ' %q' "${missing_packages[@]}"
  printf '\n'
  approve "Install the missing Ubuntu packages?"
  sudo apt-get update
  sudo apt-get install -y "${missing_packages[@]}"
fi

cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
if [[ "$TEST_MODE" != "1" && ! -x "$cuda_home/bin/nvcc" ]]; then
  distro="ubuntu${os_version//./}"
  keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb"
  if [[ "$os_version" == "22.04" ]]; then
    keyring_sha="$CUDA_KEYRING_SHA256_UBUNTU2204"
  else
    keyring_sha="$CUDA_KEYRING_SHA256_UBUNTU2404"
  fi
  say "The CUDA compiler is missing."
  say "The installer must add NVIDIA's signed CUDA package source and run:"
  say "  sudo apt-get update && sudo apt-get install -y cuda-toolkit-$CUDA_TOOLKIT_VERSION"
  say "This package does not install or change the NVIDIA driver."
  approve "Install CUDA Toolkit 12.8?"
  tmp_keyring="$(mktemp)"
  trap 'rm -f -- "$tmp_keyring"' EXIT
  curl --fail --location --proto '=https' --tlsv1.2 "$keyring_url" --output "$tmp_keyring"
  printf '%s  %s\n' "$keyring_sha" "$tmp_keyring" | sha256sum --check --status || \
    die "The NVIDIA CUDA keyring checksum does not match"
  sudo dpkg -i "$tmp_keyring"
  sudo apt-get update
  sudo apt-get install -y "cuda-toolkit-$CUDA_TOOLKIT_VERSION"
  rm -f -- "$tmp_keyring"
  trap - EXIT
fi
if [[ "$TEST_MODE" != "1" ]]; then
  [[ -x "$cuda_home/bin/nvcc" ]] || die "CUDA Toolkit 12.8 did not provide nvcc"
fi

mkdir -p "$INSTALL_ROOT/releases" "$INSTALL_ROOT/backends" "$BIN_DIR" "$CONFIG_DIR"
release_dir="$INSTALL_ROOT/releases/$VERSION"
if [[ ! -d "$release_dir" ]]; then
  stage="$INSTALL_ROOT/releases/.stage-$VERSION-$$"
  rm -rf -- "$stage"
  mkdir -p "$stage"
  (
    cd "$SOURCE_ROOT"
    tar --exclude='.git' --exclude='.venv' --exclude='.planning' \
      --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='.cursor' \
      --exclude='__pycache__' --exclude='*.pyc' -cf - .
  ) | tar -xf - -C "$stage"
  if [[ "$TEST_MODE" != "1" ]]; then
    python3 -m venv "$stage/.venv"
    "$stage/.venv/bin/pip" install --disable-pip-version-check --require-hashes \
      -r "$stage/requirements.lock"
  fi
  mv "$stage" "$release_dir"
fi

llama_root="$INSTALL_ROOT/backends/llama.cpp-$LLAMA_CPP_VERSION"
if [[ "$TEST_MODE" != "1" && ! -x "$llama_root/bin/llama-server" ]]; then
  llama_stage="$INSTALL_ROOT/backends/.stage-llama.cpp-$LLAMA_CPP_VERSION-$$"
  rm -rf -- "$llama_stage"
  git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "$llama_stage/source"
  git -C "$llama_stage/source" checkout --detach "$LLAMA_CPP_COMMIT"
  actual_commit="$(git -C "$llama_stage/source" rev-parse HEAD)"
  [[ "$actual_commit" == "$LLAMA_CPP_COMMIT" ]] || die "The llama.cpp commit does not match"
  caps="$(python3 "$SOURCE_ROOT/scripts/system_probe.py" --install-root "$INSTALL_ROOT" --cuda-architectures)"
  [[ -n "$caps" ]] || die "No supported CUDA architecture was found"
  CUDACXX="$cuda_home/bin/nvcc" cmake -S "$llama_stage/source" -B "$llama_stage/build" -G Ninja \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$caps" \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$llama_stage/build" --target llama-server --parallel
  mkdir -p "$llama_stage/bin" "$llama_stage/licenses"
  cp -a "$llama_stage/build/bin/." "$llama_stage/bin/"
  cp "$llama_stage/source/LICENSE" "$llama_stage/licenses/llama.cpp-LICENSE"
  printf '%s\n' "$LLAMA_CPP_COMMIT" > "$llama_stage/COMMIT"
  mv "$llama_stage" "$llama_root"
fi
if [[ "$TEST_MODE" != "1" ]]; then
  "$llama_root/bin/llama-server" --version
fi

if [[ -L "$INSTALL_ROOT/current" ]]; then
  old_target="$(readlink -f "$INSTALL_ROOT/current")"
  if [[ "$old_target" != "$release_dir" ]]; then
    ln -sfn "$old_target" "$INSTALL_ROOT/previous"
  fi
fi
ln -sfn "$release_dir" "$INSTALL_ROOT/current"
ln -sfn "$INSTALL_ROOT/current/scripts/lemur" "$BIN_DIR/lemur"

mkdir -p "$INSTALL_ROOT/notices"
cp "$release_dir/LICENSE" "$INSTALL_ROOT/notices/Lemur-LICENSE"
cp "$release_dir/NOTICE" "$INSTALL_ROOT/notices/Lemur-NOTICE"
cp "$release_dir/THIRD_PARTY_NOTICES.md" "$INSTALL_ROOT/notices/THIRD_PARTY_NOTICES.md"

if [[ "$WITH_DESKTOP" -eq 1 && "$TEST_MODE" != "1" ]]; then
  "$release_dir/scripts/install-desktop.sh"
fi
if [[ "$WITH_VLLM" -eq 1 && "$TEST_MODE" != "1" ]]; then
  "$release_dir/scripts/install-vllm.sh"
fi

say "Lemur $VERSION is installed."
say "Run: $BIN_DIR/lemur"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  say "Add $BIN_DIR to PATH to use the lemur command without its full path."
fi
