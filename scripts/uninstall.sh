#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT="${LEMUR_INSTALL_ROOT:-$HOME/.local/share/llm-hub}"
BIN_DIR="${LEMUR_BIN_DIR:-$HOME/.local/bin}"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/lemur"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}"
REMOVE_USER_DATA=0
ASSUME_YES=0

validate_recursive_target() {
  local resolved home_resolved
  resolved="$(realpath -m "$1")"
  home_resolved="$(realpath -m "$HOME")"
  [[ "$resolved" == "$home_resolved"/* ]] || {
    printf 'ERROR: Refusing to remove a directory outside the user home directory: %s\n' "$1" >&2
    exit 1
  }
  case "$resolved" in
    "$home_resolved"|"$home_resolved/.local"|"$home_resolved/.local/share"|"$home_resolved/.config")
      printf 'ERROR: Refusing to remove a broad user directory: %s\n' "$1" >&2
      exit 1
      ;;
  esac
}

while (($#)); do
  case "$1" in
    --remove-user-data) REMOVE_USER_DATA=1 ;;
    --yes) ASSUME_YES=1 ;;
    --help|-h)
      printf 'Usage: lemur uninstall [--remove-user-data] [--yes]\n'
      exit 0
      ;;
    *) printf 'ERROR: Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

targets=(
  "$INSTALL_ROOT"
  "$BIN_DIR/lemur"
  "$DATA_DIR/applications/lemur.desktop"
  "$DATA_DIR/applications/io.github.ironton_engineering.Lemur.desktop"
  "$DATA_DIR/icons/hicolor/scalable/apps/lemur.svg"
  "$DATA_DIR/icons/hicolor/scalable/apps/lemur.png"
)
if [[ "$REMOVE_USER_DATA" -eq 1 ]]; then
  targets+=("$CONFIG_DIR")
fi
validate_recursive_target "$INSTALL_ROOT"
if [[ "$REMOVE_USER_DATA" -eq 1 ]]; then
  validate_recursive_target "$CONFIG_DIR"
fi

printf 'Lemur will remove these paths:\n'
printf '  %s\n' "${targets[@]}"
if [[ "$REMOVE_USER_DATA" -eq 0 ]]; then
  printf 'User settings and logs will stay at %s.\n' "$CONFIG_DIR"
fi
printf 'Model files will not be removed.\n'

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p 'Continue? [y/N] ' answer
  [[ "$answer" == "y" || "$answer" == "Y" ]] || exit 0
fi

for target in "${targets[@]}"; do
  case "$target" in
    "$INSTALL_ROOT"|"$CONFIG_DIR") rm -rf -- "$target" ;;
    *) rm -f -- "$target" ;;
  esac
done
printf 'Lemur was removed.\n'
