#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPS_DIR="$HOME/.local/share/applications"
DESKTOP_DIR="${XDG_DESKTOP_DIR:-$HOME/Desktop}"

chmod +x "$ROOT/scripts/run.sh"

mkdir -p "$APPS_DIR" "$DESKTOP_DIR"

sed "s|@ROOT@|$ROOT|g" "$ROOT/lemur.desktop.in" > "$ROOT/lemur.desktop"

ln -sf "$ROOT/lemur.desktop" "$APPS_DIR/lemur.desktop"
ln -sf "$ROOT/lemur.desktop" "$DESKTOP_DIR/Lemur.desktop"

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPS_DIR"
fi

if command -v gio >/dev/null 2>&1; then
  gio set "$DESKTOP_DIR/Lemur.desktop" metadata::trusted true 2>/dev/null || true
fi

echo "Installed desktop shortcuts (symlinked to repo — edits apply immediately):"
echo "  $APPS_DIR/lemur.desktop"
echo "  $DESKTOP_DIR/Lemur.desktop"
