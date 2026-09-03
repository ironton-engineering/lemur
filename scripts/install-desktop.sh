#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${LEMUR_BIN_DIR:-$HOME/.local/bin}"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}"
APPS_DIR="$DATA_DIR/applications"
ICON_DIR="$DATA_DIR/icons/hicolor/scalable/apps"
APPLICATION_ID="io.github.ironton_engineering.Lemur"
DESKTOP_FILE="$APPS_DIR/$APPLICATION_ID.desktop"

mkdir -p "$APPS_DIR" "$ICON_DIR"

sed "s|@BIN@|$BIN_DIR|g" "$ROOT/lemur.desktop.in" > "$DESKTOP_FILE"
# Remove the old desktop-file name so that the menu does not show Lemur twice.
rm -f -- "$APPS_DIR/lemur.desktop"
rm -f -- "$ICON_DIR/lemur.svg"
cp "$ROOT/assets/lemur.png" "$ICON_DIR/lemur.png"
chmod 0644 "$DESKTOP_FILE" "$ICON_DIR/lemur.png"

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPS_DIR"
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f -t "$DATA_DIR/icons/hicolor" >/dev/null 2>&1 || true
fi

echo "Installed the Lemur application-menu entry:"
echo "  $DESKTOP_FILE"
