#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(tr -d '[:space:]' < "$ROOT/VERSION")"
DIST="${LEMUR_DIST_DIR:-$ROOT/dist}"
NAME="lemur-linux-x86_64"

resolved_dist="$(realpath -m "$DIST")"
resolved_root="$(realpath -m "$ROOT")"
[[ "$resolved_dist" == "$resolved_root"/* ]] || {
  printf 'ERROR: The release output directory must be inside the repository.\n' >&2
  exit 1
}

"$ROOT/scripts/check-release.sh"
rm -rf -- "$DIST"
mkdir -p "$DIST"
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
stage="$tmp_dir/lemur-$VERSION"
mkdir -p "$stage"

git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null
git -C "$ROOT" config --global --add safe.directory "$ROOT" >/dev/null 2>&1 || true
treeish="${LEMUR_GIT_TREEISH:-HEAD}"
git -C "$ROOT" archive --format=tar "$treeish" | tar -xf - -C "$stage"

if [[ -z "$(find "$stage" -mindepth 1 -print -quit)" ]]; then
  printf 'ERROR: The release archive stage is empty.\n' >&2
  exit 1
fi

archive="$DIST/$NAME.tar.gz"
tar -czf "$archive" -C "$tmp_dir" "lemur-$VERSION"
sha256sum "$archive" | sed "s|$archive|$NAME.tar.gz|" > "$archive.sha256"
cp "$ROOT/install.sh" "$DIST/install.sh"
cp "$ROOT/LICENSE" "$DIST/LICENSE"

archive_sha="$(sha256sum "$archive" | awk '{print $1}')"
archive_size="$(stat -c '%s' "$archive")"
python3 - "$ROOT/release/manifest.example.json" "$DIST/manifest.json" "$archive_sha" "$archive_size" <<'PY'
import json
import sys

with open(sys.argv[1]) as source:
    manifest = json.load(source)
manifest["sha256"] = sys.argv[3]
manifest["size"] = int(sys.argv[4])
version = manifest.get("version", "")
if version:
    manifest["archive_url"] = (
        f"https://github.com/ironton-engineering/lemur/releases/download/v{version}/"
        f"{manifest['archive']}"
    )
with open(sys.argv[2], "w") as output:
    json.dump(manifest, output, indent=2)
    output.write("\n")
PY

printf 'Built release assets in %s\n' "$DIST"
sha256sum "$DIST"/*
