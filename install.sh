#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY="ironton-engineering/lemur"
readonly BOOTSTRAP_VERSION="0.1.0"
readonly ARCHIVE_NAME="lemur-linux-x86_64.tar.gz"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ "${1:-}" == "--version" ]]; then
  printf 'Lemur installer %s\n' "$BOOTSTRAP_VERSION"
  exit 0
fi

if [[ -n "${LEMUR_SOURCE_DIR:-}" ]]; then
  source_dir="$(cd "$LEMUR_SOURCE_DIR" && pwd)"
  [[ -x "$source_dir/scripts/install-release.sh" ]] || \
    die "LEMUR_SOURCE_DIR does not contain scripts/install-release.sh"
  exec "$source_dir/scripts/install-release.sh" "$@"
fi

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
command -v tar >/dev/null 2>&1 || die "tar is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf -- "$tmp_dir"
}
trap cleanup EXIT

resolve_release_base_url() {
  if [[ -n "${LEMUR_RELEASE_URL:-}" ]]; then
    printf '%s' "$LEMUR_RELEASE_URL"
    return
  fi
  if [[ -n "${LEMUR_RELEASE_TAG:-}" ]]; then
    printf 'https://github.com/%s/releases/download/%s' "$REPOSITORY" "$LEMUR_RELEASE_TAG"
    return
  fi
  python3 - "$REPOSITORY" <<'PY'
import json
import sys
import urllib.request

repo = sys.argv[1]
with urllib.request.urlopen(
    f"https://api.github.com/repos/{repo}/releases?per_page=1"
) as response:
    releases = json.load(response)
if not releases:
    raise SystemExit("ERROR: No published Lemur release was found")
print(f"https://github.com/{repo}/releases/download/{releases[0]['tag_name']}")
PY
}

base_url="$(resolve_release_base_url)"
archive="$tmp_dir/$ARCHIVE_NAME"
checksum="$archive.sha256"
manifest="$tmp_dir/manifest.json"

printf 'Download the latest Lemur release.\n'
curl --fail --location --proto '=https' --tlsv1.2 \
  "$base_url/$ARCHIVE_NAME" --output "$archive"
curl --fail --location --proto '=https' --tlsv1.2 \
  "$base_url/$ARCHIVE_NAME.sha256" --output "$checksum"
curl --fail --location --proto '=https' --tlsv1.2 \
  "$base_url/manifest.json" --output "$manifest"

expected="$(awk 'NR == 1 {print $1}' "$checksum")"
[[ "$expected" =~ ^[0-9a-fA-F]{64}$ ]] || die "The release checksum is invalid"
actual="$(sha256sum "$archive" | awk '{print $1}')"
[[ "$actual" == "$expected" ]] || die "The release archive checksum does not match"
python3 - "$manifest" "$ARCHIVE_NAME" "$actual" "$archive" <<'PY'
import json
import sys
from pathlib import Path

try:
    manifest = json.loads(Path(sys.argv[1]).read_text())
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"ERROR: The release manifest is invalid: {error}")

archive = Path(sys.argv[4])
if manifest.get("archive") != sys.argv[2]:
    raise SystemExit("ERROR: The release manifest names a different archive")
if manifest.get("sha256") != sys.argv[3]:
    raise SystemExit("ERROR: The release manifest checksum does not match")
if manifest.get("size") != archive.stat().st_size:
    raise SystemExit("ERROR: The release manifest size does not match")
PY

mkdir -p "$tmp_dir/source"
tar -xzf "$archive" -C "$tmp_dir/source" --strip-components=1
installer="$tmp_dir/source/scripts/install-release.sh"
[[ -x "$installer" ]] || die "The release archive has no installer"

trap - EXIT
"$installer" "$@"
status=$?
cleanup
exit "$status"
