#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

./release/build-release.sh

version="v$(tr -d '[:space:]' < VERSION)"
gh release view "$version" --repo ironton-engineering/lemur >/dev/null 2>&1 \
  || gh release create "$version" --prerelease --generate-notes --repo ironton-engineering/lemur
gh release upload "$version" dist/* --clobber --repo ironton-engineering/lemur

echo "Published $version"
