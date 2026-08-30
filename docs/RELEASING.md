# Release Guide

## Prepare

1. Update `VERSION`.
2. Update all values in `release/versions.env`.
3. Update `THIRD_PARTY_NOTICES.md`.
4. Run `python3 -m pytest -q`.
5. Run `./scripts/check-release.sh`.
6. Run `./release/build-release.sh`.
7. Check every file in `dist/manifest.json` and `dist/*.sha256`.

## Hardware Checks

Run a clean default install on Ubuntu 22.04 with an RTX 30-series GPU. Run another clean default install on Ubuntu 24.04 with an RTX 40- or 50-series GPU.

For each machine, check install, desktop start, `lemur doctor`, model start, update, rollback, and uninstall. Check optional vLLM on each GPU generation in the release matrix.

## Publish

1. Create a protected or signed `v<VERSION>` tag.
2. Let the release workflow build the assets.
3. Publish the release as a pre-release.
4. Run the public install command from the release.
5. Promote the release only after the hardware checks pass.

The required assets are `install.sh`, `lemur-linux-x86_64.tar.gz`, its SHA-256 file, `manifest.json`, and `LICENSE`.
