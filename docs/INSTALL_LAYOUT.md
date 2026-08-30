# Lemur Install Layout

Lemur uses only user-owned files, except for Ubuntu packages that the user approves.

## Application Files

```text
~/.local/share/llm-hub/
├── current -> releases/<active-version>
├── previous -> releases/<prior-version>
├── releases/
│   └── <version>/
│       ├── .venv/
│       ├── server/
│       ├── static/
│       └── scripts/
├── backends/
│   ├── llama.cpp-<version>/
│   └── vllm-<version>-<cuda-variant>/
└── notices/
```

Application releases and backend directories are immutable after a successful install. An install uses a staging directory. The active link changes only after all required checks pass.

## Commands and Desktop Files

```text
~/.local/bin/lemur
~/.local/share/applications/lemur.desktop
~/.local/share/icons/hicolor/scalable/apps/lemur.svg
```

## Mutable User Data

```text
~/.config/lemur/
├── state.json
├── server.log
├── launch.log
└── server.pid
```

Update and rollback operations preserve this directory. Normal uninstall also preserves it. The `--remove-user-data` option removes it after explicit approval.

## Files That Lemur Does Not Own

Lemur does not own model files. Uninstall and update commands must not remove files outside the paths in this document. Lemur also does not own the NVIDIA driver or the Ubuntu CUDA packages.
