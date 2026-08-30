# Contributing to Lemur

Thank you for helping Lemur.

Open an issue for a large change. Describe the problem, the supported system, and the intended result. Do not include private prompts, model files, access tokens, or personal paths.

Run these checks:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/python -m pytest -q
./scripts/check-release.sh
```

Add tests for behavior changes. Keep all supported Ubuntu and GPU generations in scope.

- Keep each pull request focused.
- Explain user-visible changes.
- State the commands that you used for checks.
- Update licenses and notices when a dependency changes.
- Do not add model files, generated environments, caches, or logs.
- Disclose generated code and review every submitted line.

Contributions use the Apache License 2.0 under the project license.
