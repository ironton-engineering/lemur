# Third-Party Notices

Lemur installs third-party software. Lemur does not change the upstream license of that software.

## llama.cpp

- Version: `b10516`
- Commit: `b95502ba9aa0eb73a2f4fc8878d7fbe6a847a0b9`
- Source: <https://github.com/ggml-org/llama.cpp/tree/b10516>
- License: MIT
- Copyright: 2023-2026 The ggml authors

The installer builds llama.cpp from its tagged source. It installs the upstream `LICENSE` file beside the installed backend.

## vLLM

- Version: `0.26.0`
- Source: <https://github.com/vllm-project/vllm/tree/v0.26.0>
- License: Apache License 2.0
- Copyright: vLLM contributors

vLLM is optional. The installer uses the official Python package. It records all installed packages and copies the vLLM license data into the installed notice directory.

## Direct Python Dependencies

| Package | Required version | License | Source |
|---|---:|---|---|
| FastAPI | `>=0.115,<1` | MIT | <https://github.com/fastapi/fastapi> |
| Uvicorn | `>=0.32,<1` | BSD-3-Clause | <https://github.com/Kludex/uvicorn> |
| Pydantic | `>=2,<3` | MIT | <https://github.com/pydantic/pydantic> |
| HTTPX | `>=0.27,<1` | BSD-3-Clause | <https://github.com/encode/httpx> |

Each Python distribution keeps its package metadata and license files in its installed distribution directory.

## Assets

The Lemur icon and the other files in `assets/` are part of Lemur. They use the Lemur Apache License 2.0.
