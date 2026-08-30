# Privacy

Lemur has no telemetry. It does not send usage, model, prompt, response, GPU, or system data to the Lemur project.

Lemur reads local model metadata and stores settings, favorites, process identifiers, and logs under `~/.config/lemur`. Prompts and responses can appear in local process logs. Treat these logs as private data.

Lemur makes network requests during install and update. These requests go to GitHub, NVIDIA package servers, Python package servers, and PyTorch package servers. Model downloads are outside Lemur.

The local control service binds to `127.0.0.1` by default. LAN mode binds model endpoints to all network interfaces. These endpoints have no authentication.
