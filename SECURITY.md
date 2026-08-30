# Security Policy

## Supported Versions

Security fixes are provided for the latest Lemur release.

## Report a Vulnerability

Use [GitHub private vulnerability reporting](https://github.com/ironton-engineering/lemur/security/advisories/new). Do not put exploit details or private data in a public issue.

Include the Lemur version, Ubuntu version, GPU, NVIDIA driver, affected command, and a small reproduction. Remove home-directory names, model prompts, tokens, and other private data.

## Local Security Rules

- Lemur binds to localhost by default.
- LAN endpoints have no authentication.
- Do not expose Lemur or a model port to the public internet.
- Treat model files as untrusted input.
- Use models only from sources that you trust.
- Review every `sudo` command shown by the installer.
- Lemur does not install or change the NVIDIA driver.
