# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through GitHub's
[private vulnerability reporting](https://github.com/ZouhairMudakka/clmkit/security/advisories/new).
Include steps to reproduce, affected versions and the impact you expect. You should get an acknowledgement
within a few days.

## Supported versions

clmkit is pre-1.0. Security fixes land on `main` and in the next release.

## Hardening defaults

- No pickle: indexes are `.npy` (loaded with `allow_pickle=False`) plus JSON, and configs use `yaml.safe_load`.
- `trust_remote_code=False` by default. Pin `revision=` for hub models you depend on.
- The REST server binds `127.0.0.1`. Set `CLMKIT_API_KEY` for bearer auth before exposing it, and add a gateway for rate limiting.
- Write endpoints and write tools are disabled unless explicitly enabled.
- The OpenAI-compatible client refuses HTTP redirects, so API keys can't be forwarded to another host.

See [docs/AUDIT.md](docs/AUDIT.md) for the full review and known residual risks. The most important one: **text retrieved into an agent's context can contain prompt injections.**
