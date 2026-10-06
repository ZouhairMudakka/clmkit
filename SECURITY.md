# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through GitHub's
[private vulnerability reporting](https://github.com/ZouhairMudakka/clmkit/security/advisories/new).
Include steps to reproduce, affected versions and the impact you expect. You should get an acknowledgement
within a few days.

## Supported versions

clmkit is pre-1.0. Security fixes land on `main` and in the next release.

## Hardening defaults

- NumPy indexes use `.npy` with `allow_pickle=False` plus JSON; YAML uses `safe_load`. FAISS loads native binary indexes: load only trusted, complete snapshots. Validation is not a sandbox for hostile native files.
- HF encoders, rerankers and PEFT adapters require **safetensors** weights; legacy `.bin` checkpoints and implicit Transformers adapter loading are rejected. Explicit adapters use `HFEncoder(base, adapter=...)`. Prebuilt model objects are caller-controlled code.
- Supported ML versions start at torch 2.13, transformers 5.17, PEFT 0.21.1 and safetensors 0.8, with major-version bounds in `pyproject.toml`. The minimum stack and current resolved stack are tested separately; these floors do not guarantee absence of future advisories. See the [PyTorch checkpoint advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6) motivating removal of the old permissive floor.
- `trust_remote_code=False` by default. Pin `revision=` for hub models you depend on.
- The REST server binds `127.0.0.1`. Set `CLMKIT_API_KEY` for bearer auth before exposing it, and add a gateway for rate limiting.
- REST write endpoints and `retriever_tools` writes are disabled by default. Calling `memory_tools(memory)` explicitly includes the `remember` write tool.
- The OpenAI-compatible client refuses HTTP redirects, so API keys can't be forwarded to another host.

Metadata filters restrict candidates; applications must derive tenant scope from authenticated identity. They are not authentication. Snapshots require a single writer and complete files; saves are not transactional. Multiple service workers do not share in-memory updates or locks.

See [current validation status](docs/VALIDATION_STATUS.md) for evidence and limitations. **Text retrieved into an agent's context can contain prompt injections.**
