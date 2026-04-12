# AGENTS.md

This repository contains `stprobe`, a minimal Rust CLI for inspecting `.safetensors` files.

## Goal

Keep the tool small, fast, and predictable.

`stprobe` is intentionally closer to `ffprobe` than to a model management toolkit:

- one input: a local file path or `http(s)` URL
- one default plain-text output format
- header-only inspection
- no Python dependency
- no tensor payload loading

## Current Scope

The CLI currently supports:

- `stprobe <file>`
- `stprobe <http(s)-url>`
- `stprobe --help`
- `stprobe --version`

It reports:

- file path
- file size
- tensor count
- metadata
- per-tensor name, dtype, shape, parameter count, and byte size
- total parameter count
- total tensor bytes
- dtype breakdown

## Non-Goals

Do not casually expand the scope.

These are currently out of scope unless explicitly requested:

- subcommands
- JSON output
- Python bindings
- remote directory browsing
- automatic Hugging Face repo inspection
- TUI output
- complex diagnostics

If a change pushes the project toward a general safetensors toolbox, stop and reconsider.

## Repository Layout

- `src/main.rs`: thin CLI wrapper
- `src/lib.rs`: inspection logic, rendering, and remote probing
- `tests/cli.rs`: CLI integration tests
- `tests/remote_http.rs`: mock HTTP range tests
- `tests/common/mod.rs`: small generated safetensors fixtures
- `.github/workflows/ci.yml`: lint and test matrix
- `.github/workflows/release.yml`: tagged multi-platform release builds
- `.github/workflows/hf-smoke.yml`: real Hugging Face smoke test

## Development Rules

- Use Rust stable.
- Keep the implementation simple and readable.
- Prefer deterministic output over clever formatting.
- Do not add thousands separators or locale-sensitive formatting by default.
- Do not load full tensor data just to inspect metadata.
- For remote files, prefer HTTP range requests and header-only probing.
- Hugging Face support should work well through standard `resolve` URLs first.

## Testing Expectations

Before shipping code changes, run:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

When touching remote probing:

- keep main correctness tests local and mocked
- do not make PR correctness depend on live Hugging Face availability
- use the separate Hugging Face smoke workflow for real-network verification

Test fixtures should stay small.
Do not commit large `.safetensors` files to the repository.
Generate tiny fixtures in tests where possible.

## Output Stability

CLI output is part of the product.

Preserve:

- section names
- ordering
- indentation
- wording style

Avoid cosmetic churn unless there is a clear user benefit.

## Release Notes

Releases are built from Git tags matching `v*`.

Typical release flow:

1. Bump `version` in `Cargo.toml`
2. Regenerate `Cargo.lock` if needed
3. Run the validation commands
4. Commit the version bump
5. Create and push an annotated tag like `v0.2.1`

The release workflow publishes assets for:

- Linux x86_64
- macOS Intel
- macOS Apple Silicon
- Windows x86_64

## Practical Guidance For Agents

- Read `README.md` before changing user-facing behavior.
- Prefer editing `src/lib.rs` for feature work.
- Keep `src/main.rs` thin.
- If you add behavior, add or update tests in the same change.
- If you touch GitHub Actions, prefer current supported runner labels and current major versions of official `actions/*` steps.
- If you need a real model example, prefer public Hugging Face files with fixed revisions.
