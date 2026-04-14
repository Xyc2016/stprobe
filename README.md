# stprobe

`stprobe` is `ffprobe` for `.safetensors`.

It is a small Rust CLI for answering:

> "What is inside this safetensors file?"

Point it at a model file and it prints a stable plain-text summary without loading tensor payloads into memory.

It works with local files and `http(s)` URLs, including Hugging Face `resolve` links.

Because it only probes the safetensors header and metadata, it is fast even for very large remote files.

`stprobe` shows:

- file path
- file size
- tensor count
- metadata
- per-tensor name, dtype, shape, parameter count, and byte size
- total parameters
- total tensor bytes
- dtype breakdown

It is built for model inspection, debugging, issue reports, and quick shell use.

- no Python environment
- no notebook setup
- no ad hoc parsing script
- header-only inspection with the official `safetensors` crate
- remote probing over HTTP range requests instead of downloading the full file
- stable plain-text output that is easy to read, diff, grep, and paste

## Install

Use a prebuilt binary from GitHub Releases if you just want to run `stprobe` without installing Rust:

- Download the archive for Linux x86_64, macOS Intel, macOS Apple Silicon, or Windows x86_64 from:
  https://github.com/Xyc2016/stprobe/releases/latest
- Extract it and run `stprobe --version`

Linux x86_64 example:

```bash
curl -L https://github.com/Xyc2016/stprobe/releases/latest/download/stprobe-x86_64-unknown-linux-gnu.tar.gz | tar -xz
./stprobe --version
```

Or install from crates.io:

```bash
cargo install stprobe --registry crates-io
```

## Quick Start

```bash
stprobe all-MiniLM-L6-v2.safetensors
```

Example:

```text
$ stprobe all-MiniLM-L6-v2.safetensors
File: all-MiniLM-L6-v2.safetensors
Size: 90868376 bytes
Tensors: 104
Parameters: 22713728
Tensor-Bytes: 90856960

Metadata:
  format = pt

DType Breakdown:
  F32: 90852864 bytes
  I64: 4096 bytes

Tensors:
  embeddings.position_ids
    dtype: I64
    shape: [1, 512]
    numel: 512
    bytes: 4096

  embeddings.word_embeddings.weight
    dtype: F32
    shape: [30522, 384]
    numel: 11720448
    bytes: 46881792
```

## Fast On Huge Remote Files

You can also probe a large Hugging Face model directly over HTTP without downloading the full `.safetensors` file first:

```bash
time stprobe https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors
```

Example:

```text
$ time stprobe https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors
File: https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors
Size: 17246524772 bytes
Tensors: 1442
Parameters: 16871188965
Tensor-Bytes: 17246298324

Metadata:
  modelspec.architecture = Flux.1-dev
  modelspec.author = Black Forest Labs
  modelspec.date = 2024-08-01
  modelspec.description = A guidance distilled rectified flow model.
  modelspec.license = FLUX.1 [dev] Non-Commercial License
  modelspec.title = Flux.1-dev

DType Breakdown:
  F16: 247300608 bytes
  F32: 335278740 bytes
  F8_E4M3: 16663718976 bytes

Tensors:
  text_encoders.clip_l.logit_scale
    dtype: F32
    shape: []
    numel: 1
    bytes: 4

  ...

  text_encoders.t5xxl.transformer.shared.weight
    dtype: F8_E4M3
    shape: [32128, 4096]
    numel: 131596288
    bytes: 131596288

stprobe https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors  0.01s user 0.02s system 2% cpu 1.277 total
```

Example timing from one local machine using the optimized build. Exact results vary with network, filesystem cache, shell, and hardware.

## Why Not A One-Off Script

You can inspect a safetensors file with a short Rust or Python script. `stprobe` is better when you want something you can keep using:

- one command instead of re-editing one-off scripts
- stable output for bug reports, CI logs, and terminal use
- no Python packaging, no notebook, no local helper code
- uses the official `safetensors` crate instead of custom header parsing
- reads only the safetensors header and metadata when that is all you need

## Build From Source

```bash
cargo install --path .
```

Or run it without installing:

```bash
cargo run -- model.safetensors
```
