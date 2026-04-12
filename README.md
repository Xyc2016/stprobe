# stprobe

`stprobe` is a small Rust CLI that works like `ffprobe` for `.safetensors` files.

Point it at a model file and it prints a stable plain-text summary:

- file path
- file size
- tensor count
- metadata
- per-tensor name, dtype, shape, parameter count, and byte size
- total parameters
- total tensor bytes
- dtype breakdown

It reads the safetensors header and metadata only. It does not load tensor payloads into memory.

## Install

```bash
cargo install stprobe
```

## Quick Start

```bash
stprobe model.safetensors
```

Example input:

```text
$ stprobe model.safetensors
```

Example output:

```text
File: model.safetensors
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

## Why Not A One-Off Script

You can inspect a safetensors file with a short Rust or Python script, but `stprobe` is better when you want something reusable:

- stable output you can read, diff, grep, and paste into issues
- one command instead of re-editing ad hoc scripts
- no Python environment or notebook setup
- uses the official `safetensors` crate instead of custom header parsing
- avoids loading full tensor data when you only need structure and counts

## Build From Source

```bash
cargo install --path .
```

Or run it without installing:

```bash
cargo run -- model.safetensors
```
