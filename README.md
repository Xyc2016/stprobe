# stprobe

`stprobe` is `ffprobe` for `.safetensors`.

It is a small Rust CLI for answering:

> "What is inside this safetensors file?"

Point it at a model file and it prints a stable plain-text summary without loading tensor payloads into memory.

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
- stable plain-text output that is easy to read, diff, grep, and paste

## Install

```bash
cargo install stprobe
```

## Quick Start

```bash
stprobe model.safetensors
```

Example:

```text
$ stprobe model.safetensors
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

## Try It On A Real Model

Download a public `.safetensors` file from Hugging Face:

```bash
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors -O all-MiniLM-L6-v2.safetensors
```

Inspect it:

```bash
stprobe all-MiniLM-L6-v2.safetensors
```

To get a feel for how quickly header-only inspection runs, time it:

```bash
time stprobe all-MiniLM-L6-v2.safetensors
```

Example:

```text
$ time stprobe all-MiniLM-L6-v2.safetensors
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

  ...

  pooler.dense.weight
    dtype: F32
    shape: [384, 384]
    numel: 147456
    bytes: 589824

stprobe all-MiniLM-L6-v2.safetensors  0.00s user 0.00s system 87% cpu 0.004 total
```

Example timing from one local machine. Exact results vary with filesystem cache, shell, and hardware.

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
