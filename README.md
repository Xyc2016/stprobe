# stprobe

`stprobe` is a minimal Rust CLI for inspecting the header and tensor metadata inside a `.safetensors` file.

It prints a stable plain-text summary for a single `.safetensors` file, similar in spirit to `ffprobe`.

## Install

```bash
cargo install stprobe
```

## Build From Source

```bash
cargo install --path .
```

## Run

```bash
cargo run -- model.safetensors
stprobe model.safetensors
```

## What It Shows

- file path
- file size
- tensor count
- metadata
- each tensor's name, dtype, shape, parameter count, and byte size
- total parameters
- total tensor bytes
- dtype breakdown

## Example Output

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
```
