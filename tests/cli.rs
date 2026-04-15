mod common;

use std::io::{BufRead, BufReader};
use std::process::Stdio;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

#[test]
fn prints_local_file_summary() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("sample.safetensors");
    common::write_sample_file(&path);
    let path_text = path.to_string_lossy().to_string();

    Command::cargo_bin("stprobe")
        .expect("build stprobe binary")
        .arg(&path)
        .assert()
        .success()
        .stdout(
            predicate::str::contains(format!("File: {path_text}"))
                .and(predicate::str::contains("Tensors: 2"))
                .and(predicate::str::contains("Parameters: 4"))
                .and(predicate::str::contains("Tensor-Bytes: 24"))
                .and(predicate::str::contains("  format = pt"))
                .and(predicate::str::contains("  F32: 8 bytes"))
                .and(predicate::str::contains("  I64: 16 bytes"))
                .and(predicate::str::contains("  embedding.ids"))
                .and(predicate::str::contains("  embedding.weight")),
        );
}

#[test]
fn returns_nonzero_for_invalid_files() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("broken.safetensors");
    std::fs::write(&path, b"not a safetensors file").expect("write invalid input");

    Command::cargo_bin("stprobe")
        .expect("build stprobe binary")
        .arg(&path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("Error: invalid safetensors file:"));
}

#[test]
fn exits_successfully_when_stdout_pipe_closes_early() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("large-output.safetensors");
    common::write_large_sample_file(&path, 3_000);

    let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin("stprobe"))
        .arg(&path)
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn stprobe");

    let stdout = child.stdout.take().expect("stdout pipe");
    let mut reader = BufReader::new(stdout);
    let mut first_line = String::new();
    reader
        .read_line(&mut first_line)
        .expect("read first output line");
    assert!(first_line.starts_with("File: "));
    drop(reader);

    let status = child.wait().expect("wait for stprobe");
    assert!(status.success(), "expected success, got {status}");
}
