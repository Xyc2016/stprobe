use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use memmap2::MmapOptions;
use safetensors::{tensor::Metadata, SafeTensorError, SafeTensors};

#[derive(Parser, Debug)]
#[command(
    name = "stprobe",
    version,
    about = "Inspect basic metadata from a .safetensors file",
    long_about = None
)]
struct Cli {
    /// Path to a .safetensors file
    file: PathBuf,
}

#[derive(Debug)]
enum InspectError {
    FileNotFound(PathBuf),
    CannotRead {
        path: PathBuf,
        source: io::Error,
    },
    InvalidSafetensors {
        path: PathBuf,
        source: SafeTensorError,
    },
    Overflow {
        tensor: String,
    },
    MissingTensorInfo {
        tensor: String,
    },
}

impl std::fmt::Display for InspectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(path) => write!(f, "file not found: {}", path.display()),
            Self::CannotRead { path, source } => {
                write!(f, "failed to read file: {} ({source})", path.display())
            }
            Self::InvalidSafetensors { path, source } => {
                write!(f, "invalid safetensors file: {} ({source})", path.display())
            }
            Self::Overflow { tensor } => {
                write!(f, "tensor is too large to summarize safely: {tensor}")
            }
            Self::MissingTensorInfo { tensor } => {
                write!(f, "missing tensor metadata for: {tensor}")
            }
        }
    }
}

impl std::error::Error for InspectError {}

#[derive(Debug)]
struct Report {
    file_path: PathBuf,
    file_size: u64,
    tensor_count: usize,
    total_parameters: u128,
    total_tensor_bytes: u128,
    metadata: Vec<(String, String)>,
    dtype_breakdown: Vec<(String, u128)>,
    tensors: Vec<TensorSummary>,
}

#[derive(Debug)]
struct TensorSummary {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    numel: u128,
    bytes: u128,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match inspect_file(&cli.file) {
        Ok(report) => {
            print_report(&report);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn inspect_file(path: &Path) -> Result<Report, InspectError> {
    let file = File::open(path).map_err(|source| match source.kind() {
        io::ErrorKind::NotFound => InspectError::FileNotFound(path.to_path_buf()),
        _ => InspectError::CannotRead {
            path: path.to_path_buf(),
            source,
        },
    })?;

    let file_size = file
        .metadata()
        .map_err(|source| InspectError::CannotRead {
            path: path.to_path_buf(),
            source,
        })?
        .len();

    let mmap =
        unsafe { MmapOptions::new().map(&file) }.map_err(|source| InspectError::CannotRead {
            path: path.to_path_buf(),
            source,
        })?;

    let (_, metadata) =
        SafeTensors::read_metadata(&mmap).map_err(|source| InspectError::InvalidSafetensors {
            path: path.to_path_buf(),
            source,
        })?;

    build_report(path, file_size, &metadata)
}

fn build_report(path: &Path, file_size: u64, metadata: &Metadata) -> Result<Report, InspectError> {
    let mut total_parameters = 0_u128;
    let mut total_tensor_bytes = 0_u128;
    let mut tensors = Vec::new();
    let mut dtype_breakdown = BTreeMap::<String, u128>::new();

    for name in metadata.offset_keys() {
        let info = metadata
            .info(&name)
            .ok_or_else(|| InspectError::MissingTensorInfo {
                tensor: name.clone(),
            })?;

        let numel = numel(&info.shape, &name)?;
        let bytes = (info.data_offsets.1 - info.data_offsets.0) as u128;
        let dtype = info.dtype.to_string();

        total_parameters =
            total_parameters
                .checked_add(numel)
                .ok_or_else(|| InspectError::Overflow {
                    tensor: name.clone(),
                })?;
        total_tensor_bytes =
            total_tensor_bytes
                .checked_add(bytes)
                .ok_or_else(|| InspectError::Overflow {
                    tensor: name.clone(),
                })?;
        *dtype_breakdown.entry(dtype.clone()).or_insert(0) += bytes;

        tensors.push(TensorSummary {
            name,
            dtype,
            shape: info.shape.clone(),
            numel,
            bytes,
        });
    }

    let mut metadata_entries = metadata
        .metadata()
        .as_ref()
        .map(|entries| {
            entries
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    metadata_entries.sort_by(|left, right| left.0.cmp(&right.0));

    Ok(Report {
        file_path: path.to_path_buf(),
        file_size,
        tensor_count: tensors.len(),
        total_parameters,
        total_tensor_bytes,
        metadata: metadata_entries,
        dtype_breakdown: dtype_breakdown.into_iter().collect(),
        tensors,
    })
}

fn numel(shape: &[usize], tensor_name: &str) -> Result<u128, InspectError> {
    shape.iter().try_fold(1_u128, |acc, &dim| {
        acc.checked_mul(dim as u128)
            .ok_or_else(|| InspectError::Overflow {
                tensor: tensor_name.to_owned(),
            })
    })
}

fn format_shape(shape: &[usize]) -> String {
    let dims = shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{dims}]")
}

fn print_report(report: &Report) {
    println!("File: {}", report.file_path.display());
    println!("Size: {} bytes", report.file_size);
    println!("Tensors: {}", report.tensor_count);
    println!("Parameters: {}", report.total_parameters);
    println!("Tensor-Bytes: {}", report.total_tensor_bytes);
    println!();

    println!("Metadata:");
    if report.metadata.is_empty() {
        println!("  (none)");
    } else {
        for (key, value) in &report.metadata {
            println!("  {key} = {value}");
        }
    }
    println!();

    println!("DType Breakdown:");
    if report.dtype_breakdown.is_empty() {
        println!("  (none)");
    } else {
        for (dtype, bytes) in &report.dtype_breakdown {
            println!("  {dtype}: {bytes} bytes");
        }
    }
    println!();

    println!("Tensors:");
    if report.tensors.is_empty() {
        println!("  (none)");
        return;
    }

    for (index, tensor) in report.tensors.iter().enumerate() {
        if index > 0 {
            println!();
        }

        println!("  {}", tensor.name);
        println!("    dtype: {}", tensor.dtype);
        println!("    shape: {}", format_shape(&tensor.shape));
        println!("    numel: {}", tensor.numel);
        println!("    bytes: {}", tensor.bytes);
    }
}
