use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};

use memmap2::MmapOptions;
use reqwest::blocking::{Client, Response};
use reqwest::header::{ACCEPT_ENCODING, AUTHORIZATION, CONTENT_RANGE, RANGE};
use reqwest::{StatusCode, Url};
use safetensors::{tensor::Metadata, SafeTensorError, SafeTensors};

const HEADER_PREFIX_LEN: u64 = 8;
const MAX_HEADER_SIZE: usize = 100_000_000;

#[derive(Debug)]
pub enum InspectError {
    FileNotFound(PathBuf),
    CannotRead {
        path: PathBuf,
        source: io::Error,
    },
    InvalidSafetensors {
        path: String,
        source: SafeTensorError,
    },
    Overflow {
        tensor: String,
    },
    MissingTensorInfo {
        tensor: String,
    },
    UnsupportedUrlScheme(String),
    HttpClient(reqwest::Error),
    HttpRequest {
        url: String,
        source: reqwest::Error,
    },
    RangeUnsupported(String),
    InvalidRemoteResponse {
        url: String,
        reason: String,
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
                write!(f, "invalid safetensors file: {path} ({source})")
            }
            Self::Overflow { tensor } => {
                write!(f, "tensor is too large to summarize safely: {tensor}")
            }
            Self::MissingTensorInfo { tensor } => {
                write!(f, "missing tensor metadata for: {tensor}")
            }
            Self::UnsupportedUrlScheme(scheme) => {
                write!(f, "unsupported URL scheme: {scheme}")
            }
            Self::HttpClient(source) => write!(f, "failed to initialize HTTP client ({source})"),
            Self::HttpRequest { url, source } => {
                write!(f, "failed to fetch remote file: {url} ({source})")
            }
            Self::RangeUnsupported(url) => {
                write!(
                    f,
                    "remote server does not support byte range requests: {url}"
                )
            }
            Self::InvalidRemoteResponse { url, reason } => {
                write!(f, "invalid remote response for {url} ({reason})")
            }
        }
    }
}

impl std::error::Error for InspectError {}

#[derive(Debug)]
pub struct Report {
    file_path: String,
    file_size: u64,
    tensor_count: usize,
    total_parameters: u128,
    total_tensor_bytes: u128,
    metadata: Vec<(String, String)>,
    dtype_breakdown: Vec<(String, u128)>,
    tensors: Vec<TensorSummary>,
}

#[derive(Debug)]
pub struct TensorSummary {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    numel: u128,
    bytes: u128,
}

pub fn inspect_input(input: &str) -> Result<Report, InspectError> {
    match classify_input(input)? {
        Input::LocalPath(path) => inspect_local_file(input, path),
        Input::HttpUrl(url) => inspect_remote_file(input, &url),
    }
}

pub fn render_report(report: &Report) -> String {
    let mut output = String::new();

    writeln!(&mut output, "File: {}", report.file_path).unwrap();
    writeln!(&mut output, "Size: {} bytes", report.file_size).unwrap();
    writeln!(&mut output, "Tensors: {}", report.tensor_count).unwrap();
    writeln!(&mut output, "Parameters: {}", report.total_parameters).unwrap();
    writeln!(&mut output, "Tensor-Bytes: {}", report.total_tensor_bytes).unwrap();
    writeln!(&mut output).unwrap();

    writeln!(&mut output, "Metadata:").unwrap();
    if report.metadata.is_empty() {
        writeln!(&mut output, "  (none)").unwrap();
    } else {
        for (key, value) in &report.metadata {
            writeln!(&mut output, "  {key} = {value}").unwrap();
        }
    }
    writeln!(&mut output).unwrap();

    writeln!(&mut output, "DType Breakdown:").unwrap();
    if report.dtype_breakdown.is_empty() {
        writeln!(&mut output, "  (none)").unwrap();
    } else {
        for (dtype, bytes) in &report.dtype_breakdown {
            writeln!(&mut output, "  {dtype}: {bytes} bytes").unwrap();
        }
    }
    writeln!(&mut output).unwrap();

    writeln!(&mut output, "Tensors:").unwrap();
    if report.tensors.is_empty() {
        writeln!(&mut output, "  (none)").unwrap();
        return output;
    }

    for (index, tensor) in report.tensors.iter().enumerate() {
        if index > 0 {
            writeln!(&mut output).unwrap();
        }

        writeln!(&mut output, "  {}", tensor.name).unwrap();
        writeln!(&mut output, "    dtype: {}", tensor.dtype).unwrap();
        writeln!(&mut output, "    shape: {}", format_shape(&tensor.shape)).unwrap();
        writeln!(&mut output, "    numel: {}", tensor.numel).unwrap();
        writeln!(&mut output, "    bytes: {}", tensor.bytes).unwrap();
    }

    output
}

#[derive(Debug)]
enum Input<'a> {
    LocalPath(&'a Path),
    HttpUrl(Url),
}

fn classify_input(input: &str) -> Result<Input<'_>, InspectError> {
    if !input.contains("://") {
        return Ok(Input::LocalPath(Path::new(input)));
    }

    match Url::parse(input) {
        Ok(url) => match url.scheme() {
            "http" | "https" => Ok(Input::HttpUrl(url)),
            scheme => Err(InspectError::UnsupportedUrlScheme(scheme.to_owned())),
        },
        Err(_) => Err(InspectError::InvalidRemoteResponse {
            url: input.to_owned(),
            reason: "malformed URL".to_owned(),
        }),
    }
}

fn inspect_local_file(input: &str, path: &Path) -> Result<Report, InspectError> {
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
            path: input.to_owned(),
            source,
        })?;

    build_report(input, file_size, &metadata)
}

fn inspect_remote_file(input: &str, url: &Url) -> Result<Report, InspectError> {
    let client = build_http_client()?;
    let (file_size, header_len) = fetch_header_prefix(&client, url)?;
    if header_len > MAX_HEADER_SIZE {
        return Err(InspectError::InvalidSafetensors {
            path: input.to_owned(),
            source: SafeTensorError::HeaderTooLarge,
        });
    }

    let header_bytes = fetch_header_bytes(&client, url, header_len)?;
    let metadata: Metadata = serde_json::from_slice(&header_bytes).map_err(|source| {
        InspectError::InvalidSafetensors {
            path: input.to_owned(),
            source: SafeTensorError::InvalidHeaderDeserialization(source),
        }
    })?;

    let expected_size = HEADER_PREFIX_LEN
        .checked_add(header_len as u64)
        .and_then(|value| value.checked_add(metadata.data_len() as u64))
        .ok_or_else(|| InspectError::InvalidSafetensors {
            path: input.to_owned(),
            source: SafeTensorError::ValidationOverflow,
        })?;

    if expected_size != file_size {
        return Err(InspectError::InvalidSafetensors {
            path: input.to_owned(),
            source: SafeTensorError::MetadataIncompleteBuffer,
        });
    }

    build_report(input, file_size, &metadata)
}

fn build_http_client() -> Result<Client, InspectError> {
    Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .user_agent(format!("stprobe/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(InspectError::HttpClient)
}

fn fetch_header_prefix(client: &Client, url: &Url) -> Result<(u64, usize), InspectError> {
    let response = ranged_get(client, url, 0, HEADER_PREFIX_LEN - 1)?;
    if response.status() != StatusCode::PARTIAL_CONTENT {
        return Err(InspectError::RangeUnsupported(url.to_string()));
    }

    let file_size = parse_total_size(&response, url)?;
    let bytes = read_response_bytes(response, HEADER_PREFIX_LEN as usize, url)?;
    let header_len = u64::from_le_bytes(
        bytes[..HEADER_PREFIX_LEN as usize]
            .try_into()
            .expect("slice length is checked by read_response_bytes"),
    );

    let header_len = header_len
        .try_into()
        .map_err(|_| InspectError::InvalidSafetensors {
            path: url.to_string(),
            source: SafeTensorError::HeaderTooLarge,
        })?;

    Ok((file_size, header_len))
}

fn fetch_header_bytes(
    client: &Client,
    url: &Url,
    header_len: usize,
) -> Result<Vec<u8>, InspectError> {
    let start = HEADER_PREFIX_LEN;
    let end = start
        .checked_add(header_len as u64)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| InspectError::InvalidRemoteResponse {
            url: url.to_string(),
            reason: "invalid header range".to_owned(),
        })?;

    let response = ranged_get(client, url, start, end)?;
    if response.status() != StatusCode::PARTIAL_CONTENT {
        return Err(InspectError::RangeUnsupported(url.to_string()));
    }

    read_response_bytes(response, header_len, url)
}

fn ranged_get(client: &Client, url: &Url, start: u64, end: u64) -> Result<Response, InspectError> {
    let mut request = client
        .get(url.clone())
        .header(RANGE, format!("bytes={start}-{end}"))
        .header(ACCEPT_ENCODING, "identity");

    if is_hugging_face_url(url) {
        if let Ok(token) = std::env::var("HF_TOKEN") {
            if !token.is_empty() {
                request = request.header(AUTHORIZATION, format!("Bearer {token}"));
            }
        }
    }

    request.send().map_err(|source| InspectError::HttpRequest {
        url: url.to_string(),
        source,
    })
}

fn is_hugging_face_url(url: &Url) -> bool {
    matches!(
        url.host_str(),
        Some("huggingface.co") | Some("www.huggingface.co")
    )
}

fn parse_total_size(response: &Response, url: &Url) -> Result<u64, InspectError> {
    let content_range = response
        .headers()
        .get(CONTENT_RANGE)
        .ok_or_else(|| InspectError::InvalidRemoteResponse {
            url: url.to_string(),
            reason: "missing Content-Range header".to_owned(),
        })?
        .to_str()
        .map_err(|_| InspectError::InvalidRemoteResponse {
            url: url.to_string(),
            reason: "invalid Content-Range header".to_owned(),
        })?;

    parse_total_size_from_content_range(content_range).map_err(|reason| {
        InspectError::InvalidRemoteResponse {
            url: url.to_string(),
            reason,
        }
    })
}

fn parse_total_size_from_content_range(content_range: &str) -> Result<u64, String> {
    let total = content_range
        .rsplit('/')
        .next()
        .ok_or_else(|| "malformed Content-Range header".to_owned())?;

    total
        .parse::<u64>()
        .map_err(|_| "invalid total size in Content-Range header".to_owned())
}

fn read_response_bytes(
    mut response: Response,
    expected_len: usize,
    url: &Url,
) -> Result<Vec<u8>, InspectError> {
    let mut bytes = Vec::with_capacity(expected_len);
    response
        .read_to_end(&mut bytes)
        .map_err(|source| InspectError::InvalidRemoteResponse {
            url: url.to_string(),
            reason: format!("failed reading response body ({source})"),
        })?;

    if bytes.len() != expected_len {
        return Err(InspectError::InvalidRemoteResponse {
            url: url.to_string(),
            reason: format!("expected {expected_len} bytes, got {}", bytes.len()),
        });
    }

    Ok(bytes)
}

fn build_report(input: &str, file_size: u64, metadata: &Metadata) -> Result<Report, InspectError> {
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
        file_path: input.to_owned(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_local_paths() {
        match classify_input("model.safetensors").unwrap() {
            Input::LocalPath(path) => assert_eq!(path, Path::new("model.safetensors")),
            Input::HttpUrl(_) => panic!("expected local path"),
        }
    }

    #[test]
    fn classifies_https_urls() {
        match classify_input("https://example.com/model.safetensors").unwrap() {
            Input::HttpUrl(url) => {
                assert_eq!(url.scheme(), "https");
                assert_eq!(url.host_str(), Some("example.com"));
            }
            Input::LocalPath(_) => panic!("expected URL"),
        }
    }

    #[test]
    fn rejects_unsupported_schemes() {
        match classify_input("hf://org/repo/file.safetensors").unwrap_err() {
            InspectError::UnsupportedUrlScheme(scheme) => assert_eq!(scheme, "hf"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn classifies_windows_drive_paths_as_local() {
        match classify_input(r"C:\models\sample.safetensors").unwrap() {
            Input::LocalPath(path) => {
                assert_eq!(path, Path::new(r"C:\models\sample.safetensors"));
            }
            Input::HttpUrl(_) => panic!("expected local path"),
        }
    }

    #[test]
    fn parses_total_size_from_content_range() {
        let total = parse_total_size_from_content_range("bytes 0-7/17246524772").unwrap();
        assert_eq!(total, 17_246_524_772);
    }

    #[test]
    fn rejects_malformed_content_range() {
        let error = parse_total_size_from_content_range("bytes 0-7/*").unwrap_err();
        assert_eq!(error, "invalid total size in Content-Range header");
    }

    #[test]
    fn computes_numel() {
        assert_eq!(numel(&[2, 3, 4], "tensor").unwrap(), 24);
    }

    #[test]
    fn formats_shapes() {
        assert_eq!(format_shape(&[2, 3, 4]), "[2, 3, 4]");
        assert_eq!(format_shape(&[]), "[]");
    }
}
