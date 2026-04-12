mod common;

use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[test]
fn inspects_remote_files_via_range_requests() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let server = runtime.block_on(MockServer::start());
    let bytes = common::sample_safetensors_bytes();
    let total_size = bytes.len();
    let (prefix, header_bytes) = common::split_header(&bytes);
    let header_end = 8 + header_bytes.len() - 1;
    let header_range = format!("bytes=8-{header_end}");
    let base_url = server.uri();
    let resolve_path = "/resolve/main/sample.safetensors";
    let cdn_path = "/cdn/sample.safetensors";

    runtime.block_on(async {
        Mock::given(method("GET"))
            .and(path(resolve_path))
            .respond_with(
                ResponseTemplate::new(302)
                    .append_header("Location", format!("{base_url}{cdn_path}")),
            )
            .expect(2)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path(cdn_path))
            .and(header("range", "bytes=0-7"))
            .respond_with(
                ResponseTemplate::new(206)
                    .append_header("Content-Range", format!("bytes 0-7/{total_size}"))
                    .set_body_bytes(prefix),
            )
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path(cdn_path))
            .and(header("range", header_range.as_str()))
            .respond_with(
                ResponseTemplate::new(206)
                    .append_header(
                        "Content-Range",
                        format!("bytes 8-{header_end}/{total_size}"),
                    )
                    .set_body_bytes(header_bytes),
            )
            .expect(1)
            .mount(&server)
            .await;
    });

    let url = format!("{base_url}{resolve_path}");
    let report = stprobe::inspect_input(&url).expect("inspect remote safetensors");
    let output = stprobe::render_report(&report);

    assert!(output.contains(&format!("File: {url}")));
    assert!(output.contains("Tensors: 2"));
    assert!(output.contains("Parameters: 4"));
    assert!(output.contains("Tensor-Bytes: 24"));
    assert!(output.contains("  format = pt"));
    assert!(output.contains("  embedding.ids"));
    assert!(output.contains("  embedding.weight"));
}

#[test]
fn reports_servers_without_range_support() {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let server = runtime.block_on(MockServer::start());

    runtime.block_on(async {
        Mock::given(method("GET"))
            .and(path("/sample.safetensors"))
            .and(header("range", "bytes=0-7"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![0; 8]))
            .expect(1)
            .mount(&server)
            .await;
    });

    let url = format!("{}/sample.safetensors", server.uri());
    let error = stprobe::inspect_input(&url).expect_err("range support error");

    assert_eq!(
        error.to_string(),
        format!("remote server does not support byte range requests: {url}")
    );
}
