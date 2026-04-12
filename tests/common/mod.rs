use std::collections::HashMap;
use std::fs;
use std::path::Path;

use safetensors::tensor::{Dtype, TensorView};

#[allow(dead_code)]
pub fn write_sample_file(path: &Path) {
    fs::write(path, sample_safetensors_bytes()).expect("write test safetensors fixture");
}

pub fn sample_safetensors_bytes() -> Vec<u8> {
    let weight = [1.0_f32.to_le_bytes(), 2.0_f32.to_le_bytes()].concat();
    let ids = [1_i64.to_le_bytes(), 2_i64.to_le_bytes()].concat();

    let weight_view = TensorView::new(Dtype::F32, vec![2], &weight).expect("valid weight tensor");
    let ids_view = TensorView::new(Dtype::I64, vec![2], &ids).expect("valid ids tensor");

    safetensors::serialize(
        vec![
            ("embedding.ids", ids_view),
            ("embedding.weight", weight_view),
        ],
        Some(HashMap::from([("format".to_owned(), "pt".to_owned())])),
    )
    .expect("serialize test safetensors fixture")
}

#[allow(dead_code)]
pub fn split_header(bytes: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let prefix = bytes[..8].to_vec();
    let header_len = u64::from_le_bytes(prefix.as_slice().try_into().expect("8-byte prefix"));
    let header_end = 8 + header_len as usize;
    let header = bytes[8..header_end].to_vec();
    (prefix, header)
}
