use std::io::{self, Write};
use std::process::ExitCode;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "stprobe",
    version,
    about = "Inspect basic metadata from a .safetensors file",
    long_about = None
)]
struct Cli {
    /// Path or HTTP(S) URL to a .safetensors file
    file: String,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match stprobe::inspect_input(&cli.file) {
        Ok(report) => {
            let rendered = stprobe::render_report(&report);
            let mut stdout = io::stdout().lock();

            match stdout.write_all(rendered.as_bytes()) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) if error.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("Error: failed to write output ({error})");
                    ExitCode::FAILURE
                }
            }
        }
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}
