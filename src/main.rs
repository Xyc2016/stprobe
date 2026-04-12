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
            print!("{}", stprobe::render_report(&report));
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}
