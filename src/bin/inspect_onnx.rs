//! Print the input / output schema of an ONNX model.
//!
//! Usage: `cargo run --release --bin inspect_onnx --features inference -- <model.onnx>`

#[cfg(feature = "inference")]
fn main() -> Result<(), String> {
    use ort::session::Session;

    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| "usage: inspect_onnx <model.onnx>".to_string())?;
    let session = Session::builder()
        .map_err(|e| e.to_string())?
        .commit_from_file(&path)
        .map_err(|e| e.to_string())?;

    println!("=== {} ===", path);
    println!("inputs:");
    for input in session.inputs() {
        println!("  {} : {:?}", input.name(), input.dtype());
    }
    println!("outputs:");
    for output in session.outputs() {
        println!("  {} : {:?}", output.name(), output.dtype());
    }
    Ok(())
}

#[cfg(not(feature = "inference"))]
fn main() -> Result<(), String> {
    Err("inspect_onnx requires --features inference".into())
}
