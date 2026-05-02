//! ONNX Runtime session construction with the project-wide
//! "DirectML, fall back to CPU" policy. Used by the RTMW3D session in
//! `mod.rs` and re-used as `pub(in crate::tracking)` by the sibling
//! face-mesh / blendshape / yolox modules so they share the same EP
//! selection logic.

use log::{info, warn};
use ort::session::Session;

use super::InferenceBackend;

#[cfg(feature = "inference-gpu")]
pub(in crate::tracking) fn build_session(
    model_path: &str,
    intra_threads: usize,
    label: &str,
) -> Result<(Session, InferenceBackend), String> {
    match try_build_directml_session(model_path, intra_threads) {
        Ok(session) => {
            info!(
                "{} session created via DirectML execution provider ({})",
                label, model_path
            );
            Ok((session, InferenceBackend::DirectMl))
        }
        Err(e) => {
            warn!(
                "{} DirectML EP registration failed for '{}': {}. Falling back to CPU.",
                label, model_path, e
            );
            let session = build_cpu_session(model_path, intra_threads, label)?;
            Ok((
                session,
                InferenceBackend::CpuFromDirectMlFailure { reason: e },
            ))
        }
    }
}

#[cfg(not(feature = "inference-gpu"))]
pub(in crate::tracking) fn build_session(
    model_path: &str,
    intra_threads: usize,
    label: &str,
) -> Result<(Session, InferenceBackend), String> {
    let session = build_cpu_session(model_path, intra_threads, label)?;
    Ok((session, InferenceBackend::Cpu))
}

fn build_cpu_session(
    model_path: &str,
    intra_threads: usize,
    label: &str,
) -> Result<Session, String> {
    let session = Session::builder()
        .map_err(|e| format!("ORT build error ({}): {}", label, e))?
        .with_intra_threads(intra_threads)
        .map_err(|e| format!("ORT threads error ({}): {}", label, e))?
        .commit_from_file(model_path)
        .map_err(|e| format!("ORT load error ({}): {}", label, e))?;
    info!("{} session created on CPU EP ({})", label, model_path);
    Ok(session)
}

/// Force a session onto the CPU EP regardless of `inference-gpu`.
/// Used for small models (FaceMesh ~4.8 MB, Blendshape ~1.8 MB) that
/// otherwise fight DirectML for GPU command queue when a heavier
/// model (DAv2 / RTMW3D / MoGe) is running concurrently. CPU is fast
/// enough for these on modern hardware (single-digit ms) and fully
/// removes the contention.
pub(in crate::tracking) fn build_session_cpu_only(
    model_path: &str,
    intra_threads: usize,
    label: &str,
) -> Result<(Session, InferenceBackend), String> {
    let session = build_cpu_session(model_path, intra_threads, label)?;
    Ok((session, InferenceBackend::Cpu))
}

#[cfg(feature = "inference-gpu")]
fn try_build_directml_session(model_path: &str, intra_threads: usize) -> Result<Session, String> {
    let builder = Session::builder().map_err(|e| format!("builder: {}", e))?;
    let builder = builder
        .with_execution_providers([ort::ep::DirectML::default().build()])
        .map_err(|e| format!("with_execution_providers: {}", e))?;
    let mut builder = builder
        .with_intra_threads(intra_threads)
        .map_err(|e| format!("with_intra_threads: {}", e))?;
    builder
        .commit_from_file(model_path)
        .map_err(|e| format!("commit_from_file: {}", e))
}
