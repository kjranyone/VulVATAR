//! End-to-end check of the dependency resolver against the real
//! network, using the real manifest entry the app would use.
//!
//! `#[ignore]`d: it transfers ~32 MB and depends on Hugging Face being
//! reachable, so it must not run in a default `cargo test`. Run it
//! after touching `provisioning::job` — the unit tests cover the pure
//! logic, but only this exercises the parts that can silently be wrong
//! in production: the URL, curl's argument shape, the `.part` rename,
//! and progress actually arriving.
//!
//! ```powershell
//! cargo test --test provisioning_fetch -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use vulvatar_lib::provisioning::{self, ProvisionJob, StepOutcome};

/// Scratch root for a run. Under `diagnostics/`, which is gitignored —
/// test output never goes near `validation_images/`.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from("diagnostics")
        .join("provisioning_tests")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).expect("create scratch root");
    dir
}

fn dep(id: &str) -> &'static provisioning::Dependency {
    provisioning::manifest()
        .iter()
        .find(|d| d.id == id)
        .unwrap_or_else(|| panic!("no manifest entry `{id}`"))
}

/// Drive a job to completion, returning its outcomes and whether any
/// progress message carried a fraction.
fn run(job: &mut ProvisionJob, timeout: Duration) -> bool {
    let start = Instant::now();
    let mut saw_fraction = false;
    while !job.poll() {
        if let Some(p) = job.current {
            if p.fraction.is_some() {
                saw_fraction = true;
            }
        }
        assert!(
            start.elapsed() < timeout,
            "job did not finish in {timeout:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    job.join();
    saw_fraction
}

#[test]
#[ignore = "network: downloads ~32 MB from Hugging Face"]
fn fetches_the_face_landmark_model_for_real() {
    let root = scratch("face_landmarks");
    let dep = dep("face_landmarks");
    assert!(
        !dep.is_satisfied(&root),
        "scratch root must start without the model"
    );

    let mut job = ProvisionJob::spawn(root.clone(), vec![dep]);
    let saw_fraction = run(&mut job, Duration::from_secs(600));

    assert_eq!(job.outcomes.len(), 1);
    match &job.outcomes[0].1 {
        StepOutcome::Done => {}
        other => panic!("expected Done, got {other:?}"),
    }

    // The probe the app uses at startup must now pass.
    assert!(
        dep.is_satisfied(&root),
        "the model did not land where the probe looks"
    );

    // A truncated transfer must never be renamed into place.
    let landed = root.join("models/rtm_face_fp16.tflite");
    let size = std::fs::metadata(&landed).expect("stat").len();
    assert!(
        size > 30_000_000,
        "expected the full ~32 MB model, got {size} bytes"
    );
    assert!(
        !root.join("models/rtm_face.part").exists(),
        "the temporary transfer file must be gone"
    );
    assert!(saw_fraction, "a sized download should report progress");

    println!("fetched {} bytes to {}", size, landed.display());
    let _ = std::fs::remove_dir_all(&root);
}

/// Cancelling must leave nothing behind that a later scan would mistake
/// for a finished model.
#[test]
#[ignore = "network: starts a download, then aborts it"]
fn cancelling_leaves_no_partial_model() {
    let root = scratch("cancel");
    let dep = dep("face_landmarks");

    let mut job = ProvisionJob::spawn(root.clone(), vec![dep]);
    // Let curl open the connection and start writing before aborting.
    std::thread::sleep(Duration::from_millis(400));
    job.cancel();
    run(&mut job, Duration::from_secs(60));

    assert!(
        !dep.is_satisfied(&root),
        "a cancelled fetch must not satisfy the probe"
    );
    assert!(
        !root.join("models/rtm_face_fp16.tflite").exists(),
        "no model file may be left behind"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// The Python half of the resolver: build a venv from scratch and pip
/// install into it. Covers what the unit tests cannot — that
/// `python -m venv` is reachable, that the interpreter lands where
/// `venv_python` expects on this machine, and that the probe the
/// startup scan uses passes afterwards.
#[test]
#[ignore = "network + python: creates a venv and pip installs ~60 MB"]
fn provisions_the_face_sidecar_venv_from_scratch() {
    let root = scratch("face_venv");
    let dep = dep("face_sidecar_venv");
    assert!(!dep.is_satisfied(&root), "scratch root must start empty");

    let mut job = ProvisionJob::spawn(root.clone(), vec![dep]);
    run(&mut job, Duration::from_secs(900));

    match &job.outcomes[0].1 {
        StepOutcome::Done => {}
        other => panic!("expected Done, got {other:?}"),
    }
    assert!(
        dep.is_satisfied(&root),
        "the venv exists but cannot import ai_edge_litert / numpy"
    );
    println!("provisioned {}", root.join("tools/face98-venv").display());
    let _ = std::fs::remove_dir_all(&root);
}

/// The blocking dependency: export the detector from the repo's
/// ultralytics checkpoint. Reuses an existing ultralytics venv when one
/// is present, so this does not necessarily pay the torch download.
#[test]
#[ignore = "python: runs an ultralytics ONNX export (may download torch)"]
fn exports_the_body_detector() {
    let root = scratch("detector");
    // The export reads the checkpoint from the job root, so give the
    // scratch root its own copy rather than polluting the repo.
    std::fs::copy("yolo26n-pose.pt", root.join("yolo26n-pose.pt"))
        .expect("repo checkpoint must exist");

    let dep = dep("yolo26_pose");
    assert!(!dep.is_satisfied(&root));

    let mut job = ProvisionJob::spawn(root.clone(), vec![dep]);
    run(&mut job, Duration::from_secs(1800));

    match &job.outcomes[0].1 {
        StepOutcome::Done => {}
        other => panic!("expected Done, got {other:?}"),
    }
    assert!(
        dep.is_satisfied(&root),
        "export finished but the detector probe still fails"
    );
    let onnx = root.join("models/yolo26n-pose_480.onnx");
    let size = std::fs::metadata(&onnx).expect("stat").len();
    assert!(size > 1_000_000, "suspiciously small export: {size} bytes");
    // The `_480` suffix is the input-size contract the loader reads.
    assert!(
        !root.join("yolo26n-pose.onnx").exists(),
        "the raw ultralytics output must be moved, not copied"
    );
    println!("exported {} bytes to {}", size, onnx.display());
    let _ = std::fs::remove_dir_all(&root);
}

/// The hand landmark model: a zip bundle, and the one hand file the
/// chain cannot start without.
///
/// This asserts the CONTRACT, not just the download, because the
/// bundle's own metadata lies about it — `detail.json` says
/// `input_shape: [192, 256]` while the graph is square 256. The check
/// that matters is whether the app's own loader accepts it, so the model
/// is loaded through `HandBackend` exactly as the tracking worker would.
#[test]
#[ignore = "network: downloads a ~51 MB mmdeploy bundle"]
fn fetches_and_loads_the_hand_landmark_model() {
    let root = scratch("hand");
    let dep = dep("hand_rtmpose");
    assert!(!dep.is_satisfied(&root));

    let mut job = ProvisionJob::spawn(root.clone(), vec![dep]);
    run(&mut job, Duration::from_secs(900));

    match &job.outcomes[0].1 {
        StepOutcome::Done => {}
        other => panic!("expected Done, got {other:?}"),
    }
    assert!(
        dep.is_satisfied(&root),
        "the probe still fails after fetching"
    );

    let onnx = root.join("models/rtmpose-m-hand_256.onnx");
    let size = std::fs::metadata(&onnx).expect("stat").len();
    assert!(size > 50_000_000, "expected the ~55 MB export, got {size}");
    // The scratch directory and the downloaded archive must be gone.
    assert!(
        !root.join("models/rtmpose-m-hand_256.unzip").exists(),
        "the extraction scratch directory was left behind"
    );

    // The authoritative check: the app's own loader, same models dir
    // layout, same onnxruntime. `Ok(_)` means the SimCC contract in
    // `fusion::hands` matched the graph.
    let backend = vulvatar_lib::tracking::fusion::hands::HandBackend::try_from_models_dir(
        root.join("models"),
    );
    match backend {
        Ok(b) => println!("loaded via {} ({} bytes)", b.label(), size),
        Err(e) => panic!("HandBackend rejected the fetched model: {e}"),
    }
    let _ = std::fs::remove_dir_all(&root);
}
