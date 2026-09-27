//! The worker that actually resolves dependencies.
//!
//! Runs off the UI thread and streams [`ProvisionMessage`]s back, the
//! same shape the avatar loader uses (`gui::avatar_load`): the GUI holds
//! an `Option<ProvisionJob>` and drains it once per frame.
//!
//! External tools rather than crates, on purpose:
//!
//! * **`curl.exe`** for downloads — shipped with Windows 10 1803+, and
//!   already what `dev.ps1` uses, so the two provisioning paths behave
//!   identically (same redirects, same TLS stack). Pulling in a Rust
//!   HTTP client plus a TLS stack for three URLs is not worth the
//!   build-time and audit surface.
//! * **`python` / `py -3`** for venvs and the ultralytics export. There
//!   is no way around an interpreter here: the export is a torch
//!   graph trace, and the face sidecar *is* a Python process.
//!
//! Every child is spawned with `CREATE_NO_WINDOW`. The app is a console
//! subsystem binary, so without it a `pip install` would flash console
//! windows over a live stream.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use log::{info, warn};

use super::{Dependency, Resolution, ZipItem, YOLO_EXPORT_VENV};

/// `CREATE_NO_WINDOW` — keep provisioning children off the user's
/// screen (see module docs).
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// How often a running child is polled for completion / cancellation.
const POLL_INTERVAL: Duration = Duration::from_millis(150);

/// Which part of a dependency's resolution is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Downloading,
    Extracting,
    CreatingVenv,
    InstallingPackages,
    Exporting,
}

impl Stage {
    /// i18n key for the status line.
    pub fn i18n_key(self) -> &'static str {
        match self {
            Stage::Downloading => "provisioning.stage.downloading",
            Stage::Extracting => "provisioning.stage.extracting",
            Stage::CreatingVenv => "provisioning.stage.creating_venv",
            Stage::InstallingPackages => "provisioning.stage.installing",
            Stage::Exporting => "provisioning.stage.exporting",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProvisionProgress {
    pub id: &'static str,
    /// 0-based position in the selected set.
    pub step_index: usize,
    pub step_total: usize,
    pub stage: Stage,
    /// `None` when the stage has no measurable size — `pip` resolves a
    /// dependency closure whose total is not known up front, and a
    /// spinner is more honest than a fabricated bar.
    pub fraction: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum StepOutcome {
    Done,
    Failed(String),
    Cancelled,
}

#[derive(Debug, Clone)]
pub enum ProvisionMessage {
    Progress(ProvisionProgress),
    StepDone {
        id: &'static str,
        outcome: StepOutcome,
    },
    /// Every selected dependency has been attempted.
    Finished,
}

pub struct ProvisionJob {
    pub receiver: mpsc::Receiver<ProvisionMessage>,
    cancel: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    /// Latest progress seen by `poll`, so the dialog can redraw without
    /// the channel having produced anything this frame.
    pub current: Option<ProvisionProgress>,
    /// Per-dependency outcomes, in completion order.
    pub outcomes: Vec<(&'static str, StepOutcome)>,
    pub finished: bool,
}

impl ProvisionJob {
    /// Spawn a worker resolving `deps` under `root`, in order.
    ///
    /// `root` is made absolute first. The export step runs its child
    /// with `current_dir(root)`, so a relative root would be resolved a
    /// second time inside itself — and ultralytics, finding no
    /// checkpoint at the doubled path, silently downloads one from the
    /// network and reports success. The failure is then a model nobody
    /// asked for, in a directory nobody looks at.
    ///
    /// Joined against the cwd rather than `canonicalize`d: on Windows
    /// canonicalisation returns an extended-length path (the `\?\`
    /// prefix), and ultralytics' own path handling rejects it
    /// ("acceptable suffix is {'.pt'}, not .//" — measured).
    pub fn spawn(root: PathBuf, deps: Vec<&'static Dependency>) -> Self {
        let root = if root.is_absolute() {
            root
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(&root))
                .unwrap_or(root)
        };
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_for_worker = Arc::clone(&cancel);
        let worker = thread::spawn(move || {
            let total = deps.len();
            for (index, dep) in deps.iter().enumerate() {
                if cancel_for_worker.load(Ordering::Relaxed) {
                    let _ = tx.send(ProvisionMessage::StepDone {
                        id: dep.id,
                        outcome: StepOutcome::Cancelled,
                    });
                    continue;
                }
                let ctx = StepCtx {
                    root: &root,
                    tx: &tx,
                    cancel: &cancel_for_worker,
                    id: dep.id,
                    step_index: index,
                    step_total: total,
                };
                let outcome = match resolve(dep, &ctx) {
                    Ok(()) => {
                        info!("provisioning: {} resolved", dep.id);
                        StepOutcome::Done
                    }
                    Err(e) if cancel_for_worker.load(Ordering::Relaxed) => {
                        info!("provisioning: {} cancelled ({e})", dep.id);
                        StepOutcome::Cancelled
                    }
                    Err(e) => {
                        warn!("provisioning: {} failed: {e}", dep.id);
                        StepOutcome::Failed(e)
                    }
                };
                let _ = tx.send(ProvisionMessage::StepDone {
                    id: dep.id,
                    outcome,
                });
            }
            let _ = tx.send(ProvisionMessage::Finished);
        });
        Self {
            receiver: rx,
            cancel,
            worker: Some(worker),
            current: None,
            outcomes: Vec::new(),
            finished: false,
        }
    }

    /// Drain pending messages. Returns `true` once the job has finished
    /// (the caller then reads [`Self::outcomes`] and drops the job).
    pub fn poll(&mut self) -> bool {
        while let Ok(msg) = self.receiver.try_recv() {
            match msg {
                ProvisionMessage::Progress(p) => self.current = Some(p),
                ProvisionMessage::StepDone { id, outcome } => {
                    self.outcomes.push((id, outcome));
                    self.current = None;
                }
                ProvisionMessage::Finished => self.finished = true,
            }
        }
        self.finished
    }

    /// Ask the worker to stop. The in-flight child is killed, so a
    /// half-written download is discarded rather than left where the
    /// next scan would mistake it for a complete model.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Join the worker thread. Called once `poll` reports finished.
    pub fn join(&mut self) {
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ProvisionJob {
    fn drop(&mut self) {
        // A dropped job must not leave a detached child writing into
        // `models/` behind the app's back.
        self.cancel();
        self.join();
    }
}

/// Everything a resolution step needs to report and to be interrupted.
struct StepCtx<'a> {
    root: &'a Path,
    tx: &'a mpsc::Sender<ProvisionMessage>,
    cancel: &'a AtomicBool,
    id: &'static str,
    step_index: usize,
    step_total: usize,
}

impl StepCtx<'_> {
    fn progress(&self, stage: Stage, fraction: Option<f32>) {
        let _ = self.tx.send(ProvisionMessage::Progress(ProvisionProgress {
            id: self.id,
            step_index: self.step_index,
            step_total: self.step_total,
            stage,
            fraction,
        }));
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

fn resolve(dep: &Dependency, ctx: &StepCtx) -> Result<(), String> {
    match dep.resolution {
        Resolution::Download { url, dest } => download(url, dest, dep.approx_bytes, ctx),
        Resolution::DownloadZip { items } => {
            // Split the size estimate across the archives so the bar
            // advances per item instead of jumping to "done" on the first.
            let each = dep.approx_bytes / items.len().max(1) as u64;
            for item in items {
                download_zip(item, each, ctx)?;
            }
            Ok(())
        }
        Resolution::PythonVenv { dir, packages } => ensure_venv(dir, packages, ctx).map(|_| ()),
        Resolution::UltralyticsExport {
            weights,
            dest,
            imgsz,
        } => ultralytics_export(weights, dest, imgsz, ctx),
        // `scan` never routes a manual entry here; a panic-free no-op
        // keeps the invariant cheap to hold.
        Resolution::Manual { doc } => Err(format!(
            "{} has no automated resolution (see {doc})",
            dep.id
        )),
    }
}

// ---------------------------------------------------------------- download

/// Fetch `url` to `dest`, reporting progress from the partial file's
/// size.
///
/// The transfer lands on `<dest>.part` and is renamed only after curl
/// exits successfully. A truncated model left at the real filename would
/// satisfy the next [`super::scan`] and then fail deep inside
/// onnxruntime with an opaque parse error.
fn download(url: &str, dest: &str, approx_bytes: u64, ctx: &StepCtx) -> Result<(), String> {
    let dest = ctx.root.join(dest);
    create_parent(&dest)?;
    let part = dest.with_extension("part");
    fetch(url, &part, approx_bytes, ctx)?;
    std::fs::rename(&part, &dest)
        .map_err(|e| format!("could not move the download into place: {e}"))?;
    Ok(())
}

/// Fetch a zip, keep the one entry named `keep`, write it to `dest`.
///
/// Extraction goes through `tar.exe` — bsdtar, shipped in System32 next
/// to `curl.exe`, and it reads zip. Resolved through `%SystemRoot%`
/// rather than PATH: a developer shell can easily put GNU tar first,
/// and GNU tar cannot read zip at all ("This does not look like a tar
/// archive", measured).
///
/// Nothing is written to `dest` until the wanted entry is in hand, so a
/// failure here cannot leave a file that the next [`super::scan`] would
/// accept.
fn download_zip(item: &ZipItem, approx_bytes: u64, ctx: &StepCtx) -> Result<(), String> {
    let (url, keep) = (item.url, item.keep);
    let dest = ctx.root.join(item.dest);
    // Already there from an earlier item or an earlier run: a multi-item
    // dependency is only partly missing most of the time.
    if dest.is_file() {
        return Ok(());
    }
    create_parent(&dest)?;

    // Scratch space beside the destination: same volume, so the final
    // move is a rename rather than a cross-device copy.
    let work = dest.with_extension("unzip");
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).map_err(|e| format!("create {}: {e}", work.display()))?;
    let archive = work.join("bundle.zip");

    let result = (|| -> Result<(), String> {
        fetch(url, &archive, approx_bytes, ctx)?;
        ctx.progress(Stage::Extracting, None);
        let mut cmd = Command::new(system_tar());
        cmd.arg("-xf").arg(&archive).arg("-C").arg(&work);
        let mut child = spawn(cmd).map_err(|e| {
            format!("tar.exe could not be started ({e}); Windows 10 1803+ ships it in System32")
        })?;
        let status = wait_with_progress(&mut child, ctx, Stage::Extracting, || None)?;
        if !status.ok {
            return Err(format!(
                "could not extract the archive: {}",
                status.detail("tar")
            ));
        }
        let found = find_file(&work, keep).ok_or_else(|| {
            format!("the archive did not contain `{keep}`; the upstream bundle layout changed")
        })?;
        std::fs::rename(&found, &dest)
            .map_err(|e| format!("could not move `{keep}` into place: {e}"))
    })();

    // Always clear the scratch directory: it holds a 50 MB archive plus
    // the expanded bundle, and on the success path the wanted file has
    // already been moved out of it.
    let _ = std::fs::remove_dir_all(&work);
    result
}

/// `curl.exe` transfer to `path`, with progress from the partial size.
///
/// Callers stage into a temporary path and move afterwards: a truncated
/// model left at the real filename would satisfy the next
/// [`super::scan`] and then fail deep inside onnxruntime with an opaque
/// parse error.
fn fetch(url: &str, path: &Path, approx_bytes: u64, ctx: &StepCtx) -> Result<(), String> {
    let _ = std::fs::remove_file(path);
    ctx.progress(Stage::Downloading, Some(0.0));
    let mut cmd = Command::new("curl.exe");
    cmd.arg("--fail")
        .arg("--location")
        .arg("--silent")
        .arg("--show-error")
        .arg(url)
        .arg("-o")
        .arg(path);
    let mut child = spawn(cmd).map_err(|e| {
        format!("curl.exe could not be started ({e}); Windows 10 1803+ ships it in System32")
    })?;

    let status = wait_with_progress(&mut child, ctx, Stage::Downloading, || {
        if approx_bytes == 0 {
            return None;
        }
        let got = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        Some((got as f32 / approx_bytes as f32).clamp(0.0, 1.0))
    })?;

    if !status.ok {
        let _ = std::fs::remove_file(path);
        return Err(format!("download failed: {}", status.detail(url)));
    }
    Ok(())
}

fn create_parent(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    Ok(())
}

/// bsdtar from System32 (see [`download_zip`] for why not bare PATH).
fn system_tar() -> PathBuf {
    if let Ok(root) = std::env::var("SystemRoot") {
        let p = PathBuf::from(root).join("System32").join("tar.exe");
        if p.is_file() {
            return p;
        }
    }
    PathBuf::from("tar.exe")
}

/// First file named `name` anywhere under `dir`. mmdeploy bundles nest
/// the payload under a dated directory, so the depth is not fixed.
fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut dirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            dirs.push(path);
        } else if path.file_name().is_some_and(|f| f == name) {
            return Some(path);
        }
    }
    dirs.into_iter().find_map(|d| find_file(&d, name))
}

// ------------------------------------------------------------------ python

/// A usable interpreter: either `python` or the `py -3` launcher.
struct Interpreter {
    program: String,
    leading_args: Vec<String>,
}

impl Interpreter {
    fn command(&self) -> Command {
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.leading_args);
        cmd
    }
}

/// Marker the probe script prints. Presence of this string in stdout is
/// the acceptance signal — see [`find_interpreter`].
const INTERPRETER_OK: &str = "VULVATAR_PY_OK";

/// Locate a *working* Python interpreter, trying `python` before the
/// `py -3` launcher (the order `dev.ps1` uses).
///
/// The probe runs a real import and checks for [`INTERPRETER_OK`] in
/// stdout. Neither half of that is redundant:
///
/// * `--version` is not enough. A broken conda install answers
///   `--version` with `Python 3.12.9` and then dies on every actual
///   invocation with `Fatal Python error: init_fs_encoding` — measured
///   on this project's own dev machine, where it silently shadowed a
///   perfectly good `py -3`.
/// * The exit status is not enough either. That same interpreter exits
///   **0** while printing the fatal error, so only its output
///   distinguishes it from a healthy one.
///
/// `venv` is imported specifically because that is the first thing every
/// caller needs; a stdlib that cannot produce it is no use here.
fn find_interpreter() -> Result<Interpreter, String> {
    let mut rejected = Vec::new();
    for (program, leading) in [("python", vec![]), ("py", vec!["-3".to_string()])] {
        let mut cmd = Command::new(program);
        cmd.args(&leading)
            .arg("-c")
            .arg(format!("import venv; print('{INTERPRETER_OK}')"));
        let Ok(mut child) = spawn(cmd) else {
            rejected.push(format!("{program}: not on PATH"));
            continue;
        };
        let mut stdout = String::new();
        if let Some(mut pipe) = child.stdout.take() {
            let _ = pipe.read_to_string(&mut stdout);
        }
        let ok = child.wait().map(|s| s.success()).unwrap_or(false);
        if ok && stdout.contains(INTERPRETER_OK) {
            return Ok(Interpreter {
                program: program.to_string(),
                leading_args: leading,
            });
        }
        rejected.push(format!("{program}: did not report a working stdlib"));
    }
    Err(format!(
        "no working Python interpreter found ({})",
        rejected.join("; ")
    ))
}

/// Interpreter path inside a venv directory (Windows layout).
fn venv_python(root: &Path, dir: &str) -> PathBuf {
    root.join(dir).join("Scripts").join("python.exe")
}

/// True when `python` can import every name in `imports`.
///
/// Used by [`super::Probe::Venv`], so it runs on the startup scan path:
/// one short process, and only when the interpreter file already exists.
///
/// Acceptance requires [`INTERPRETER_OK`] on stdout, not just exit 0 —
/// an interpreter with a broken stdlib can exit 0 while printing a fatal
/// error (see [`find_interpreter`]). Trusting the status alone here would
/// make the scan call a dead venv "provisioned", and the face sidecar
/// would then respawn against it once per frame.
pub fn venv_imports_ok(python: &Path, imports: &[&str]) -> bool {
    if imports.is_empty() {
        return true;
    }
    let script = imports
        .iter()
        .map(|m| format!("import {m}; "))
        .chain(std::iter::once(format!("print('{INTERPRETER_OK}')")))
        .collect::<String>();
    let mut cmd = Command::new(python);
    cmd.arg("-c").arg(script);
    let Ok(mut child) = spawn(cmd) else {
        return false;
    };
    let mut stdout = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        let _ = pipe.read_to_string(&mut stdout);
    }
    child.wait().map(|s| s.success()).unwrap_or(false) && stdout.contains(INTERPRETER_OK)
}

/// Create `dir` as a venv (if absent) and install `packages` into it.
/// Returns the venv interpreter.
///
/// Idempotent: an existing venv that already imports everything is left
/// alone, so a second dependency sharing the venv costs one `import`
/// probe rather than a re-download.
fn ensure_venv(dir: &str, packages: &[&str], ctx: &StepCtx) -> Result<PathBuf, String> {
    let python = venv_python(ctx.root, dir);
    let import_names: Vec<String> = packages.iter().map(|p| import_name(p)).collect();
    let import_refs: Vec<&str> = import_names.iter().map(|s| s.as_str()).collect();
    if python.is_file() && venv_imports_ok(&python, &import_refs) {
        return Ok(python);
    }

    if !python.is_file() {
        ctx.progress(Stage::CreatingVenv, None);
        let interpreter = find_interpreter()?;
        let target = ctx.root.join(dir);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
        let mut cmd = interpreter.command();
        cmd.arg("-m").arg("venv").arg(&target);
        let mut child = spawn(cmd).map_err(|e| format!("python -m venv: {e}"))?;
        let status = wait_with_progress(&mut child, ctx, Stage::CreatingVenv, || None)?;
        if !status.ok || !python.is_file() {
            return Err(format!(
                "could not create the virtual environment at {}: {}",
                target.display(),
                status.detail("python -m venv")
            ));
        }
    }

    ctx.progress(Stage::InstallingPackages, None);
    let mut cmd = Command::new(&python);
    cmd.arg("-m")
        .arg("pip")
        .arg("install")
        .arg("--disable-pip-version-check")
        .args(packages);
    let mut child = spawn(cmd).map_err(|e| format!("pip install: {e}"))?;
    let status = wait_with_progress(&mut child, ctx, Stage::InstallingPackages, || None)?;
    if !status.ok {
        return Err(format!(
            "pip install {} failed: {}",
            packages.join(" "),
            status.detail("pip")
        ));
    }
    Ok(python)
}

/// Import name for a distribution name, where the two differ.
fn import_name(package: &str) -> String {
    match package {
        "ai-edge-litert" => "ai_edge_litert".to_string(),
        other => other.replace('-', "_"),
    }
}

/// An already-provisioned ultralytics environment, if one exists.
///
/// Checked in preference order: the canonical `tools/` location, then
/// the `%TEMP%` path `dev.ps1` used. Only returns an interpreter that
/// can actually import `ultralytics` — a venv whose install failed
/// half-way would otherwise be picked up and fail at export time, far
/// from the cause.
fn existing_export_venv(root: &Path) -> Option<PathBuf> {
    let mut candidates = vec![venv_python(root, YOLO_EXPORT_VENV)];
    if let Ok(temp) = std::env::var("TEMP") {
        candidates.push(
            PathBuf::from(temp)
                .join("yolo_export_venv")
                .join("Scripts")
                .join("python.exe"),
        );
    }
    candidates
        .into_iter()
        .find(|p| p.is_file() && venv_imports_ok(p, &["ultralytics"]))
}

/// Export `weights` to `dest` at `imgsz`, provisioning the shared
/// ultralytics venv first.
///
/// ultralytics writes the ONNX next to the checkpoint under its own
/// name, so the result is moved into `models/` under the `_<imgsz>`
/// filename the detector's candidate list expects — that suffix is the
/// input-size contract, not decoration.
fn ultralytics_export(weights: &str, dest: &str, imgsz: u32, ctx: &StepCtx) -> Result<(), String> {
    // Absolute (see `ProvisionJob::spawn`) — this is both the existence
    // check and what the child is handed, so the two cannot disagree.
    let weights_path = ctx.root.join(weights);
    if !weights_path.is_file() {
        return Err(format!(
            "{weights} is not in the working directory; the export needs the \
             ultralytics checkpoint, which ships with the source tree but not \
             with the installer"
        ));
    }

    // torch is ~2.5 GB, so never build a second environment that
    // already exists somewhere else. `dev.ps1`'s export entry has
    // historically used `%TEMP%\yolo_export_venv`; reuse it when it is
    // still usable rather than making the user pay for torch twice.
    // The canonical location stays `tools/`, because `%TEMP%` is
    // subject to Windows disk-cleanup.
    let python = match existing_export_venv(ctx.root) {
        Some(p) => {
            info!(
                "provisioning: reusing the ultralytics venv at {}",
                p.display()
            );
            p
        }
        None => ensure_venv(YOLO_EXPORT_VENV, &["ultralytics", "onnx", "onnxslim"], ctx)?,
    };
    if ctx.cancelled() {
        return Err("cancelled".to_string());
    }

    ctx.progress(Stage::Exporting, None);
    let script = format!(
        "from ultralytics import YOLO; \
         YOLO(r'{}').export(format='onnx', imgsz={imgsz}, opset=17, simplify=True)",
        weights_path.display()
    );
    let mut cmd = Command::new(&python);
    cmd.arg("-c").arg(script).current_dir(ctx.root);
    let mut child = spawn(cmd).map_err(|e| format!("ultralytics export: {e}"))?;
    let status = wait_with_progress(&mut child, ctx, Stage::Exporting, || None)?;
    if !status.ok {
        return Err(format!(
            "ONNX export failed: {}",
            status.detail("ultralytics")
        ));
    }

    let produced = weights_path.with_extension("onnx");
    if !produced.is_file() {
        return Err(format!(
            "the export reported success but {} was not written",
            produced.display()
        ));
    }
    let dest = ctx.root.join(dest);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    std::fs::rename(&produced, &dest)
        .map_err(|e| format!("could not move the export into models/: {e}"))?;
    Ok(())
}

// ----------------------------------------------------------- process utils

fn spawn(mut cmd: Command) -> std::io::Result<Child> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd.spawn()
}

/// Exit status plus whatever the child said, for error messages.
struct ChildStatus {
    ok: bool,
    code: Option<i32>,
    stderr: String,
}

impl ChildStatus {
    /// A short, quotable reason. Falls back to the exit code when the
    /// tool was silent (`curl --silent` on a DNS failure still prints,
    /// but `pip` can die quietly).
    fn detail(&self, what: &str) -> String {
        let trimmed = self.stderr.trim();
        if !trimmed.is_empty() {
            // Keep the message toast-sized; the full text is in the log.
            let line = trimmed.lines().next_back().unwrap_or(trimmed);
            return line.chars().take(300).collect();
        }
        match self.code {
            Some(c) => format!("{what} exited with status {c}"),
            None => format!("{what} was terminated"),
        }
    }
}

/// Wait for `child`, re-emitting `stage` progress from `fraction` and
/// killing it if the job is cancelled.
///
/// `fraction` returning `None` means "this stage has no measurable
/// size" — the loop then just polls for exit and cancellation, and the
/// single announcement the caller made before spawning stands.
fn wait_with_progress(
    child: &mut Child,
    ctx: &StepCtx,
    stage: Stage,
    fraction: impl Fn() -> Option<f32>,
) -> Result<ChildStatus, String> {
    loop {
        if ctx.cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Err("cancelled".to_string());
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stderr = String::new();
                if let Some(mut pipe) = child.stderr.take() {
                    let _ = pipe.read_to_string(&mut stderr);
                }
                // Drain stdout too: a full pipe buffer would have
                // blocked the child before it could exit, and some of
                // these tools report errors there.
                if let Some(mut pipe) = child.stdout.take() {
                    let mut out = String::new();
                    let _ = pipe.read_to_string(&mut out);
                    if stderr.trim().is_empty() {
                        stderr = out;
                    }
                }
                return Ok(ChildStatus {
                    ok: status.success(),
                    code: status.code(),
                    stderr,
                });
            }
            Ok(None) => {
                if let Some(f) = fraction() {
                    ctx.progress(stage, Some(f));
                }
                thread::sleep(POLL_INTERVAL);
            }
            Err(e) => return Err(format!("waiting for a child process failed: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_names_map_distribution_names() {
        assert_eq!(import_name("ai-edge-litert"), "ai_edge_litert");
        assert_eq!(import_name("numpy"), "numpy");
        assert_eq!(import_name("onnxslim"), "onnxslim");
    }

    #[test]
    fn venv_python_uses_the_windows_layout() {
        let p = venv_python(Path::new("C:/app"), "tools/face98-venv");
        assert!(p.ends_with("Scripts/python.exe") || p.ends_with("Scripts\\python.exe"));
    }

    /// A missing interpreter must be reported as absent, not as a
    /// successful import.
    #[test]
    fn venv_imports_fail_for_a_missing_interpreter() {
        assert!(!venv_imports_ok(
            Path::new("C:/definitely/not/here/python.exe"),
            &["numpy"]
        ));
    }

    #[test]
    fn empty_import_list_is_trivially_satisfied() {
        assert!(venv_imports_ok(Path::new("C:/nope/python.exe"), &[]));
    }

    /// An interpreter is only accepted when it PRINTS the marker.
    ///
    /// Regression guard for a real machine: a broken conda install
    /// answered `--version` with a version string and exited 0 from
    /// `-c` while printing `Fatal Python error: init_fs_encoding`, so
    /// both the old signals (version OK, status OK) said "healthy" and
    /// every venv/pip/export step then failed. `cmd.exe /c exit 0`
    /// stands in for that shape: exits 0, prints nothing.
    #[test]
    fn exit_zero_without_the_marker_is_not_a_working_interpreter() {
        assert!(
            !venv_imports_ok(Path::new("cmd.exe"), &["sys"]),
            "a silent exit-0 process must not pass as a Python stdlib"
        );
    }

    #[test]
    fn child_status_prefers_the_tool_message_over_the_exit_code() {
        let s = ChildStatus {
            ok: false,
            code: Some(22),
            stderr: "curl: (22) The requested URL returned error: 404\n".to_string(),
        };
        assert!(s.detail("curl").contains("404"));

        let silent = ChildStatus {
            ok: false,
            code: Some(1),
            stderr: String::new(),
        };
        assert_eq!(silent.detail("pip"), "pip exited with status 1");
    }

    /// Error text is embedded in a toast; an unbounded stderr dump would
    /// push the rest of the UI off screen.
    #[test]
    fn child_status_detail_is_bounded() {
        let s = ChildStatus {
            ok: false,
            code: Some(1),
            stderr: "x".repeat(10_000),
        };
        assert!(s.detail("curl").chars().count() <= 300);
    }
}
