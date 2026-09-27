//! Startup dependency resolution.
//!
//! `models/` is gitignored and the installer ships only part of it, so a
//! fresh checkout — and a fresh install — starts with a runtime that
//! cannot track. Before this module the app only *reported* that
//! ("run ./dev.ps1 setup"), which is both an extra manual step and, for
//! the distilled hand nets, plain wrong advice: `dev.ps1` cannot obtain
//! them either.
//!
//! What this module adds is a machine-readable manifest of every runtime
//! dependency, a cheap [`scan`] that says which are missing, and (in
//! [`job`]) a worker that resolves the ones that *can* be resolved. The
//! manifest is deliberately the single source of truth for URLs and
//! filenames on the Rust side; `dev.ps1` keeps its own copy because the
//! installer does not ship `dev.ps1`, so the app cannot delegate to it.
//!
//! Three honest categories, and the UI must keep them distinct:
//!
//! * [`Resolution::Download`] — a plain fetch. Cheap, always possible.
//! * [`Resolution::PythonVenv`] / [`Resolution::UltralyticsExport`] —
//!   needs a Python toolchain. The YOLO26 export in particular pulls
//!   torch (~2.5 GB on first run), so it is never started without
//!   explicit consent.
//! * [`Resolution::Manual`] — distilled offline from recordings that
//!   are not redistributable (the hand presence/palm nets, the face
//!   blendshape MLP). No automation exists; the UI says so instead of
//!   pointing at a script that would not help.
//!
//! Paths in the manifest are relative to the process working directory,
//! which is where the runtime already looks for `models/` (see
//! `tracking::worker`'s `create_pose_provider_live("models", …)`) and
//! what the installer's shortcut `WorkingDir` pins.

pub mod job;

use std::path::Path;

pub use job::{ProvisionJob, ProvisionMessage, ProvisionProgress, StepOutcome};

/// What the user loses while a dependency is missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Tracking refuses to start at all.
    Blocking,
    /// Tracking runs; one chain (face, hands) stays off.
    Degraded,
}

/// How the presence of a dependency is detected.
#[derive(Debug, Clone, Copy)]
pub enum Probe {
    /// Satisfied when ANY listed path exists — the export-filename
    /// convention where several interchangeable variants are accepted
    /// (`yolo26n-pose_480.onnx` or `yolo26s-pose_640.onnx` or …).
    AnyFile(&'static [&'static str]),
    /// Satisfied when EVERY listed path exists.
    AllFiles(&'static [&'static str]),
    /// Satisfied when the interpreter exists *and* every import
    /// resolves. A venv directory that exists but whose `pip install`
    /// failed half-way is worse than no venv: the sidecar would spawn
    /// it, die on the missing module, and respawn every frame.
    Venv {
        python: &'static str,
        imports: &'static [&'static str],
    },
}

/// One archive in a [`Resolution::DownloadZip`]: fetch `url`, pull out
/// the entry whose file name is `keep`, write it to `dest`.
///
/// `keep` is matched on the file name alone, at any depth — upstream
/// bundles nest their payload under a dated or versioned directory that
/// changes between releases.
#[derive(Debug, Clone, Copy)]
pub struct ZipItem {
    pub url: &'static str,
    pub keep: &'static str,
    pub dest: &'static str,
}

/// How a missing dependency is obtained.
#[derive(Debug, Clone, Copy)]
pub enum Resolution {
    /// Fetch `url` into `dest` (via `curl.exe`, same as `dev.ps1`).
    Download {
        url: &'static str,
        dest: &'static str,
    },
    /// Fetch each zip and keep one entry out of it.
    ///
    /// A slice rather than a single item because some dependencies are
    /// one logical thing split across archives upstream — the CJK font
    /// set is three separate release zips, and listing them as three
    /// dependencies would put three lines in the consent dialog for what
    /// the user thinks of as "the fonts".
    DownloadZip { items: &'static [ZipItem] },
    /// Create a venv at `dir` and `pip install` `packages` into it.
    PythonVenv {
        dir: &'static str,
        packages: &'static [&'static str],
    },
    /// Export `dest` from the local ultralytics `weights` checkpoint at
    /// square input size `imgsz`. Needs the shared export venv
    /// ([`YOLO_EXPORT_VENV`]), which this drags in on demand.
    UltralyticsExport {
        weights: &'static str,
        dest: &'static str,
        imgsz: u32,
    },
    /// Produced by offline distillation — there is nothing to download
    /// and nothing to run. `doc` names where the procedure is written
    /// down.
    Manual { doc: &'static str },
}

impl Resolution {
    /// Whether this app can produce the artifact on its own.
    pub fn is_automatic(&self) -> bool {
        !matches!(self, Resolution::Manual { .. })
    }

    /// Whether resolving this needs a Python interpreter on PATH.
    pub fn needs_python(&self) -> bool {
        matches!(
            self,
            Resolution::PythonVenv { .. } | Resolution::UltralyticsExport { .. }
        )
    }
}

/// One runtime dependency.
#[derive(Debug, Clone, Copy)]
pub struct Dependency {
    /// Stable identifier; also the i18n key suffix
    /// (`provisioning.dep.<id>`) for the label and the "what you lose"
    /// line.
    pub id: &'static str,
    pub probe: Probe,
    pub resolution: Resolution,
    /// Approximate bytes transferred, for the consent dialog. Zero when
    /// the cost is not a download (or is unknowable, like pip's
    /// dependency closure) — the UI then shows the package list instead
    /// of a bogus number.
    pub approx_bytes: u64,
    pub severity: Severity,
}

impl Dependency {
    /// True when the artifact is already present under `root`.
    pub fn is_satisfied(&self, root: &Path) -> bool {
        match self.probe {
            Probe::AnyFile(paths) => paths.iter().any(|p| root.join(p).is_file()),
            Probe::AllFiles(paths) => paths.iter().all(|p| root.join(p).is_file()),
            Probe::Venv { python, imports } => {
                let interpreter = root.join(python);
                interpreter.is_file() && job::venv_imports_ok(&interpreter, imports)
            }
        }
    }
}

/// Shared venv for ultralytics exports. Under `tools/`, which is
/// gitignored, and reused across exports so torch is downloaded once.
pub const YOLO_EXPORT_VENV: &str = "tools/yolo-export-venv";

/// Venv the RTMPose-face sidecar is launched with. Mirrors the path
/// `dev.ps1`'s `Install-FaceSidecarEnv` provisions, so the two agree on
/// one location and a user who ran either gets the other's benefit.
pub const FACE_SIDECAR_VENV: &str = "tools/face98-venv";

/// Interpreter inside a venv, relative to the working directory.
/// Windows-only layout, like the rest of this crate.
pub const FACE_SIDECAR_PYTHON: &str = "tools/face98-venv/Scripts/python.exe";

/// Every runtime dependency, most severe first.
///
/// Deliberately does NOT list `rtmw3d.onnx` / `yolox.onnx`: `dev.ps1`
/// still fetches them (449 MB), but nothing in `src/` loads either any
/// more — not the runtime, not the diagnostic bins. The installer
/// stopped shipping them on 2026-09-23. Listing them here would make
/// the app download half a gigabyte it never opens.
pub fn manifest() -> &'static [Dependency] {
    &[
        // The body detector. Without it `create_pose_provider` fails and
        // tracking reports a blocking error, so this is the one entry
        // whose absence stops the product working at all.
        //
        // The probe mirrors `detector::yolo26`'s candidate list exactly;
        // if that list gains a filename, this must gain it too or the
        // app will offer to re-export a model it can already load.
        Dependency {
            id: "yolo26_pose",
            probe: Probe::AnyFile(&[
                "models/yolo26-pose.onnx",
                "models/yolo26n-pose_480.onnx",
                "models/yolo26s-pose_480.onnx",
                "models/yolo26n-pose_640.onnx",
                "models/yolo26s-pose_640.onnx",
            ]),
            resolution: Resolution::UltralyticsExport {
                weights: "yolo26n-pose.pt",
                dest: "models/yolo26n-pose_480.onnx",
                imgsz: 480,
            },
            // Dominated by the one-time torch wheel, not the 11 MB result.
            approx_bytes: 2_500_000_000,
            severity: Severity::Blocking,
        },
        // Face chain, part 1: the WFLW98 landmark model. A ready-made
        // Apache-2 export, so this is the one dependency that is a plain
        // download.
        Dependency {
            id: "face_landmarks",
            probe: Probe::AnyFile(&["models/rtm_face_fp16.tflite"]),
            resolution: Resolution::Download {
                // One line, like every URL here — see `manifest_urls_are_clean`.
                url: "https://huggingface.co/litert-community/RTMPose-Face-WFLW-LiteRT/resolve/main/rtm_face_fp16.tflite",
                dest: "models/rtm_face_fp16.tflite",
            },
            approx_bytes: 33_600_000,
            severity: Severity::Degraded,
        },
        // Face chain, part 2: the interpreter the sidecar runs in. The
        // Rust runtime is onnxruntime and cannot load tflite, so the
        // landmark model above is useless without this.
        Dependency {
            id: "face_sidecar_venv",
            probe: Probe::Venv {
                python: FACE_SIDECAR_PYTHON,
                imports: &["ai_edge_litert", "numpy"],
            },
            resolution: Resolution::PythonVenv {
                dir: FACE_SIDECAR_VENV,
                packages: &["ai-edge-litert", "numpy"],
            },
            approx_bytes: 60_000_000,
            severity: Severity::Degraded,
        },
        // Face chain, part 3: the WFLW98 -> MP478 correspondence and the
        // canonical mesh the Procrustes fit needs. Generated offline by
        // `scratchpad/build_face98_mapping.py` from the WFLW test split,
        // but tiny (<50 KB combined) — they are committed rather than
        // provisioned, so this entry exists only to name them if someone
        // deletes them.
        Dependency {
            id: "face_canonical",
            probe: Probe::AllFiles(&["models/mp_canonical478.npy", "models/mp_wflw98_idx.json"]),
            resolution: Resolution::Manual {
                doc: "scratchpad/build_face98_mapping.py",
            },
            approx_bytes: 0,
            severity: Severity::Degraded,
        },
        // Hand chain, part 1: the landmark model — the ONE hand file
        // that is mandatory. `RtmposeHand::try_from_models_dir` returns
        // `Ok(None)` without it, which `HandBackend` turns into "the
        // hand chain cannot start"; the other two nets below are
        // `Option` fields that only degrade quality.
        //
        // This IS downloadable: mmpose publishes the exact export as an
        // mmdeploy bundle. Verified against the graph, not the bundle's
        // own metadata — `detail.json` claims `input_shape: [192, 256]`,
        // but the ONNX reads `(batch, 3, 256, 256)` ->
        // `simcc_x/simcc_y (batch, 21, 512)`, which is exactly the
        // contract `fusion::hands` implements (512 bins = 2x the input
        // side, ImageNet-normalised RGB).
        Dependency {
            id: "hand_rtmpose",
            probe: Probe::AnyFile(&["models/rtmpose-m-hand_256.onnx"]),
            resolution: Resolution::DownloadZip {
                items: &[ZipItem {
                    // URLs stay on one line — see `manifest_urls_are_clean`.
                    url: "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip",
                    keep: "end2end.onnx",
                    dest: "models/rtmpose-m-hand_256.onnx",
                }],
            },
            approx_bytes: 51_300_000,
            severity: Severity::Degraded,
        },
        // Hand chain, part 2: the presence classifier and the palm
        // proposal heatmap. Both are distilled from this camera's own
        // recordings against MediaPipe labels — the datasets are not
        // redistributable, no upstream publishes these contracts, and
        // the training scripts AGENTS.md names were never tracked in git
        // (`/scratchpad/` and `/datasets/` are both ignored), so they
        // cannot be recovered from history either.
        //
        // Their absence is a quality loss, not an outage: presence falls
        // back to the SimCC sharpness proxy and acquisition falls back
        // to heuristic crops (see the `warn!` in `RtmposeHand`).
        Dependency {
            id: "hand_refiners",
            probe: Probe::AllFiles(&[
                "models/rtmpose-hand-presence_64.onnx",
                "models/rtmpose-hand-palm_256.onnx",
            ]),
            resolution: Resolution::Manual {
                doc: "AGENTS.md#tracking-fusion-estimator (retrain required)",
            },
            approx_bytes: 0,
            severity: Severity::Degraded,
        },
        // The GUI's fonts. `assets/` is gitignored and none of these are
        // committed, so a fresh checkout renders Japanese/Korean/Chinese
        // as tofu and every icon as a blank box — and the warning it
        // printed said "run dev.ps1 Install-Font", which a user who
        // cannot read the UI cannot act on.
        //
        // `build_font_definitions` reads them from `assets/` at startup,
        // but egui accepts `set_fonts` at any time, so `gui::provisioning`
        // re-applies them the moment these land — no restart.
        Dependency {
            id: "cjk_fonts",
            probe: Probe::AllFiles(&[
                "assets/NotoSansJP-Regular.otf",
                "assets/NotoSansKR-Regular.otf",
                "assets/NotoSansSC-Regular.otf",
            ]),
            resolution: Resolution::DownloadZip {
                items: &[
                    ZipItem {
                        url: "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/16_NotoSansJP.zip",
                        keep: "NotoSansJP-Regular.otf",
                        dest: "assets/NotoSansJP-Regular.otf",
                    },
                    // KR and SC are not redundant with JP: without KR,
                    // Hangul is tofu; without SC, simplified-only forms
                    // fall back to Japanese shapes.
                    ZipItem {
                        url: "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/17_NotoSansKR.zip",
                        keep: "NotoSansKR-Regular.otf",
                        dest: "assets/NotoSansKR-Regular.otf",
                    },
                    ZipItem {
                        url: "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/18_NotoSansSC.zip",
                        keep: "NotoSansSC-Regular.otf",
                        dest: "assets/NotoSansSC-Regular.otf",
                    },
                ],
            },
            approx_bytes: 120_000_000,
            severity: Severity::Degraded,
        },
        // Material Symbols Rounded — every icon glyph in the mode nav,
        // top bar and status indicators. Google publishes no
        // static-weight build, so the variable font is the only
        // embeddable distribution.
        Dependency {
            id: "icon_font",
            probe: Probe::AnyFile(&["assets/MaterialSymbolsRounded.ttf"]),
            resolution: Resolution::Download {
                url: "https://github.com/google/material-design-icons/raw/master/variablefont/MaterialSymbolsRounded%5BFILL%2CGRAD%2Copsz%2Cwght%5D.ttf",
                dest: "assets/MaterialSymbolsRounded.ttf",
            },
            approx_bytes: 15_000_000,
            severity: Severity::Degraded,
        },
        // Optional refinement: without it the sidecar still produces
        // expressions, geometrically.
        Dependency {
            id: "face_blendshape_mlp",
            probe: Probe::AnyFile(&["models/rtmpose-face-blendshape_98.onnx"]),
            resolution: Resolution::Manual {
                doc: "scratchpad/make_face_distill_data.py",
            },
            approx_bytes: 0,
            severity: Severity::Degraded,
        },
    ]
}

/// Result of a startup scan, split by what the app can do about it.
#[derive(Debug, Default, Clone)]
pub struct Scan {
    /// Missing and resolvable without human work.
    pub resolvable: Vec<&'static Dependency>,
    /// Missing with no automated path — reported, never "fixed".
    pub manual: Vec<&'static Dependency>,
}

impl Scan {
    pub fn is_empty(&self) -> bool {
        self.resolvable.is_empty() && self.manual.is_empty()
    }

    /// True when something missing stops tracking entirely.
    pub fn has_blocking(&self) -> bool {
        self.resolvable
            .iter()
            .chain(self.manual.iter())
            .any(|d| d.severity == Severity::Blocking)
    }

    /// Total approximate transfer for the resolvable set.
    pub fn resolvable_bytes(&self) -> u64 {
        self.resolvable.iter().map(|d| d.approx_bytes).sum()
    }

    /// True when resolving the set needs a Python interpreter.
    pub fn needs_python(&self) -> bool {
        self.resolvable.iter().any(|d| d.resolution.needs_python())
    }
}

/// Probe every manifest entry against `root` (the working directory).
///
/// Cheap enough to run on the startup path: a handful of `is_file`
/// calls, plus one short `python -c "import …"` only when a venv
/// interpreter is actually present.
pub fn scan(root: impl AsRef<Path>) -> Scan {
    let root = root.as_ref();
    let mut scan = Scan::default();
    for dep in manifest() {
        if dep.is_satisfied(root) {
            continue;
        }
        if dep.resolution.is_automatic() {
            scan.resolvable.push(dep);
        } else {
            scan.manual.push(dep);
        }
    }
    scan
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A scratch root with `models/` present but empty.
    fn empty_root() -> tempdir::Root {
        tempdir::Root::new()
    }

    /// Minimal temp-dir helper — the crate has no dev-dependency on
    /// `tempfile`, and these tests only need a unique directory that
    /// cleans itself up.
    mod tempdir {
        use std::path::{Path, PathBuf};
        use std::sync::atomic::{AtomicU32, Ordering};

        static COUNTER: AtomicU32 = AtomicU32::new(0);

        pub struct Root(PathBuf);

        impl Root {
            pub fn new() -> Self {
                let n = COUNTER.fetch_add(1, Ordering::Relaxed);
                let p = std::env::temp_dir().join(format!(
                    "vulvatar_provisioning_test_{}_{n}",
                    std::process::id()
                ));
                let _ = std::fs::remove_dir_all(&p);
                std::fs::create_dir_all(p.join("models")).unwrap();
                Self(p)
            }

            pub fn path(&self) -> &Path {
                &self.0
            }
        }

        impl Drop for Root {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    fn dep(id: &str) -> &'static Dependency {
        manifest().iter().find(|d| d.id == id).unwrap()
    }

    #[test]
    fn empty_models_dir_reports_everything_missing() {
        let root = empty_root();
        let scan = scan(root.path());
        assert_eq!(scan.resolvable.len() + scan.manual.len(), manifest().len());
        assert!(scan.has_blocking(), "no detector => tracking cannot start");
    }

    #[test]
    fn distilled_hand_refiners_are_never_offered_as_automatic() {
        let root = empty_root();
        let scan = scan(root.path());
        assert!(
            scan.manual.iter().any(|d| d.id == "hand_refiners"),
            "the distilled refiners have no download; offering to fetch \
             them is the misdirection this module exists to remove"
        );
        assert!(!scan.resolvable.iter().any(|d| d.id == "hand_refiners"));
    }

    /// ...but the one MANDATORY hand file is a plain download, so it
    /// must be offered. Getting this wrong is what left the hand chain
    /// dead while the warning pointed at a script that could not help:
    /// `RtmposeHand` holds presence/palm as `Option` and treats only
    /// this file as required.
    #[test]
    fn the_required_hand_model_is_resolvable() {
        let root = empty_root();
        let scan = scan(root.path());
        let d = dep("hand_rtmpose");
        assert!(
            matches!(d.resolution, Resolution::DownloadZip { .. }),
            "mmpose publishes this export; it must not be marked Manual"
        );
        assert!(scan.resolvable.iter().any(|x| x.id == "hand_rtmpose"));
        assert!(d.approx_bytes > 0, "the consent dialog shows this size");
    }

    /// The refiners must not gate the landmark model: a root holding
    /// only the mandatory file still reports the refiners missing, and
    /// no longer re-offers the download.
    #[test]
    fn hand_parts_are_probed_independently() {
        let root = empty_root();
        fs::write(root.path().join("models/rtmpose-m-hand_256.onnx"), b"stub").unwrap();
        assert!(dep("hand_rtmpose").is_satisfied(root.path()));
        assert!(!dep("hand_refiners").is_satisfied(root.path()));
        let scan = scan(root.path());
        assert!(!scan.resolvable.iter().any(|d| d.id == "hand_rtmpose"));
        assert!(scan.manual.iter().any(|d| d.id == "hand_refiners"));
    }

    /// The detector accepts any of several export filenames; a scan must
    /// not offer a 2.5 GB torch install when one of them is present.
    #[test]
    fn any_detector_export_satisfies_the_blocking_entry() {
        for name in [
            "yolo26-pose.onnx",
            "yolo26n-pose_480.onnx",
            "yolo26s-pose_480.onnx",
            "yolo26n-pose_640.onnx",
            "yolo26s-pose_640.onnx",
        ] {
            let root = empty_root();
            fs::write(root.path().join("models").join(name), b"stub").unwrap();
            assert!(
                dep("yolo26_pose").is_satisfied(root.path()),
                "{name} should satisfy the detector probe"
            );
            assert!(
                !scan(root.path()).has_blocking(),
                "{name} present => nothing blocking"
            );
        }
    }

    /// `AllFiles` must not be satisfied by a partial set — a half-present
    /// half-present set.
    #[test]
    fn all_files_probe_requires_every_file() {
        let root = empty_root();
        assert!(!dep("hand_refiners").is_satisfied(root.path()));
        fs::write(
            root.path().join("models/rtmpose-hand-presence_64.onnx"),
            b"stub",
        )
        .unwrap();
        assert!(!dep("hand_refiners").is_satisfied(root.path()));
        fs::write(
            root.path().join("models/rtmpose-hand-palm_256.onnx"),
            b"stub",
        )
        .unwrap();
        assert!(dep("hand_refiners").is_satisfied(root.path()));
    }

    /// A venv directory that exists but cannot import its packages is
    /// NOT satisfied: the sidecar would spawn it and respawn-loop at
    /// 30 fps.
    #[test]
    fn venv_probe_rejects_a_missing_interpreter() {
        let root = empty_root();
        fs::create_dir_all(root.path().join("tools/face98-venv/Scripts")).unwrap();
        assert!(
            !dep("face_sidecar_venv").is_satisfied(root.path()),
            "an empty venv directory must not count as provisioned"
        );
    }

    /// URLs must contain no whitespace.
    ///
    /// This is not paranoia: a `\`-continued URL literal in this file
    /// was collapsed by rustfmt onto one line with its indentation left
    /// INSIDE the string, and curl rejected it ("URL rejected:
    /// Malformed input to a URL function"). Formatting must not be able
    /// to break a download again.
    #[test]
    fn manifest_urls_are_clean() {
        for dep in manifest() {
            let urls: Vec<&str> = match dep.resolution {
                Resolution::Download { url, .. } => vec![url],
                Resolution::DownloadZip { items } => items.iter().map(|i| i.url).collect(),
                _ => continue,
            };
            assert!(!urls.is_empty(), "`{}` has no URL to fetch", dep.id);
            for url in urls {
                assert!(
                    !url.chars().any(char::is_whitespace),
                    "`{}` has whitespace in its URL: {url:?}",
                    dep.id
                );
                assert!(url.starts_with("https://"), "`{}` is not https", dep.id);
            }
        }
    }

    #[test]
    fn manifest_ids_are_unique() {
        let mut ids: Vec<_> = manifest().iter().map(|d| d.id).collect();
        let total = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), total, "manifest ids double as i18n keys");
    }

    /// The manifest must not resurrect the two models that cost 449 MB
    /// and are loaded by nothing.
    #[test]
    fn unused_legacy_models_are_not_dependencies() {
        for dead in ["rtmw3d", "yolox"] {
            assert!(
                !manifest().iter().any(|d| d.id.contains(dead)),
                "{dead} is not loaded by any runtime path"
            );
        }
    }

    #[test]
    fn python_requirement_is_reported_for_consent() {
        let root = empty_root();
        let scan = scan(root.path());
        assert!(
            scan.needs_python(),
            "detector export + face venv both need an interpreter; the \
             consent dialog has to say so before starting"
        );
        assert!(scan.resolvable_bytes() > 2_000_000_000);
    }
}
