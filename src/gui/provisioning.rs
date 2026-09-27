//! Startup dependency prompt: consent dialog, progress modal, polling.
//!
//! The scan itself lives in [`crate::provisioning`] and runs once in
//! `GuiApp::new`. This module is only the surface: it asks before
//! spending the user's bandwidth, shows what is running, and reports
//! what happened.
//!
//! Two rules shape the dialog, both of them reactions to what the old
//! "run ./dev.ps1 setup" string got wrong:
//!
//! 1. **Resolvable and unresolvable are never mixed.** The distilled
//!    hand nets cannot be fetched by anything — the dialog says so and
//!    offers no button, instead of implying a script would fix it.
//! 2. **The cost is stated before it is spent.** The detector export
//!    pulls torch; that is a ~2.5 GB first run, so the size and the
//!    Python requirement are on screen before the user agrees.

use eframe::egui;

use crate::provisioning::{self, Dependency, Scan, Severity, StepOutcome};
use crate::t;

use super::components::{self, ButtonTone};
use super::GuiApp;

/// Human-readable size for the consent dialog. Approximations only —
/// the manifest's numbers are estimates, so precision would be a lie.
fn format_bytes(bytes: u64) -> String {
    const MB: f64 = 1_000_000.0;
    const GB: f64 = 1_000_000_000.0;
    let b = bytes as f64;
    if b >= GB {
        format!("~{:.1} GB", b / GB)
    } else {
        format!("~{} MB", (b / MB).round() as u64)
    }
}

/// Localized display name for a manifest entry.
fn dep_name(dep: &Dependency) -> String {
    // Bound, not inlined: `t!` borrows its key for the duration of the
    // call, so a temporary `format!` would not outlive the expression.
    let key = format!("provisioning.dep.{}", dep.id);
    t!(&key).to_string()
}

fn severity_label(severity: Severity) -> String {
    match severity {
        Severity::Blocking => t!("provisioning.severity.blocking").to_string(),
        Severity::Degraded => t!("provisioning.severity.degraded").to_string(),
    }
}

/// Run the startup scan and decide whether to raise the prompt.
///
/// Returns the state the GUI stores. Called from `GuiApp::new`; the
/// probes are file-existence checks plus — only when a venv interpreter
/// is actually on disk — one short `python -c "import …"`.
pub fn initial_state(prompt_enabled: bool) -> ProvisioningUiState {
    let scan = provisioning::scan(".");
    ProvisioningUiState {
        // Nothing missing, or the user opted out: no dialog. The manual
        // list is still kept so the Tracking panel can explain a
        // degraded chain without a second scan.
        prompt: (prompt_enabled && !scan.resolvable.is_empty()).then(|| scan.clone()),
        manual: scan.manual.clone(),
        job: None,
    }
}

/// GUI-side provisioning state. Lives on `GuiApp`.
#[derive(Default)]
pub struct ProvisioningUiState {
    /// Pending consent dialog; cleared once the user answers.
    pub prompt: Option<Scan>,
    /// Missing dependencies with no automated path. Retained for the
    /// whole session as the honest answer to "why are my hands not
    /// tracked".
    pub manual: Vec<&'static Dependency>,
    /// In-flight resolver.
    pub job: Option<provisioning::ProvisionJob>,
}

impl ProvisioningUiState {
    pub fn is_busy(&self) -> bool {
        self.job.is_some()
    }
}

/// Drain the resolver and report the outcome. Called once per frame.
pub(super) fn poll(state: &mut GuiApp) {
    let Some(job) = state.provisioning.job.as_mut() else {
        return;
    };
    if !job.poll() {
        return;
    }
    let mut job = state.provisioning.job.take().expect("checked above");
    job.join();

    let mut ok = 0usize;
    let mut cancelled = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for (id, outcome) in &job.outcomes {
        let name = provisioning::manifest()
            .iter()
            .find(|d| d.id == *id)
            .map(dep_name)
            .unwrap_or_else(|| (*id).to_string());
        match outcome {
            StepOutcome::Done => ok += 1,
            StepOutcome::Cancelled => cancelled += 1,
            StepOutcome::Failed(e) => {
                failures.push(t!("provisioning.failed_one", name = name, error = e).to_string())
            }
        }
    }

    for failure in &failures {
        state.push_error_notification(failure.clone());
    }
    if ok > 0 {
        state.push_success_notification(t!("provisioning.done", count = ok).to_string());
        // The models are read when a provider is built, so a running
        // worker is still using the old set. Say so rather than letting
        // the user wonder why the freshly fetched chain is absent.
        if state.app.is_tracking_running() {
            state.push_notification(t!("provisioning.restart_tracking").to_string());
        }
    }
    if cancelled > 0 && failures.is_empty() && ok == 0 {
        state.push_notification(t!("provisioning.cancelled").to_string());
    }

    // Re-scan: what is still missing after this pass is what the
    // Tracking panel should keep explaining.
    state.provisioning.manual = provisioning::scan(".").manual;
}

/// Draw the consent dialog and the progress modal.
pub(super) fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    draw_prompt(ctx, state);
    draw_progress(ctx, state);
}

fn draw_prompt(ctx: &egui::Context, state: &mut GuiApp) {
    let Some(scan) = state.provisioning.prompt.clone() else {
        return;
    };
    if state.provisioning.is_busy() {
        return;
    }

    let mut start = false;
    let mut dismiss = false;
    let mut never_again = false;

    egui::Window::new(t!("provisioning.title"))
        .id(egui::Id::new("provisioning_prompt_window"))
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        // Above the viewport and any panel, like the other startup
        // modals — this one gates whether tracking can work at all.
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.set_max_width(460.0);
            ui.label(if scan.has_blocking() {
                t!("provisioning.intro_blocking")
            } else {
                t!("provisioning.intro")
            });
            ui.add_space(8.0);

            for dep in &scan.resolvable {
                ui.horizontal(|ui| {
                    ui.label("•");
                    ui.vertical(|ui| {
                        ui.label(dep_name(dep));
                        let detail = if dep.approx_bytes > 0 {
                            format!(
                                "{} — {}",
                                format_bytes(dep.approx_bytes),
                                severity_label(dep.severity)
                            )
                        } else {
                            severity_label(dep.severity)
                        };
                        ui.weak(detail);
                    });
                });
            }

            if scan.needs_python() {
                ui.add_space(6.0);
                ui.label(t!("provisioning.needs_python"));
            }

            if !scan.manual.is_empty() {
                ui.add_space(10.0);
                ui.separator();
                ui.label(t!("provisioning.manual_heading"));
                ui.weak(t!("provisioning.manual_hint"));
                for dep in &scan.manual {
                    ui.horizontal(|ui| {
                        ui.label("•");
                        ui.vertical(|ui| {
                            ui.label(dep_name(dep));
                            if let provisioning::Resolution::Manual { doc } = dep.resolution {
                                ui.weak(doc);
                            }
                        });
                    });
                }
            }

            ui.add_space(12.0);
            ui.horizontal(|ui| {
                if components::filled_button(ui, None, &t!("provisioning.fetch"), true).clicked() {
                    start = true;
                }
                if components::tonal_button(
                    ui,
                    None,
                    &t!("provisioning.later"),
                    ButtonTone::Primary,
                    true,
                )
                .clicked()
                {
                    dismiss = true;
                }
                if ui.button(t!("provisioning.never")).clicked() {
                    never_again = true;
                }
            });
        });

    if start {
        state.provisioning.job = Some(provisioning::ProvisionJob::spawn(
            // The real directory, not ".": the resolver hands this path
            // to child processes that run with their own cwd.
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
            scan.resolvable.clone(),
        ));
        state.provisioning.prompt = None;
    } else if dismiss {
        state.provisioning.prompt = None;
    } else if never_again {
        state.provisioning.prompt = None;
        state.settings.provisioning_auto_prompt = Some(false);
        state.project_status.app_settings_dirty = true;
    }
}

fn draw_progress(ctx: &egui::Context, state: &mut GuiApp) {
    let Some(job) = state.provisioning.job.as_ref() else {
        return;
    };
    let cancelling = job.is_cancelled();
    let current = job.current;
    let done = job.outcomes.len();

    let mut cancel = false;
    egui::Window::new(t!("provisioning.progress_title"))
        .id(egui::Id::new("provisioning_progress_window"))
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.set_min_width(360.0);
            match current {
                Some(p) => {
                    let name = provisioning::manifest()
                        .iter()
                        .find(|d| d.id == p.id)
                        .map(dep_name)
                        .unwrap_or_else(|| p.id.to_string());
                    ui.label(t!(
                        "provisioning.step",
                        index = p.step_index + 1,
                        total = p.step_total,
                        name = name
                    ));
                    ui.weak(t!(p.stage.i18n_key()));
                    match p.fraction {
                        Some(f) => {
                            ui.add(egui::ProgressBar::new(f).show_percentage());
                        }
                        // pip resolves a dependency closure of unknown
                        // total size; a spinner beats a fake bar.
                        None => {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.weak(t!("provisioning.no_eta"));
                            });
                        }
                    }
                }
                None => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(t!("provisioning.starting", done = done));
                    });
                }
            }
            ui.add_space(10.0);
            if cancelling {
                ui.weak(t!("provisioning.cancelling"));
            } else if components::tonal_button(
                ui,
                None,
                &t!("provisioning.cancel"),
                ButtonTone::Error,
                true,
            )
            .clicked()
            {
                cancel = true;
            }
        });

    if cancel {
        if let Some(job) = state.provisioning.job.as_ref() {
            job.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_formatting_switches_unit_at_a_gigabyte() {
        assert_eq!(format_bytes(33_600_000), "~34 MB");
        assert_eq!(format_bytes(2_500_000_000), "~2.5 GB");
    }

    /// Opting out must suppress the dialog even when the scan finds
    /// resolvable work — otherwise "never ask again" would not stick.
    #[test]
    fn opt_out_suppresses_the_prompt() {
        let state = initial_state(false);
        assert!(state.prompt.is_none());
    }

    /// The manual list is populated regardless of the opt-out, because
    /// it explains a degraded runtime rather than asking for anything.
    #[test]
    fn manual_entries_survive_the_opt_out() {
        let with = initial_state(true);
        let without = initial_state(false);
        assert_eq!(with.manual.len(), without.manual.len());
    }

    /// `rust-i18n` returns the key itself on a miss, so an untranslated
    /// dependency would ship as a literal `provisioning.dep.hand_chain`
    /// in the dialog. Check every id against every locale.
    #[test]
    fn every_dependency_is_named_in_every_locale() {
        let previous = crate::i18n::locale();
        for locale in crate::i18n::available_locales() {
            crate::i18n::set_locale(locale);
            for dep in provisioning::manifest() {
                let name = dep_name(dep);
                assert!(
                    !name.contains("provisioning.dep."),
                    "{locale}: no name for `{}`",
                    dep.id
                );
                assert!(!name.is_empty(), "{locale}: empty name for `{}`", dep.id);
            }
        }
        crate::i18n::set_locale(&previous);
    }

    /// Same for the progress stages and the severity lines — these are
    /// the strings a user reads while a 2.5 GB download runs.
    #[test]
    fn every_stage_and_severity_is_translated() {
        use crate::provisioning::job::Stage;
        let previous = crate::i18n::locale();
        for locale in crate::i18n::available_locales() {
            crate::i18n::set_locale(locale);
            for stage in [
                Stage::Downloading,
                Stage::Extracting,
                Stage::CreatingVenv,
                Stage::InstallingPackages,
                Stage::Exporting,
            ] {
                let key = stage.i18n_key();
                assert_ne!(t!(key), key, "{locale}: untranslated {key}");
            }
            for severity in [Severity::Blocking, Severity::Degraded] {
                let label = severity_label(severity);
                assert!(
                    !label.contains("provisioning.severity."),
                    "{locale}: {label}"
                );
            }
        }
        crate::i18n::set_locale(&previous);
    }
}
