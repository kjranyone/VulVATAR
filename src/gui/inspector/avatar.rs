use eframe::egui;

use crate::gui::GuiApp;
use crate::t;

use super::library;

pub(super) fn draw_avatar(ui: &mut egui::Ui, state: &mut GuiApp) {
    egui::CollapsingHeader::new(t!("inspector.current_avatar"))
        .default_open(true)
        .show(ui, |ui| {
            if let Some(avatar) = state.app.active_avatar() {
                let asset = &avatar.asset;
                let meta = &asset.vrm_meta;

                let version_color = match meta.spec_version {
                    crate::asset::VrmSpecVersion::V1 => egui::Color32::from_rgb(120, 200, 140),
                    crate::asset::VrmSpecVersion::V0 => egui::Color32::from_rgb(220, 180, 100),
                    crate::asset::VrmSpecVersion::Unknown => egui::Color32::from_rgb(200, 100, 100),
                };
                ui.horizontal(|ui| {
                    ui.label(t!("inspector.spec"));
                    ui.label(
                        egui::RichText::new(meta.spec_version.label())
                            .strong()
                            .color(version_color),
                    );
                    if let Some(raw) = &meta.spec_version_raw {
                        ui.label(
                            egui::RichText::new(format!("({})", raw))
                                .small()
                                .color(egui::Color32::GRAY),
                        );
                    }
                });

                if meta.spec_version == crate::asset::VrmSpecVersion::V0 {
                    ui.label(
                        egui::RichText::new(t!("inspector.vrm0x_note"))
                        .small()
                        .color(egui::Color32::from_rgb(220, 180, 100)),
                    );
                }

                let title = meta.title.as_deref().unwrap_or("—");
                ui.horizontal(|ui| {
                    ui.label(t!("inspector.title"));
                    ui.label(title);
                });
                let authors = if meta.authors.is_empty() {
                    "—".to_string()
                } else {
                    meta.authors.join(", ")
                };
                ui.horizontal(|ui| {
                    ui.label(t!("inspector.authors"));
                    ui.label(authors);
                });
                if let Some(v) = &meta.model_version {
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.model_version"));
                        ui.label(v);
                    });
                }
                if let Some(c) = &meta.contact_information {
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.contact"));
                        ui.label(c);
                    });
                }
                if let Some(l) = &meta.license {
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.license"));
                        ui.label(l);
                    });
                }
                if let Some(cr) = &meta.copyright_information {
                    ui.horizontal(|ui| {
                        ui.label(t!("inspector.copyright"));
                        ui.label(cr);
                    });
                }
                if !meta.references.is_empty() {
                    ui.collapsing(t!("inspector.references"), |ui| {
                        for r in &meta.references {
                            ui.label(
                                egui::RichText::new(r).small().color(egui::Color32::GRAY),
                            );
                        }
                    });
                }

                ui.separator();
                ui.label(
                    egui::RichText::new(t!("inspector.source", path = asset.source_path.display().to_string()))
                        .small()
                        .color(egui::Color32::GRAY),
                );
                ui.horizontal(|ui| {
                    ui.label(t!("inspector.meshes", count = asset.meshes.len()));
                    ui.label(t!("inspector.materials", count = asset.materials.len()));
                });
                ui.horizontal(|ui| {
                    ui.label(t!("inspector.spring_chains", count = asset.spring_bones.len()));
                    ui.label(t!("inspector.colliders", count = asset.colliders.len()));
                });
                ui.horizontal(|ui| {
                    ui.label(
                        if asset.humanoid.is_some() { t!("inspector.humanoid_yes") } else { t!("inspector.humanoid_no") }
                    );
                    ui.label(t!("inspector.expressions", count = asset.default_expressions.expressions.len()));
                    ui.label(t!("inspector.animations", count = asset.animation_clips.len()));
                });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.button(t!("inspector.reload")).clicked() {
                        state.app.reload_avatar();
                    }
                    if ui.button(t!("inspector.detach")).clicked()
                        && !state.app.avatars.is_empty()
                    {
                        state.app.remove_avatar_at(state.app.active_avatar_index);
                    }
                });
            } else {
                ui.label(t!("inspector.no_avatar_loaded"));
                ui.label(
                    egui::RichText::new(t!("inspector.no_avatar_hint"))
                        .small()
                        .color(egui::Color32::GRAY),
                );
            }
        });

    library::draw_model_library(ui, state);
}
