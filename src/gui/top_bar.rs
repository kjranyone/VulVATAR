use eframe::egui;

use crate::gui::GuiApp;

pub fn draw(ctx: &egui::Context, state: &mut GuiApp) {
    egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.heading("VulVATAR");
            ui.separator();

            if ui.button("Open Avatar").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("VRM 1.0", &["vrm"])
                    .pick_file()
                {
                    let loader = crate::asset::vrm::VrmAssetLoader::new();
                    match loader.load(path.to_string_lossy().as_ref()) {
                        Ok(asset) => {
                            let instance_id =
                                crate::avatar::AvatarInstanceId(state.app.next_avatar_instance_id);
                            state.app.next_avatar_instance_id += 1;
                            state.app.physics.attach_avatar(&asset);
                            state.app.avatar =
                                Some(crate::avatar::AvatarInstance::new(instance_id, asset));
                        }
                        Err(e) => {
                            eprintln!("failed to load avatar: {}", e);
                        }
                    }
                }
            }

            ui.menu_button("Recent", |ui| {
                ui.label("No recent avatars");
            });

            ui.separator();

            if ui.button("Save Project").clicked() {}
            if ui.button("Save As").clicked() {}

            ui.separator();

            if ui.button("Open Overlay").clicked() {}
            if ui.button("Save Overlay").clicked() {
                state.app.editor.save_overlay();
            }

            ui.separator();

            if state.paused {
                if ui.button("Resume").clicked() {
                    state.paused = false;
                }
            } else {
                if ui.button("Pause").clicked() {
                    state.paused = true;
                }
            }
        });
    });
}
