use eframe::egui;

use crate::asset::AvatarAsset;
use crate::editor::cloth_authoring::RegionSelection;
use crate::t;

pub struct ViewportOverlayState {
    pub show_wireframe: bool,
    pub show_sim_mesh: bool,
    pub show_render_mesh: bool,
    pub show_pins: bool,
    pub show_constraints: bool,
    pub show_collision_proxies: bool,
}

impl Default for ViewportOverlayState {
    fn default() -> Self {
        Self {
            show_wireframe: true,
            show_sim_mesh: true,
            show_render_mesh: true,
            show_pins: true,
            show_constraints: true,
            show_collision_proxies: false,
        }
    }
}

#[allow(dead_code)]
pub fn draw_selection_overlay(
    ui: &mut egui::Ui,
    selection: Option<&RegionSelection>,
    avatar: &AvatarAsset,
) {
    let painter = ui.painter();
    let rect = ui.available_rect_before_wrap();

    match selection {
        Some(sel) => {
            let prim_label = avatar
                .meshes
                .iter()
                .flat_map(|m| m.primitives.iter())
                .find(|p| p.id == sel.target_primitive)
                .map(|p| t!("viewport.primitive", id = p.id.0))
                .unwrap_or_else(|| t!("viewport.primitive", id = sel.target_primitive.0));

            let vert_count = sel.selected_vertices.len();
            let range_label = t!(
                "viewport.range",
                start = sel.selected_vertex_range.0,
                end = sel.selected_vertex_range.1
            );

            let lines = [
                t!("viewport.selected", count = vert_count),
                t!("viewport.target", label = prim_label),
                range_label,
            ];

            let mut y = rect.top() + 6.0;
            for line in &lines {
                painter.text(
                    egui::pos2(rect.left() + 8.0, y),
                    egui::Align2::LEFT_TOP,
                    line,
                    egui::FontId::monospace(12.0),
                    egui::Color32::from_rgba_unmultiplied(255, 220, 80, 220),
                );
                y += 16.0;
            }
        }
        None => {
            painter.text(
                egui::pos2(rect.left() + 8.0, rect.top() + 6.0),
                egui::Align2::LEFT_TOP,
                t!("viewport.no_region"),
                egui::FontId::monospace(12.0),
                egui::Color32::from_rgba_unmultiplied(180, 180, 180, 160),
            );
        }
    }
}

pub fn draw_viewport_controls(ui: &mut egui::Ui, overlay_state: &mut ViewportOverlayState) {
    ui.horizontal(|ui| {
        ui.checkbox(&mut overlay_state.show_wireframe, t!("inspector.wireframe"));
        ui.checkbox(&mut overlay_state.show_sim_mesh, t!("inspector.sim_mesh_label"));
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut overlay_state.show_render_mesh, t!("inspector.render_mesh"));
        ui.checkbox(&mut overlay_state.show_pins, t!("inspector.pins"));
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut overlay_state.show_constraints, t!("inspector.constraints_label"));
        ui.checkbox(&mut overlay_state.show_collision_proxies, t!("inspector.colliders_label"));
    });
}
