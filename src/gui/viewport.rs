use eframe::egui;

use crate::gui::GuiApp;

pub fn draw(ctx: &egui::Context, _state: &mut GuiApp) {
    egui::CentralPanel::default()
        .frame(egui::Frame::none())
        .show(ctx, |ui| {
            let desired = ui.available_size();
            let (rect, _response) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());
            let painter = ui.painter_at(rect);

            painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(25, 25, 30));

            let grid_color = egui::Color32::from_rgb(45, 45, 55);
            let spacing = 40.0_f32;

            let mut y = rect.top();
            while y <= rect.bottom() {
                painter.line_segment(
                    [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                    egui::Stroke::new(0.5, grid_color),
                );
                y += spacing;
            }

            let mut x = rect.left();
            while x <= rect.right() {
                painter.line_segment(
                    [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                    egui::Stroke::new(0.5, grid_color),
                );
                x += spacing;
            }

            let center = rect.center();
            let cross_color = egui::Color32::from_rgb(70, 70, 85);
            painter.line_segment(
                [
                    egui::pos2(center.x - 12.0, center.y),
                    egui::pos2(center.x + 12.0, center.y),
                ],
                egui::Stroke::new(1.0, cross_color),
            );
            painter.line_segment(
                [
                    egui::pos2(center.x, center.y - 12.0),
                    egui::pos2(center.x, center.y + 12.0),
                ],
                egui::Stroke::new(1.0, cross_color),
            );

            painter.text(
                egui::pos2(center.x, center.y - 20.0),
                egui::Align2::CENTER_CENTER,
                "Viewport",
                egui::FontId::proportional(20.0),
                egui::Color32::from_rgb(100, 100, 115),
            );
            painter.text(
                egui::pos2(center.x, center.y + 20.0),
                egui::Align2::CENTER_CENTER,
                "Vulkano render target placeholder",
                egui::FontId::proportional(13.0),
                egui::Color32::from_rgb(75, 75, 85),
            );
        });
}
