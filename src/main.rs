mod app;
mod asset;
mod avatar;
mod editor;
mod gui;
mod math_utils;
mod output;
mod persistence;
mod renderer;
mod simulation;
mod tracking;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_title("VulVATAR"),
        ..Default::default()
    };

    eframe::run_native(
        "VulVATAR",
        options,
        Box::new(|cc| Ok(Box::new(gui::GuiApp::new(cc)))),
    )
}
