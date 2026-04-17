fn main() -> Result<(), eframe::Error> {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_title("VulVATAR"),
        ..Default::default()
    };

    eframe::run_native(
        "VulVATAR",
        options,
        Box::new(|cc| Ok(Box::new(vulvatar_lib::gui::GuiApp::new(cc)))),
    )
}
