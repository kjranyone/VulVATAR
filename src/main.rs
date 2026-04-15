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
        Box::new(|cc| Ok(Box::new(vulvatar::gui::GuiApp::new(cc)))),
    )
}
