fn main() -> Result<(), eframe::Error> {
    env_logger::init();

    let _guard = match vulvatar_lib::single_instance::acquire() {
        vulvatar_lib::single_instance::SingleInstanceResult::First(guard) => guard,
        vulvatar_lib::single_instance::SingleInstanceResult::ExistingAlive { pid } => {
            let msg = format!(
                "VulVATAR is already running (PID {}).\n\n\
                 Force-quit the existing process and start a new one?",
                pid,
            );
            let yes = rfd::MessageDialog::new()
                .set_title("VulVATAR — Already Running")
                .set_description(&msg)
                .set_buttons(rfd::MessageButtons::YesNo)
                .set_level(rfd::MessageLevel::Warning)
                .show();
            if yes != rfd::MessageDialogResult::Yes {
                return Ok(());
            }
            match vulvatar_lib::single_instance::force_kill_and_acquire(pid) {
                Some(guard) => guard,
                None => {
                    rfd::MessageDialog::new()
                        .set_title("VulVATAR — Error")
                        .set_description("Could not stop the existing process.")
                        .set_level(rfd::MessageLevel::Error)
                        .show();
                    return Ok(());
                }
            }
        }
        vulvatar_lib::single_instance::SingleInstanceResult::ExistingDead { pid } => {
            log::warn!(
                "single_instance: stale lock from PID {} detected, could not acquire mutex",
                pid
            );
            rfd::MessageDialog::new()
                .set_title("VulVATAR — Startup Error")
                .set_description("Could not acquire the single-instance lock. Another process may hold it.")
                .set_level(rfd::MessageLevel::Error)
                .show();
            return Ok(());
        }
    };

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
