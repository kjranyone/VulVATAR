mod app;
mod asset;
mod avatar;
mod output;
mod renderer;
mod simulation;
mod tracking;

use app::Application;

fn main() {
    let mut app = Application::new();
    app.bootstrap();
    app.run_frame();
}
