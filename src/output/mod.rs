#[derive(Clone, Debug)]
pub enum FrameSink {
    VirtualCamera,
    SharedTexture,
    SharedMemory,
    ImageSequence,
}

#[derive(Clone, Debug)]
pub struct OutputFrame {
    pub width: u32,
    pub height: u32,
    pub color_space: ColorSpace,
    pub has_alpha: bool,
    pub timestamp_nanos: u64,
}

#[derive(Clone, Copy, Debug)]
pub enum ColorSpace {
    Srgb,
    LinearSrgb,
}

pub struct OutputRouter {
    sink: FrameSink,
}

impl OutputRouter {
    pub fn new(sink: FrameSink) -> Self {
        Self { sink }
    }

    pub fn publish(&self, frame: &OutputFrame) {
        println!(
            "output: publish {}x{} {:?} alpha={} to {:?} at {} ns",
            frame.width,
            frame.height,
            frame.color_space,
            frame.has_alpha,
            self.sink,
            frame.timestamp_nanos
        );
    }
}
