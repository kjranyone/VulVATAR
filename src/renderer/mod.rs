use crate::avatar::AvatarInstance;
use crate::output::{ColorSpace, OutputFrame};

pub struct VulkanRenderer {
    initialized: bool,
}

impl VulkanRenderer {
    pub fn new() -> Self {
        Self { initialized: false }
    }

    pub fn initialize(&mut self) {
        self.initialized = true;
        println!("renderer: Vulkano initialization stub");
    }

    pub fn render(&self, avatar: &AvatarInstance) -> OutputFrame {
        if !self.initialized {
            eprintln!("renderer: skipped because Vulkan is not initialized");
            return OutputFrame {
                width: 0,
                height: 0,
                color_space: ColorSpace::Srgb,
                has_alpha: true,
                timestamp_nanos: 0,
            };
        }

        println!(
            "renderer: draw avatar '{}' with {} mesh(es), {} material(s), {} cloth region(s)",
            avatar.asset.name,
            avatar.asset.meshes.len(),
            avatar.asset.materials.len(),
            avatar.cloth_state.regions.len()
        );

        OutputFrame {
            width: 1920,
            height: 1080,
            color_space: ColorSpace::Srgb,
            has_alpha: true,
            timestamp_nanos: 16_666_667,
        }
    }
}
