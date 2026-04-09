pub mod frame_input;
pub mod material;
pub mod mtoon;
pub mod output_export;
pub mod pipeline;

use frame_input::RenderFrameInput;
use output_export::OutputExporter;

pub struct RenderResult {
    pub extent: [u32; 2],
    pub timestamp_nanos: u64,
    pub has_alpha: bool,
    pub stats: RenderStats,
    pub exported_frame: Option<output_export::ExportedFrame>,
}

#[derive(Clone, Debug, Default)]
pub struct RenderStats {
    pub instance_count: u32,
    pub mesh_count: u32,
    pub material_count: u32,
    pub cloth_instances: u32,
}

pub struct VulkanRenderer {
    initialized: bool,
    output_exporter: OutputExporter,
}

impl VulkanRenderer {
    pub fn new() -> Self {
        Self {
            initialized: false,
            output_exporter: OutputExporter::new(),
        }
    }

    pub fn initialize(&mut self) {
        self.initialized = true;
        println!("renderer: Vulkano initialization stub");
    }

    pub fn render(&mut self, input: &RenderFrameInput) -> RenderResult {
        if !self.initialized {
            eprintln!("renderer: skipped because Vulkan is not initialized");
            return RenderResult {
                extent: [0, 0],
                timestamp_nanos: 0,
                has_alpha: false,
                stats: RenderStats::default(),
                exported_frame: None,
            };
        }

        let total_meshes: u32 = input
            .instances
            .iter()
            .map(|i| i.mesh_instances.len() as u32)
            .sum();

        let total_materials: u32 = input
            .instances
            .iter()
            .flat_map(|i| i.mesh_instances.iter())
            .count() as u32;

        let cloth_instances: u32 = input
            .instances
            .iter()
            .filter(|i| i.cloth_deform.is_some())
            .count() as u32;

        for instance in &input.instances {
            println!(
                "renderer: draw instance {} mesh(es)={} cloth={}",
                instance.instance_id.0,
                instance.mesh_instances.len(),
                instance.cloth_deform.is_some(),
            );
        }

        let timestamp_nanos = 16_666_667u64;
        let extent = input.output_request.extent;

        let export_result = self.output_exporter.export(
            &output_export::ExportRequest {
                output_request: input.output_request.clone(),
                color_image_id: 0,
                alpha_image_id: None,
            },
            extent[0],
            extent[1],
            true,
            timestamp_nanos,
        );

        RenderResult {
            extent,
            timestamp_nanos,
            has_alpha: true,
            stats: RenderStats {
                instance_count: input.instances.len() as u32,
                mesh_count: total_meshes,
                material_count: total_materials,
                cloth_instances,
            },
            exported_frame: export_result.exported_frame,
        }
    }
}
