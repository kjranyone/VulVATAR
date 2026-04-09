use crate::output::OutputFrame;
use crate::renderer::frame_input::OutputTargetRequest;

#[derive(Clone, Debug)]
pub struct ExportRequest {
    pub output_request: OutputTargetRequest,
    pub color_image_id: u64,
    pub alpha_image_id: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct ExportResult {
    pub exported_frame: Option<ExportedFrame>,
    pub export_succeeded: bool,
}

#[derive(Clone, Debug)]
pub struct ExportedFrame {
    pub output_frame: OutputFrame,
    pub gpu_token_id: u64,
    pub export_metadata: ExportMetadata,
}

#[derive(Clone, Debug)]
pub struct ExportMetadata {
    pub width: u32,
    pub height: u32,
    pub has_alpha: bool,
    pub color_space: String,
    pub timestamp_nanos: u64,
}

pub struct OutputExporter {
    export_count: u64,
}

impl OutputExporter {
    pub fn new() -> Self {
        Self { export_count: 0 }
    }

    pub fn export(
        &mut self,
        request: &ExportRequest,
        result_width: u32,
        result_height: u32,
        result_has_alpha: bool,
        timestamp_nanos: u64,
    ) -> ExportResult {
        if !request.output_request.output_enabled {
            return ExportResult {
                exported_frame: None,
                export_succeeded: false,
            };
        }

        let token_id = self.export_count;
        self.export_count += 1;

        let output_frame =
            OutputFrame::new(token_id, request.output_request.extent, timestamp_nanos);

        ExportResult {
            exported_frame: Some(ExportedFrame {
                output_frame,
                gpu_token_id: token_id,
                export_metadata: ExportMetadata {
                    width: result_width,
                    height: result_height,
                    has_alpha: result_has_alpha,
                    color_space: "srgb".to_string(),
                    timestamp_nanos,
                },
            }),
            export_succeeded: true,
        }
    }
}
