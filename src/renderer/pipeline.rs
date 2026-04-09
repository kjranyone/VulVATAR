use crate::renderer::frame_input::{RenderAlphaMode, RenderCullMode};
use crate::renderer::material::MaterialShaderMode;

#[derive(Clone, Debug)]
pub enum RenderPipeline {
    SkinningUnlit,
    SkinningSimpleLit,
    SkinningToon,
    Outline,
    DepthOnly,
}

pub struct PipelineState {
    pub active_pipeline: RenderPipeline,
    pub initialized: bool,
}

impl PipelineState {
    pub fn new(pipeline: RenderPipeline) -> Self {
        Self {
            active_pipeline: pipeline,
            initialized: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub material_mode: MaterialShaderMode,
    pub alpha_mode: RenderAlphaMode,
    pub outline_enabled: bool,
    pub cull_mode: RenderCullMode,
    pub vertex_layout: VertexLayout,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum VertexLayout {
    Skinned,
    Static,
}

impl Default for VertexLayout {
    fn default() -> Self {
        Self::Skinned
    }
}

impl PipelineKey {
    pub fn select_pipeline(&self) -> RenderPipeline {
        if self.outline_enabled {
            return RenderPipeline::Outline;
        }
        match self.material_mode {
            MaterialShaderMode::Unlit => RenderPipeline::SkinningUnlit,
            MaterialShaderMode::SimpleLit => RenderPipeline::SkinningSimpleLit,
            MaterialShaderMode::ToonLike => RenderPipeline::SkinningToon,
        }
    }

    pub fn from_mesh_instance(
        material_mode: MaterialShaderMode,
        alpha_mode: RenderAlphaMode,
        cull_mode: RenderCullMode,
        outline_enabled: bool,
        has_skin: bool,
    ) -> Self {
        Self {
            material_mode,
            alpha_mode,
            outline_enabled,
            cull_mode,
            vertex_layout: if has_skin {
                VertexLayout::Skinned
            } else {
                VertexLayout::Static
            },
        }
    }
}
