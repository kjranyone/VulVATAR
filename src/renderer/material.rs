use crate::asset::{AlphaMode, MaterialMode, Vec3, Vec4};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::image::sampler::Sampler;
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{GraphicsPipeline, Pipeline};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MaterialShaderMode {
    Unlit,
    SimpleLit,
    ToonLike,
}

#[derive(Clone, Debug)]
pub struct MaterialUploadRequest {
    pub base_color: Vec4,
    pub mode: MaterialShaderMode,
    pub alpha_mode: AlphaMode,
    pub double_sided: bool,
    pub toon_ramp_threshold: f32,
    pub shadow_softness: f32,
    pub outline_width: f32,
    pub outline_color: Vec3,
    pub textures: MaterialTextureSlots,
    pub mtoon_params: Option<crate::asset::MtoonStagedParams>,
}

#[derive(Clone, Debug, Default)]
pub struct MaterialTextureSlots {
    pub base_color: Option<MaterialTextureBinding>,
    pub normal_map: Option<MaterialTextureBinding>,
    pub shade_ramp: Option<MaterialTextureBinding>,
    pub emissive: Option<MaterialTextureBinding>,
}

#[derive(Clone, Debug)]
pub struct MaterialTextureBinding {
    pub uri: String,
}

impl MaterialUploadRequest {
    /// Returns a white opaque default material for use when no materials are available.
    pub fn default_material() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            mode: MaterialShaderMode::Unlit,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            toon_ramp_threshold: 0.5,
            shadow_softness: 0.5,
            outline_width: 0.0,
            outline_color: [0.0, 0.0, 0.0],
            textures: MaterialTextureSlots::default(),
            mtoon_params: None,
        }
    }

    pub fn from_asset_material(asset: &crate::asset::MaterialAsset) -> Self {
        Self {
            base_color: asset.base_color,
            mode: match asset.base_mode {
                MaterialMode::Unlit => MaterialShaderMode::Unlit,
                MaterialMode::SimpleLit => MaterialShaderMode::SimpleLit,
                MaterialMode::ToonLike => MaterialShaderMode::ToonLike,
            },
            alpha_mode: asset.alpha_mode,
            double_sided: asset.double_sided,
            toon_ramp_threshold: asset.toon_params.ramp_threshold,
            shadow_softness: asset.toon_params.shadow_softness,
            outline_width: asset.toon_params.outline_width,
            outline_color: asset.toon_params.outline_color,
            textures: MaterialTextureSlots {
                base_color: asset
                    .texture_bindings
                    .base_color_texture
                    .as_ref()
                    .map(|t| MaterialTextureBinding { uri: t.uri.clone() }),
                normal_map: asset
                    .texture_bindings
                    .normal_map_texture
                    .as_ref()
                    .map(|t| MaterialTextureBinding { uri: t.uri.clone() }),
                shade_ramp: None,
                emissive: None,
            },
            mtoon_params: asset.mtoon_params.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// GPU material uniform layout (matches shader `MaterialData`)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MaterialUniform {
    pub base_color: [f32; 4],
    pub alpha_cutoff: f32,
    pub alpha_mode: i32, // 0=opaque, 1=cutout, 2=blend
    pub toon_threshold: f32,
    pub shadow_softness: f32,
    pub shading_mode: i32, // 0=standard, 1=toon
    /// Padding to align struct to 16-byte boundary (std140 layout requirement).
    /// Total size: 48 bytes = 3 * 16-byte alignment.
    pub _pad: [f32; 3],
}

// ---------------------------------------------------------------------------
// MaterialUploader -- creates GPU uniform buffers and descriptor sets
// ---------------------------------------------------------------------------

pub struct MaterialUploader;

impl MaterialUploader {
    pub fn new() -> Self {
        Self
    }

    /// Upload material data to a GPU uniform buffer and return a descriptor set
    /// bound at set 2 of the given pipeline layout.
    ///
    /// The descriptor set contains:
    /// - binding 0: material uniform buffer
    /// - binding 1: combined image sampler (base color texture)
    pub fn upload_to_gpu(
        &mut self,
        request: &MaterialUploadRequest,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        pipeline: &Arc<GraphicsPipeline>,
        texture_view: Arc<ImageView>,
        sampler: Arc<Sampler>,
    ) -> Arc<PersistentDescriptorSet> {
        let alpha_mode_int = match request.alpha_mode {
            AlphaMode::Opaque => 0i32,
            AlphaMode::Mask(_) => 1i32,
            AlphaMode::Blend => 2i32,
        };
        let alpha_cutoff = match request.alpha_mode {
            AlphaMode::Mask(cutoff) => cutoff,
            _ => 0.5,
        };

        let (toon_threshold, shadow_softness) = if let Some(ref mtoon) = request.mtoon_params {
            let threshold = (1.0 - mtoon.shade_toony).max(0.01);
            let softness = (1.0 - mtoon.shade_toony) * 0.5;
            (threshold, softness)
        } else {
            (request.toon_ramp_threshold, request.shadow_softness)
        };

        let shading_mode = match request.mode {
            MaterialShaderMode::ToonLike => 1i32,
            _ => 0i32,
        };

        let uniform_data = MaterialUniform {
            base_color: request.base_color,
            alpha_cutoff,
            alpha_mode: alpha_mode_int,
            toon_threshold,
            shadow_softness,
            shading_mode,
            _pad: [0.0; 3],
        };

        let uniform_buffer = Buffer::from_data(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            uniform_data,
        )
        .expect("failed to create material uniform buffer");

        let layout = pipeline.layout().set_layouts().get(2).unwrap().clone();
        PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            layout,
            [
                WriteDescriptorSet::buffer(0, uniform_buffer),
                WriteDescriptorSet::image_view_sampler(1, texture_view, sampler),
            ],
            [],
        )
        .expect("failed to create material descriptor set")
    }

}
